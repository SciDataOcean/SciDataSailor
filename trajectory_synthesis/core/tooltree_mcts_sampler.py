from __future__ import annotations

import asyncio
import bdb
import hashlib
import json
import math
import random
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..prompt.tooltree_mcts import (
    CANDIDATES_SYSTEM, CANDIDATES_USER_TEMPLATE, DEFAULT_ACTION_TYPES,
    FINAL_ANSWER_SYSTEM_TEMPLATE, FINAL_ANSWER_USER_TEMPLATE,
    POST_JUDGE_TEMPLATE, PRE_JUDGE_TEMPLATE,
    REFINE_SYSTEM, REFINE_USER_TEMPLATE,
    STRATEGY_SYSTEM, STRATEGY_USER_TEMPLATE,
    format_action_catalog,
)
from .config import SynthesisConfig
from .models import TrajectoryNode


def _uct_value(
    parent: TrajectoryNode,
    child: TrajectoryNode,
    siblings: List[TrajectoryNode],
    exploration_lambda: float,
    fpu_reduction: float = 0.1,
    depth_bonus_weight: float = 0.05,
    visit_decay: str = "sqrt",
) -> float:
    """DF-FPU: Dual-Feedback First-Play Urgency for dataset-exploration MCTS."""
    n_parent = max(parent.mcts_visit_count, 1)
    log_term = math.sqrt(math.log(n_parent + 1.0))
    depth_bonus = depth_bonus_weight * float(child.depth)

    if child.mcts_visit_count == 0:
        # (2) Sibling calibration: observe r_post - r_pre bias locally.
        visited_sibs = [s for s in siblings if s.mcts_visit_count > 0]
        if visited_sibs:
            bias = sum(s.mcts_q_value - (s.mcts_r_pre or 0.0) for s in visited_sibs) / len(visited_sibs)
        else:
            bias = 0.0
        # (1) Anchor FPU on r_pre (a direct Q-estimate), calibrated.
        fpu_q = max((child.mcts_r_pre or 0.0) + bias - fpu_reduction, 0.0)
        # Same exploration-bonus shape as visited case with (1+v) = 1.
        bonus = exploration_lambda * (child.mcts_r_pre or 0.0) * log_term
        return fpu_q + bonus + depth_bonus

    # (4) Softened visit decay for visited children.
    if visit_decay == "sqrt":
        denom = math.sqrt(1.0 + child.mcts_visit_count)
    else:
        denom = 1.0 + child.mcts_visit_count

    exploit = child.mcts_q_value
    bonus = (exploration_lambda * (child.mcts_r_pre or 0.0) * log_term / denom)
    return exploit + bonus + depth_bonus


def _prior_entropy_norm(
    priors: List[float],
    temperature: float = 0.3,
) -> float:
    n = len(priors)
    if n <= 1 or temperature <= 0:
        return 0.0
    max_p = max(priors)
    exps = [math.exp((p - max_p) / temperature) for p in priors]
    total = sum(exps)
    if total <= 0:
        return 0.0
    probs = [e / total for e in exps]
    raw_H = -sum(p * math.log(p + 1e-12) for p in probs)
    norm_H = raw_H / math.log(n)
    return max(0.0, min(1.0, norm_H))


def _k_from_entropy(
    norm_H: float,
    max_k: int,
    min_k: int = 1,
    n_candidates: Optional[int] = None,
) -> int:
    """Linearly interpolate the branching factor ``k`` from a [0, 1] entropy."""
    if n_candidates is not None and n_candidates <= 0:
        return 0
    cap = max_k if n_candidates is None else min(max_k, n_candidates)
    k_cap = max(1, cap)
    k_floor = max(1, min(min_k, k_cap))
    if k_cap <= k_floor:
        return k_floor
    norm_H = max(0.0, min(1.0, float(norm_H)))
    k_float = k_floor + (k_cap - k_floor) * norm_H
    return max(k_floor, min(k_cap, int(round(k_float))))


def _entropy_based_branching(
    priors: List[float],
    max_k: int,
    min_k: int = 1,
    temperature: float = 0.3,
) -> int:
    """Back-compat wrapper: dynamic k from prior-only entropy."""
    n = len(priors)
    if n == 0:
        return 0
    norm_H = _prior_entropy_norm(priors, temperature=temperature)
    return _k_from_entropy(norm_H, max_k=max_k, min_k=min_k, n_candidates=n)


def _apply_strategy_diversity(
    ranked_items: List[Dict[str, Any]],
    target_k: int,
    action_type_key: str = "action_type",
    score_key: str = "r_pre",
) -> List[Dict[str, Any]]:
    if target_k <= 0:
        return []
    if target_k >= len(ranked_items):
        return list(ranked_items)

    def _type_of(item: Dict[str, Any]) -> str:
        t = item.get(action_type_key)
        if isinstance(t, str) and t.strip():
            return t.strip()
        return "__untyped__"

    # First pass: one-per-type by descending score (ranked_items is assumed sorted).
    picked: List[Dict[str, Any]] = []
    seen_types: set = set()
    remaining: List[Dict[str, Any]] = []
    for item in ranked_items:
        if len(picked) >= target_k:
            remaining.append(item)
            continue
        t = _type_of(item)
        if t in seen_types:
            remaining.append(item)
            continue
        picked.append(item)
        seen_types.add(t)

    # Second pass: fill remaining slots by score.
    for item in remaining:
        if len(picked) >= target_k:
            break
        picked.append(item)

    # Keep original ranking (by score) stable within the selection.
    picked.sort(key=lambda it: it.get(score_key, 0.0), reverse=True)
    return picked


class ToolTreeMCTSSampler:
    def __init__(
        self,
        llm_client,
        python_tool,
        config: SynthesisConfig,
        dataset_path: str = "",
    ):
        self.llm_client = llm_client
        self.python_tool = python_tool
        self.config = config
        self.dataset_path = dataset_path or ""

        self.nodes: Dict[str, TrajectoryNode] = {}
        self.root_id: Optional[str] = None
        self.last_run_stats: Dict[str, Any] = {}
        self._llm_usage_stats: Dict[str, int] = {}

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(self.config, name, default)

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return default
        return default

    async def sample_trajectory_tree(
        self,
        seed_data: str,
        seed_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, TrajectoryNode]:
        if seed_kwargs is None:
            seed_kwargs = {}
        run_start = time.perf_counter()
        self._reset_llm_usage_stats()

        print(f"\n{'=' * 60}")
        print("ToolTree-style MCTS Trajectory Tree (dataset exploration)")
        print(f"Seed: {seed_data[:120]}...")
        print(f"Dataset path: {self.dataset_path}")
        print(
            f"mcts_rollouts={self.config.mcts_rollouts} "
            f"branching={self.config.mcts_branching} "
            f"lambda={self.config.mcts_lambda} "
            f"tau_pre={self.config.mcts_tau_pre} tau_post={self.config.mcts_tau_post} "
            f"fpu_reduction={self._cfg('mcts_fpu_reduction', 0.1)} "
            f"depth_bonus={self._cfg('mcts_depth_bonus', 0.05)} "
            f"visit_decay={self._cfg('mcts_visit_decay', 'sqrt')} "
            f"dynamic_branching={self._cfg('mcts_dynamic_branching', True)} "
            f"hierarchical={self._cfg('mcts_hierarchical_actions', False)} "
            f"strategy_diversity={self._cfg('mcts_strategy_diversity', True)}"
        )
        print(f"{'=' * 60}\n")

        self.nodes = {}
        self.root_id = self._new_id()
        root = TrajectoryNode(
            node_id=self.root_id,
            thought="Root: start exploration",
            action=None,
            observation=self._root_observation(seed_data),
            parent_id=None,
            children_ids=[],
            depth=0,
            mcts_expandable=True,
        )
        self.nodes[self.root_id] = root

        used_code_hashes: set = set()

        for rollout_idx in range(self.config.mcts_rollouts):
            path = self._select_path_to_expand()
            leaf = path[-1]

            if leaf.depth >= self.config.mcts_max_depth:
                continue
            if leaf.children_ids:
                # Leaf for expansion must have no children; otherwise we should have descended.
                continue

            new_children = await self._expand_leaf(
                leaf,
                seed_data,
                seed_kwargs,
                used_code_hashes,
                rollout_idx,
            )
            if not new_children:
                continue

            for child in new_children:
                reward = float(child.mcts_r_post or 0.0)
                self._backpropagate(path + [child], reward)

                post_pruned = reward < self.config.mcts_tau_post
                if post_pruned:
                    child.mcts_expandable = False

                print(
                    f"\033[36m[MCTS {rollout_idx + 1}/{self.config.mcts_rollouts}]\033[0m "
                    f"depth={child.depth} r_pre={child.mcts_r_pre:.3f} "
                    f"r_post={reward:.3f} expand={'Y' if child.mcts_expandable else 'N'}"
                )

        await self._constrain_leaf_final_answers(seed_data)
        elapsed_sec = time.perf_counter() - run_start
        self.last_run_stats = {
            "elapsed_seconds": round(elapsed_sec, 4),
            "llm_calls": self._llm_usage_stats.get("calls", 0),
            "llm_prompt_tokens": self._llm_usage_stats.get("prompt_tokens", 0),
            "llm_completion_tokens": self._llm_usage_stats.get("completion_tokens", 0),
            "llm_total_tokens": self._llm_usage_stats.get("total_tokens", 0),
            "llm_usage_exact_calls": self._llm_usage_stats.get("exact_calls", 0),
            "llm_usage_estimated_calls": self._llm_usage_stats.get("estimated_calls", 0),
            "nodes": len(self.nodes),
        }
        print(f"\nTree complete. Nodes={len(self.nodes)}")
        print(
            "[Tree stats] "
            f"time={elapsed_sec:.2f}s "
            f"llm_calls={self.last_run_stats['llm_calls']} "
            f"total_tokens={self.last_run_stats['llm_total_tokens']} "
            f"(prompt={self.last_run_stats['llm_prompt_tokens']}, "
            f"completion={self.last_run_stats['llm_completion_tokens']}, "
            f"exact_calls={self.last_run_stats['llm_usage_exact_calls']}, "
            f"estimated_calls={self.last_run_stats['llm_usage_estimated_calls']})"
        )
        return self.nodes

    def _reset_llm_usage_stats(self) -> None:
        self._llm_usage_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "exact_calls": 0,
            "estimated_calls": 0,
        }

    @staticmethod
    def _extract_usage_tokens(result: Any) -> Optional[Dict[str, int]]:
        if not isinstance(result, dict):
            return None
        usage = result.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if prompt is None and completion is None and total is None:
            return None
        try:
            prompt_i = int(prompt or 0)
            completion_i = int(completion or 0)
            total_i = int(total if total is not None else (prompt_i + completion_i))
        except (TypeError, ValueError):
            return None
        return {
            "prompt_tokens": max(0, prompt_i),
            "completion_tokens": max(0, completion_i),
            "total_tokens": max(0, total_i),
        }

    @staticmethod
    def _estimate_tokens_from_chars(chars: int) -> int:
        if chars <= 0:
            return 0
        # Heuristic fallback when backend doesn't return usage.
        return max(1, chars // 4)

    def _accumulate_llm_usage(self, messages: List[Dict[str, str]], result: Any) -> None:
        self._llm_usage_stats["calls"] += 1
        usage = self._extract_usage_tokens(result)
        if usage is not None:
            self._llm_usage_stats["prompt_tokens"] += usage["prompt_tokens"]
            self._llm_usage_stats["completion_tokens"] += usage["completion_tokens"]
            self._llm_usage_stats["total_tokens"] += usage["total_tokens"]
            self._llm_usage_stats["exact_calls"] += 1
            return

        input_chars = 0
        for msg in messages:
            input_chars += len(str(msg.get("content", "")))

        output_text = ""
        if isinstance(result, dict):
            output_text = str(result.get("text") or "")
        elif result is not None:
            output_text = str(result)

        prompt_est = self._estimate_tokens_from_chars(input_chars)
        completion_est = self._estimate_tokens_from_chars(len(output_text))
        total_est = prompt_est + completion_est
        self._llm_usage_stats["prompt_tokens"] += prompt_est
        self._llm_usage_stats["completion_tokens"] += completion_est
        self._llm_usage_stats["total_tokens"] += total_est
        self._llm_usage_stats["estimated_calls"] += 1

    def _root_observation(self, seed_data: str) -> str:
        obs = f"Starting point: {seed_data}"
        if self.dataset_path:
            obs += f"\nDataset root path: {self.dataset_path}"
        return obs

    def _select_path_to_expand(self) -> List[TrajectoryNode]:
        assert self.root_id is not None
        node = self.nodes[self.root_id]
        path = [node]

        while node.children_ids:
            expandable_ids = [
                cid
                for cid in node.children_ids
                if self.nodes[cid].mcts_expandable
            ]
            if not expandable_ids:
                break

            parent = node
            children = [self.nodes[cid] for cid in expandable_ids]
            assert parent.node_id is not None
            fpu = self._cfg("mcts_fpu_reduction", 0.1)
            depth_bonus = self._cfg("mcts_depth_bonus", 0.05)
            visit_decay = self._cfg("mcts_visit_decay", "sqrt")
            best = max(
                children,
                key=lambda ch: _uct_value(
                    parent,
                    ch,
                    children,
                    self.config.mcts_lambda,
                    fpu_reduction=fpu,
                    depth_bonus_weight=depth_bonus,
                    visit_decay=visit_decay,
                )
                + random.random() * 1e-6,
            )
            # Tiny jitter breaks ties without changing semantics much.
            node = best
            path.append(node)

        return path

    async def _expand_leaf(
        self,
        parent: TrajectoryNode,
        seed_data: str,
        seed_kwargs: Dict[str, Any],
        used_hashes: set,
        rollout_idx: int,
    ) -> List[TrajectoryNode]:
        k = max(1, self.config.mcts_num_candidates)
        hierarchical = bool(self._cfg("mcts_hierarchical_actions", False))

        if hierarchical:
            scored, token_H = await self._expand_hierarchical(
                parent, seed_data, seed_kwargs, used_hashes, k, rollout_idx
            )
        else:
            scored, token_H = await self._expand_flat(
                parent, seed_data, seed_kwargs, used_hashes, k, rollout_idx
            )

        if not scored:
            return []

        # Dynamic (entropy-based) branching: combine prior-softmax entropy
        # with ARPO-style token-level entropy from the proposal LLM call.
        priors = [item["r_pre"] for item in scored]
        max_k = max(1, self.config.mcts_branching)
        if self._cfg("mcts_dynamic_branching", True):
            H_step, H_prior, H_token = self._combined_step_entropy(priors, token_H)
            k_dyn = _k_from_entropy(
                H_step,
                max_k=max_k,
                min_k=self._cfg("mcts_min_branching", 1),
                n_candidates=len(scored),
            )
        else:
            H_step = H_prior = 0.0
            H_token = token_H if token_H is not None else 0.0
            k_dyn = min(max_k, len(scored))

        scored.sort(key=lambda it: it["r_pre"], reverse=True)

        if self._cfg("mcts_strategy_diversity", True):
            chosen = _apply_strategy_diversity(
                scored, k_dyn, action_type_key="action_type", score_key="r_pre"
            )
        else:
            chosen = scored[:k_dyn]

        print(
            f"\033[33m[MCTS expand]\033[0m "
            f"survivors={len(scored)} k_dyn={k_dyn} "
            f"H_step={H_step:.3f} H_prior={H_prior:.3f} H_token={H_token:.3f} "
            f"chosen_types={[it.get('action_type') for it in chosen]}"
        )

        return await self._materialize_children(
            parent, seed_data, chosen, used_hashes
        )

    async def _expand_flat(
        self,
        parent: TrajectoryNode,
        seed_data: str,
        seed_kwargs: Dict[str, Any],
        used_hashes: set,
        k: int,
        rollout_idx: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        candidates, token_H = await self._propose_candidates(
            parent, seed_data, seed_kwargs, k, rollout_idx
        )
        if not candidates:
            return [], token_H

        unique: List[Dict[str, Any]] = []
        seen_local_hashes: set = set()
        for thought, code, action_type in candidates:
            h = hashlib.md5(code.encode("utf-8")).hexdigest()
            if h in used_hashes or h in seen_local_hashes:
                continue
            seen_local_hashes.add(h)
            unique.append(
                {"thought": thought, "code": code, "action_type": action_type}
            )

        if not unique:
            return [], token_H

        pre_tasks = [
            self._pre_judge(seed_data, parent, item["code"]) for item in unique
        ]
        pre_results = await asyncio.gather(*pre_tasks)

        survivors: List[Dict[str, Any]] = []
        for item, (r_pre, _) in zip(unique, pre_results):
            if r_pre < self.config.mcts_tau_pre:
                continue
            item["r_pre"] = float(r_pre)
            survivors.append(item)
        return survivors, token_H

    async def _expand_hierarchical(
        self,
        parent: TrajectoryNode,
        seed_data: str,
        seed_kwargs: Dict[str, Any],
        used_hashes: set,
        k: int,
        rollout_idx: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        strategies, token_H = await self._propose_strategies(
            parent, seed_data, seed_kwargs, k, rollout_idx
        )
        if not strategies:
            return [], token_H

        # Filter by tau_pre on the strategy's self-rated prior BEFORE paying
        # for the (more expensive) refinement LLM call.
        kept = [s for s in strategies if s["prior"] >= self.config.mcts_tau_pre]
        if not kept:
            return [], token_H

        # Trim to at most ``k`` strategies to bound refinement cost.
        kept.sort(key=lambda s: s["prior"], reverse=True)
        kept = kept[:k]

        refine_tasks = [
            self._refine_strategy_to_code(parent, seed_data, s) for s in kept
        ]
        refinements = await asyncio.gather(*refine_tasks, return_exceptions=True)

        survivors: List[Dict[str, Any]] = []
        seen_local_hashes: set = set()
        for strat, ref in zip(kept, refinements):
            if isinstance(ref, Exception):
                print(
                    f"  [refine err] action_type={strat.get('action_type')!r} "
                    f"error={ref!s}"
                )
                continue
            if not isinstance(ref, dict):
                continue
            code = str(ref.get("code", "")).strip()
            if not code:
                continue
            h = hashlib.md5(code.encode("utf-8")).hexdigest()
            if h in used_hashes or h in seen_local_hashes:
                continue
            seen_local_hashes.add(h)

            # r_pre = strategy prior * refinement confidence (clipped to [0, 1]).
            conf = float(ref.get("confidence", 1.0) or 0.0)
            conf = max(0.0, min(1.0, conf))
            r_pre = max(0.0, min(1.0, float(strat["prior"]) * conf))
            if r_pre < self.config.mcts_tau_pre:
                continue

            survivors.append(
                {
                    "thought": f"[{strat['action_type']}] {strat['thought']}",
                    "code": code,
                    "action_type": strat["action_type"],
                    "r_pre": r_pre,
                }
            )
        return survivors, token_H

    async def _materialize_children(
        self,
        parent: TrajectoryNode,
        seed_data: str,
        chosen: List[Dict[str, Any]],
        used_hashes: set,
    ) -> List[TrajectoryNode]:
        out: List[TrajectoryNode] = []
        for item in chosen:
            code = item["code"]
            code_hash = hashlib.md5(code.encode("utf-8")).hexdigest()
            if code_hash in used_hashes:
                continue
            used_hashes.add(code_hash)

            obs = await self._execute_python(code)
            r_post, _ = await self._post_judge(seed_data, code, obs)

            action_payload: Dict[str, Any] = {
                "tool_name": "python_interpreter",
                "parameters": {"code": code},
            }
            if item.get("action_type"):
                action_payload["action_type"] = item["action_type"]

            child_id = self._new_id()
            child = TrajectoryNode(
                node_id=child_id,
                thought=item["thought"],
                action=action_payload,
                action_type=item.get("action_type"),
                observation=obs,
                parent_id=parent.node_id,
                children_ids=[],
                depth=parent.depth + 1,
                mcts_visit_count=0,
                mcts_q_value=0.0,
                mcts_r_pre=float(item["r_pre"]),
                mcts_r_post=r_post,
                mcts_expandable=True,
            )
            self.nodes[child_id] = child
            parent.children_ids.append(child_id)
            out.append(child)
        return out

    def _backpropagate(self, path: List[TrajectoryNode], reward: float) -> None:
        """Leaf-to-root incremental mean (ToolTree backprop on r_post)."""
        for node in reversed(path):
            n = node.mcts_visit_count + 1
            node.mcts_visit_count = n
            node.mcts_q_value += (reward - node.mcts_q_value) / n

    def _resolve_action_types(self) -> List[str]:
        configured = list(getattr(self.config, "mcts_action_types", []) or [])
        return configured if configured else list(DEFAULT_ACTION_TYPES)

    async def _propose_candidates(
        self,
        parent: TrajectoryNode,
        seed_data: str,
        seed_kwargs: Dict[str, Any],
        k: int,
        rollout_idx: int,
    ) -> Tuple[List[Tuple[str, str, Optional[str]]], Optional[float]]:
        ctx = self._path_summary_for_prompt(parent)
        action_types = self._resolve_action_types()
        catalog = format_action_catalog(action_types)
        user = CANDIDATES_USER_TEMPLATE.format(
            seed_data=seed_data,
            dataset_path=self.dataset_path or "/data",
            context_summary=ctx,
            k=k,
            action_catalog=catalog,
        )
        if self.config.sampling_tips:
            user += f"\n\n[Exploration strategy]\n{self.config.sampling_tips}"

        text, logprobs_info = await self._call_llm_with_logprobs(
            messages=[
                {"role": "system", "content": CANDIDATES_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=min(0.85, 0.55 + 0.03 * rollout_idx),
        )
        token_H = self._token_entropy_norm(
            logprobs_info,
            max_tokens=int(self._cfg("mcts_entropy_token_max_tokens", 20)),
            top_logprobs=int(self._cfg("mcts_entropy_top_logprobs", 5)),
        ) if logprobs_info is not None else None

        arr = self._parse_json_array(text)
        out: List[Tuple[str, str, Optional[str]]] = []
        if not isinstance(arr, list):
            return out, token_H

        allowed = set(action_types)
        for item in arr:
            if not isinstance(item, dict):
                continue
            thought = str(item.get("thought", "")).strip()
            code = str(item.get("code", "")).strip()
            raw_type = str(item.get("action_type", "")).strip()
            action_type: Optional[str] = raw_type if raw_type in allowed else None
            if thought and code:
                out.append((thought, code, action_type))
            if len(out) >= k:
                break
        return out[:k], token_H

    async def _propose_strategies(
        self,
        parent: TrajectoryNode,
        seed_data: str,
        seed_kwargs: Dict[str, Any],
        k: int,
        rollout_idx: int,
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        ctx = self._path_summary_for_prompt(parent)
        action_types = self._resolve_action_types()
        catalog = format_action_catalog(action_types)
        user = STRATEGY_USER_TEMPLATE.format(
            seed_data=seed_data,
            dataset_path=self.dataset_path or "/data",
            context_summary=ctx,
            k=k,
            action_catalog=catalog,
        )
        if self.config.sampling_tips:
            user += f"\n\n[Exploration strategy]\n{self.config.sampling_tips}"

        text, logprobs_info = await self._call_llm_with_logprobs(
            messages=[
                {"role": "system", "content": STRATEGY_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=min(0.85, 0.55 + 0.03 * rollout_idx),
        )
        token_H = self._token_entropy_norm(
            logprobs_info,
            max_tokens=int(self._cfg("mcts_entropy_token_max_tokens", 20)),
            top_logprobs=int(self._cfg("mcts_entropy_top_logprobs", 5)),
        ) if logprobs_info is not None else None

        arr = self._parse_json_array(text)
        out: List[Dict[str, Any]] = []
        if not isinstance(arr, list):
            return out, token_H

        allowed = set(action_types)
        for item in arr:
            if not isinstance(item, dict):
                continue
            action_type = str(item.get("action_type", "")).strip()
            thought = str(item.get("thought", "")).strip()
            prior = self._coerce_float(item.get("prior", 0.0), default=0.0)
            prior = max(0.0, min(1.0, prior))
            if action_type not in allowed or not thought:
                continue
            out.append(
                {
                    "action_type": action_type,
                    "thought": thought,
                    "prior": prior,
                }
            )
            if len(out) >= k:
                break
        return out[:k], token_H

    async def _refine_strategy_to_code(
        self,
        parent: TrajectoryNode,
        seed_data: str,
        strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        ctx = self._path_summary_for_prompt(parent)
        user = REFINE_USER_TEMPLATE.format(
            seed_data=seed_data,
            dataset_path=self.dataset_path or "/data",
            context_summary=ctx,
            action_type=strategy.get("action_type", ""),
            strategy_thought=strategy.get("thought", ""),
        )
        text = await self._call_llm(
            messages=[
                {"role": "system", "content": REFINE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
        )
        obj = self._first_json_object(text)
        if not isinstance(obj, dict):
            return {}
        code = str(obj.get("code", "")).strip()
        conf = self._coerce_float(obj.get("confidence", 1.0), default=1.0)
        conf = max(0.0, min(1.0, conf))
        return {"code": code, "confidence": conf}

    def _path_summary_for_prompt(self, node: TrajectoryNode) -> str:
        """Compact ancestor chain for planner/judge context.

        Uses a token-budget-friendly layout:
        - older steps are compressed into short one-line summaries;
        - only the most recent few steps keep (truncated) code + observation.
        """
        chain: List[TrajectoryNode] = []
        cur: Optional[TrajectoryNode] = node
        while cur is not None:
            chain.append(cur)
            if cur.parent_id is None:
                break
            cur = self.nodes.get(cur.parent_id)
        chain.reverse()
        steps: List[Tuple[int, TrajectoryNode, str, str]] = []
        for i, n in enumerate(chain):
            if n.action and n.action.get("tool_name") == "python_interpreter":
                code = str((n.action.get("parameters") or {}).get("code", "") or "")
                obs = str(n.observation or "")
                steps.append((i, n, code, obs))

        if not steps:
            return "(no prior code steps yet)"

        recent_raw_steps = max(1, int(self._cfg("mcts_context_recent_steps", 2)))
        max_code_chars = max(200, int(self._cfg("mcts_context_max_code_chars", 1200)))
        max_obs_chars = max(300, int(self._cfg("mcts_context_max_obs_chars", 2500)))
        max_total_chars = max(1000, int(self._cfg("mcts_context_max_total_chars", 12000)))

        split_idx = max(0, len(steps) - recent_raw_steps)
        old_steps = steps[:split_idx]
        recent_steps = steps[split_idx:]

        compressed: List[str] = []
        for i, n, code, obs in old_steps:
            code_short = self._compact_line(code, max_len=180)
            obs_short = self._compact_line(obs, max_len=240)
            action_type = (n.action_type or "").strip() or "unknown"
            compressed.append(
                f"- Step {i} [{action_type}] code={code_short} | obs={obs_short}"
            )

        detailed: List[str] = []
        for i, n, code, obs in recent_steps:
            action_type = (n.action_type or "").strip() or "unknown"
            code_clip = self._clip_text(code, max_code_chars)
            obs_clip = self._clip_text(obs, max_obs_chars)
            detailed.append(
                f"Step {i} [{action_type}] code:\n{code_clip}\n\nobservation:\n{obs_clip}"
            )

        sections: List[str] = []
        if compressed:
            sections.append("[Earlier step summaries]\n" + "\n".join(compressed))
        if detailed:
            sections.append("[Recent detailed steps]\n" + "\n\n".join(detailed))
        blob = "\n\n".join(sections)
        if len(blob) > max_total_chars:
            blob = self._clip_text(blob, max_total_chars)
        return blob

    @staticmethod
    def _compact_line(text: str, max_len: int) -> str:
        one_line = re.sub(r"\s+", " ", (text or "")).strip()
        return ToolTreeMCTSSampler._clip_text(one_line, max_len)

    @staticmethod
    def _clip_text(text: str, max_len: int) -> str:
        if not text:
            return ""
        if max_len <= 0 or len(text) <= max_len:
            return text
        if max_len < 80:
            return text[:max_len]
        keep = (max_len - 20) // 2
        return f"{text[:keep]}\n...<truncated>...\n{text[-keep:]}"

    async def _pre_judge(
        self,
        seed_data: str,
        parent: TrajectoryNode,
        code: str,
    ) -> Tuple[float, str]:
        path_summary = self._path_summary_for_prompt(parent)
        prompt = PRE_JUDGE_TEMPLATE.format(
            seed_data=seed_data[:2000],
            path_summary=path_summary,
            code=code[:6000],
        )
        text = await self._call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.mcts_judge_temperature,
        )
        return self._parse_judge_score(text)

    async def _post_judge(
        self,
        seed_data: str,
        code: str,
        observation: str,
    ) -> Tuple[float, str]:
        prompt = POST_JUDGE_TEMPLATE.format(
            seed_data=seed_data[:2000],
            code=code[:4000],
            observation=(observation or "")[:8000],
        )
        text = await self._call_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.mcts_judge_temperature,
        )
        return self._parse_judge_score(text)

    def _parse_judge_score(self, text: str) -> Tuple[float, str]:
        score = 0.0
        reason = ""
        obj = self._first_json_object(text)
        if isinstance(obj, dict):
            raw = obj.get("score", 0)
            score = self._coerce_float(raw, default=0.0)
            reason = str(obj.get("reason", "")).strip()
        score = max(0.0, min(1.0, score))
        return score, reason

    @staticmethod
    def _first_json_object(text: str) -> Any:
        if not text:
            return None
        t = re.sub(
            r"<think>.*?</think>\s*",
            "",
            text,
            flags=re.DOTALL,
        )
        start = t.find("{")
        if start == -1:
            return None
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _parse_json_array(text: str) -> Any:
        if not text:
            return None
        t = re.sub(
            r"<think>.*?</think>\s*",
            "",
            text,
            flags=re.DOTALL,
        )
        start = t.find("[")
        if start == -1:
            return None
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    async def _execute_python(self, code: str) -> str:
        if not code.strip():
            return "Error: empty code."
        try:
            raw_obs = await self.python_tool.execute(
                code, timeout=self.config.sandbox_timeout
            )
        except Exception as exc:
            return f"Error executing python_interpreter: {exc}"
        return self._truncate_observation(raw_obs or "")

    def _truncate_observation(self, observation: str) -> str:
        """Cap a tool observation so it never balloons the next prompt.

        If ``len(observation) > mcts_obs_max_chars``, keep the first
        ``mcts_obs_head_chars`` and last ``mcts_obs_tail_chars`` verbatim and
        insert ``mcts_obs_truncate_notice`` in between, followed by a short
        summary of how many chars were dropped.
        """
        if not isinstance(observation, str):
            observation = str(observation)
        max_chars = int(self._cfg("mcts_obs_max_chars", 6000))
        if max_chars <= 0 or len(observation) <= max_chars:
            return observation

        head_chars = max(0, int(self._cfg("mcts_obs_head_chars", 4000)))
        tail_chars = max(0, int(self._cfg("mcts_obs_tail_chars", 1500)))
        notice = str(
            self._cfg(
                "mcts_obs_truncate_notice",
                "... the response is too long to show ...",
            )
        )

        if head_chars + tail_chars >= len(observation):
            return observation
        if head_chars + tail_chars > max_chars:
            head_chars = max(0, max_chars - tail_chars)

        head = observation[:head_chars]
        tail = observation[-tail_chars:] if tail_chars > 0 else ""
        dropped = max(0, len(observation) - head_chars - tail_chars)
        summary = (
            f"\n\n{notice} "
            f"(original_length={len(observation)} chars, "
            f"dropped={dropped} chars)\n\n"
        )
        return head + summary + tail

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        for attempt in range(4):
            try:
                result = await self.llm_client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                )
                self._accumulate_llm_usage(messages, result)
                if not result:
                    return ""
                if isinstance(result, dict):
                    return result.get("text") or ""
                return str(result)
            except bdb.BdbQuit:
                raise
            except Exception as exc:
                if attempt >= 3:
                    print(f"  LLM call failed: {exc}")
                    return ""
                await asyncio.sleep(0.4 * (2**attempt))
        return ""

    async def _call_llm_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        if not self._cfg("mcts_entropy_use_token_logprobs", True):
            text = await self._call_llm(messages, temperature)
            return text, None

        top_k = int(self._cfg("mcts_entropy_top_logprobs", 5))
        for attempt in range(4):
            try:
                result = await self.llm_client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=top_k,
                )
                if not result:
                    return "", None
                if isinstance(result, dict):
                    return (result.get("text") or ""), result.get("logprobs")
                return str(result), None
            except bdb.BdbQuit:
                raise
            except Exception as exc:
                if attempt >= 3:
                    print(f"  LLM call (with logprobs) failed: {exc}")
                    return "", None
                await asyncio.sleep(0.4 * (2**attempt))
        return "", None

    @staticmethod
    def _calc_entropy(logprobs: List[float]) -> float:
        """ARPO-style Shannon entropy over a flat list of logprobs.

        Mirrors ``vLLMRolloutWithTools._calc_entropy``:
            H = - sum_i p_i * logp_i, where p_i = exp(logp_i)
        """
        if not logprobs:
            return 0.0
        p_list = [math.exp(l) for l in logprobs]
        return -sum(p * l for p, l in zip(p_list, logprobs))

    @classmethod
    def _token_entropy_norm(
        cls,
        logprobs_info: Optional[List[Dict[str, Any]]],
        max_tokens: int = 20,
        top_logprobs: int = 10,
    ) -> float:
        if not logprobs_info:
            return 0.0
        window = logprobs_info[: max(1, max_tokens)]
        flat: List[float] = []
        actual_ks: List[int] = []  # per-token count of logprobs actually used
        for tok in window:
            tops = tok.get("top_logprobs") or []
            if not tops:
                lp = tok.get("logprob")
                if lp is not None:
                    flat.append(float(lp))
                    actual_ks.append(1)
                continue
            cnt = 0
            for entry in tops:
                lp = entry.get("logprob")
                if lp is None:
                    continue
                flat.append(float(lp))
                cnt += 1
            if cnt > 0:
                actual_ks.append(cnt)
        if not flat or not actual_ks:
            return 0.0
        raw_H = cls._calc_entropy(flat)
        eff_tokens = len(actual_ks)
        # Average top-K width actually returned by the backend.
        avg_k = sum(actual_ks) / eff_tokens
        if avg_k > 1.0:
            denom = eff_tokens * math.log(avg_k)
        else:
            # K == 1 fallback: flat entries are the sampled-token logprobs.
            # Max of ``-p * log p`` is ``1/e`` per position.
            denom = eff_tokens * (1.0 / math.e)
        if denom <= 0:
            return 0.0
        return max(0.0, min(1.0, raw_H / denom))

    def _combined_step_entropy(
        self,
        priors: List[float],
        token_entropy_norm: Optional[float],
    ) -> Tuple[float, float, float]:
        temperature = self._cfg("mcts_entropy_temperature", 0.3)
        H_prior = _prior_entropy_norm(priors, temperature=temperature)
        H_token = (
            max(0.0, min(1.0, float(token_entropy_norm)))
            if token_entropy_norm is not None
            else 0.0
        )
        w_prior = float(self._cfg("mcts_entropy_prior_weight", 0.5))
        w_token = float(self._cfg("mcts_entropy_token_weight", 0.5))
        if token_entropy_norm is None:
            w_token = 0.0
        total = w_prior + w_token
        if total <= 0:
            return H_prior, H_prior, H_token
        H_step = (w_prior * H_prior + w_token * H_token) / total
        return max(0.0, min(1.0, H_step)), H_prior, H_token

    @staticmethod
    def _new_id() -> str:
        return f"node_{uuid.uuid4().hex[:8]}"

    async def _constrain_leaf_final_answers(self, seed_data: str) -> None:
        pending: List[TrajectoryNode] = []
        for node in self.nodes.values():
            if node.children_ids:
                continue
            if node.parent_id is None:
                continue
            if (node.final_answer or "").strip():
                continue
            if not node.action:
                # Already an answer-only node (e.g. resumed from disk).
                continue
            tool_name = str(node.action.get("tool_name", "")).upper()
            if tool_name == "STOP":
                continue
            pending.append(node)

        if not pending:
            return

        results = await asyncio.gather(
            *(self._generate_final_answer(seed_data, leaf) for leaf in pending),
            return_exceptions=True,
        )

        for leaf, result in zip(pending, results):
            if isinstance(result, Exception) or not isinstance(result, tuple):
                reasoning, answer_body = "", ""
            else:
                reasoning, answer_body = result

            if not answer_body:
                answer_body = (
                    "The final answer is \\[ \\boxed{answer here} \\]"
                )

            if not reasoning.strip():
                reasoning = self._fallback_final_think(answer_body)

            answer_id = self._new_id()
            answer_node = TrajectoryNode(
                node_id=answer_id,
                thought=reasoning,
                action=None,
                action_type=None,
                observation="",
                parent_id=leaf.node_id,
                children_ids=[],
                depth=leaf.depth + 1,
                mcts_visit_count=leaf.mcts_visit_count,
                mcts_q_value=leaf.mcts_q_value,
                mcts_r_pre=leaf.mcts_r_pre,
                mcts_r_post=leaf.mcts_r_post,
                mcts_expandable=False,
                final_answer=answer_body,
            )
            self.nodes[answer_id] = answer_node
            leaf.children_ids.append(answer_id)
            leaf.mcts_expandable = False

    async def _generate_final_answer(
        self,
        seed_data: str,
        leaf: TrajectoryNode,
    ) -> Tuple[str, str]:
        trajectory_text = self._render_path_as_tagged_logs(leaf)
        max_python_times = max(1, int(self.config.mcts_max_depth))
        system_prompt = FINAL_ANSWER_SYSTEM_TEMPLATE.format(
            max_python_times=max_python_times,
            input_path=self.dataset_path or "/data",
        )
        user_prompt = FINAL_ANSWER_USER_TEMPLATE.format(
            seed_data=seed_data,
            trajectory_text=trajectory_text,
        )
        text = await self._call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.mcts_judge_temperature,
        )
        return self._parse_final_think_answer(text)

    def _render_path_as_tagged_logs(self, leaf: TrajectoryNode) -> str:
        chain: List[TrajectoryNode] = []
        cur: Optional[TrajectoryNode] = leaf
        while cur is not None:
            chain.append(cur)
            if cur.parent_id is None:
                break
            cur = self.nodes.get(cur.parent_id)
        chain.reverse()

        fragments: List[str] = []
        for n in chain:
            if n.parent_id is None:
                continue
            fragments.extend(n.build_tagged_logs())
        return "\n".join(fragments) if fragments else "(no prior steps)"

    @staticmethod
    def _fallback_final_think(answer_body: str) -> str:
        cleaned = re.sub(r"\s+", " ", (answer_body or "").strip())
        cleaned = re.sub(r"^[#>*\-\s]+", "", cleaned)
        if not cleaned:
            return (
                "I have gathered enough metadata from the previous exploration "
                "steps; let me compile everything into the final answer."
            )
        preview = cleaned[:200].rstrip()
        return (
            "Now I have all the information needed to answer the question. "
            f"Let me compile the metadata summary: {preview}"
            + ("..." if len(cleaned) > len(preview) else "")
        )

    @staticmethod
    def _parse_final_think_answer(text: str) -> Tuple[str, str]:
        if not text:
            return "", ""

        think_m = re.search(
            r"<think>\s*(.*?)\s*</think>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        reasoning = think_m.group(1).strip() if think_m else ""

        answer_m = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if answer_m:
            answer_body = answer_m.group(1).strip()
        else:
            tail_start = text.lower().rfind("</think>")
            if tail_start != -1:
                answer_body = text[tail_start + len("</think>") :].strip()
            else:
                answer_body = text.strip()
            answer_body = re.sub(
                r"</?(?:think|python|result|answer)>",
                "",
                answer_body,
                flags=re.IGNORECASE,
            ).strip()

        return reasoning, answer_body