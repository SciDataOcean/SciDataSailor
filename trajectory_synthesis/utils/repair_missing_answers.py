from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# File moved to trajectory_synthesis/utils/, so repo root is 3 levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "trajectory_synthesis" / "results" / "OpenScienceLab"

# Make imports robust when launched outside repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_client import LLMClient
from trajectory_synthesis.prompt.tooltree_mcts import (
    FINAL_ANSWER_SYSTEM_TEMPLATE,
    FINAL_ANSWER_USER_TEMPLATE,
)


def _resolve_env_vars(value: str) -> str:
    text = str(value or "")
    return os.path.expandvars(os.path.expanduser(text))


def _parse_final_think_answer(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    think_m = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
    reasoning = think_m.group(1).strip() if think_m else ""

    answer_m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if answer_m:
        answer_body = answer_m.group(1).strip()
    else:
        tail_start = text.lower().rfind("</think>")
        if tail_start != -1:
            answer_body = text[tail_start + len("</think>") :].strip()
        else:
            answer_body = text.strip()
        answer_body = re.sub(r"</?(?:think|python|result|answer)>", "", answer_body, flags=re.IGNORECASE).strip()
    return reasoning, answer_body


def _fallback_final_think(answer_body: str) -> str:
    cleaned = re.sub(r"\s+", " ", (answer_body or "").strip())
    if not cleaned:
        return (
            "I have gathered enough metadata from the previous exploration steps; "
            "let me compile everything into the final answer."
        )
    preview = cleaned[:200].rstrip()
    suffix = "..." if len(cleaned) > len(preview) else ""
    return (
        "Now I have all the information needed to answer the question. "
        f"Let me compile the metadata summary: {preview}{suffix}"
    )


def _extract_answer_body(logs: Sequence[str]) -> str:
    joined = "\n".join(str(x) for x in logs)
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", joined, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _looks_placeholder_answer(answer_body: str) -> bool:
    body = (answer_body or "").strip().lower()
    if not body:
        return True
    if "boxed{answer here}" in body:
        return True
    if re.search(r"boxed\s*\{\s*\}", body):
        return True
    if body in {"answer here", "the final answer is"}:
        return True
    return False


def _build_nodes_index(trajectory: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for node in trajectory.get("nodes", []) or []:
        node_id = str(node.get("node_id", ""))
        if node_id:
            out[node_id] = node
    return out


def _path_to_root(node_id: str, nodes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    chain: List[Dict[str, Any]] = []
    cur_id: Optional[str] = node_id
    while cur_id:
        node = nodes.get(cur_id)
        if not node:
            break
        chain.append(node)
        parent = node.get("parent_id")
        cur_id = str(parent) if parent else None
    chain.reverse()
    return chain


def _render_trajectory_text_for_answer(leaf_id: str, nodes: Dict[str, Dict[str, Any]]) -> str:
    chain = _path_to_root(leaf_id, nodes)
    fragments: List[str] = []
    for n in chain:
        if n.get("parent_id") is None:
            continue
        for log in n.get("logs", []) or []:
            fragments.append(str(log))
    return "\n".join(fragments) if fragments else "(no prior steps)"


def _find_bad_answer_leaf(trajectory: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], str]]:
    nodes = trajectory.get("nodes", []) or []
    if not nodes:
        return None
    leaves = [n for n in nodes if not (n.get("children_ids") or [])]
    if not leaves:
        return None
    leaves_sorted = sorted(leaves, key=lambda n: int(n.get("depth", 0)), reverse=True)
    for leaf in leaves_sorted:
        logs = leaf.get("logs", []) or []
        answer_body = _extract_answer_body(logs)
        if not answer_body:
            continue
        if _looks_placeholder_answer(answer_body):
            parent_id = leaf.get("parent_id")
            if parent_id:
                return leaf, str(parent_id)
    return None


async def _regen_answer_for_trajectory(
    llm: LLMClient,
    trajectory: Dict[str, Any],
    dataset_path: str,
    *,
    judge_temperature: float,
    max_tokens: int,
    max_python_times: int,
) -> bool:
    found = _find_bad_answer_leaf(trajectory)
    if not found:
        return False
    bad_leaf, evidence_leaf_id = found
    nodes_idx = _build_nodes_index(trajectory)
    trajectory_text = _render_trajectory_text_for_answer(evidence_leaf_id, nodes_idx)
    seed_data = str(trajectory.get("seed_data", ""))
    system_prompt = FINAL_ANSWER_SYSTEM_TEMPLATE.format(
        max_python_times=max_python_times,
        input_path=dataset_path or "/data",
    )
    user_prompt = FINAL_ANSWER_USER_TEMPLATE.format(
        seed_data=seed_data,
        trajectory_text=trajectory_text,
    )
    result = await llm.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=float(judge_temperature),
        max_tokens=int(max_tokens),
    )
    text = str((result or {}).get("text", "") or "")
    think, answer = _parse_final_think_answer(text)
    if not answer.strip():
        return False
    if not think.strip():
        think = _fallback_final_think(answer)

    bad_leaf["logs"] = [
        f"<think>\n{think.strip()}\n</think>",
        f"<answer>\n{answer.strip()}\n</answer>",
    ]
    return True


def _discover_trajectory_files(results_root: Path) -> List[Path]:
    # 1) Direct file path.
    if results_root.is_file() and results_root.name == "trajectories.jsonl":
        return [results_root]
    # 2) Single dataset directory containing trajectories.jsonl.
    direct = results_root / "trajectories.jsonl"
    if direct.is_file():
        return [direct]
    # 3) Multi-dataset root where each child has trajectories.jsonl.
    return sorted(p for p in results_root.glob("*/trajectories.jsonl") if p.is_file())


def _load_trajectories(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected JSON array")
        return data, "json_array"

    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows, "jsonl"


def _save_trajectories(path: Path, rows: List[Dict[str, Any]], fmt: str) -> None:
    if fmt == "json_array":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_dataset_path(dataset_root: Path, trajectory_file: Path) -> str:
    # If caller already provides a concrete dataset raw path, use it directly.
    # Example:
    #   --dataset-root /root/code/SciDataCrawler/data/OpenScienceLab/3DTurbulence-DNS/OpenScienceLab___3DTurbulence-DNS/raw
    if dataset_root.name == "raw":
        return str(dataset_root)

    # Otherwise, treat dataset_root as the OpenScienceLab base and infer by dataset folder name.
    return str(dataset_root / trajectory_file.parent.name / "raw")


async def _process_file(
    llm: LLMClient,
    traj_path: Path,
    dataset_path: str,
    dry_run: bool,
    max_fixes: Optional[int],
    *,
    judge_temperature: float,
    max_tokens: int,
    max_python_times: int,
) -> Tuple[int, int]:
    rows, fmt = _load_trajectories(traj_path)
    scanned = 0
    fixed = 0
    for row in rows:
        scanned += 1
        if max_fixes is not None and fixed >= max_fixes:
            break
        changed = await _regen_answer_for_trajectory(
            llm,
            row,
            dataset_path=dataset_path,
            judge_temperature=judge_temperature,
            max_tokens=max_tokens,
            max_python_times=max_python_times,
        )
        if changed:
            fixed += 1
    if fixed > 0 and not dry_run:
        _save_trajectories(traj_path, rows, fmt)
    return scanned, fixed


async def _amain() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible endpoint URL.")
    parser.add_argument("--api-key", required=True, help="API key for the endpoint.")
    parser.add_argument("--model", required=True, help="Model name.")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing dataset result folders.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature for final-answer regeneration.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens for each final-answer call.",
    )
    parser.add_argument(
        "--max-python-times",
        type=int,
        default=8,
        help="Value injected into FINAL_ANSWER_SYSTEM_TEMPLATE.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and call LLM but do not write files.",
    )
    parser.add_argument(
        "--max-fixes",
        type=int,
        default=None,
        help="Stop after fixing this many trajectories per file.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "data" / "OpenScienceLab"),
        help="Dataset root used to infer per-dataset input_path in prompts.",
    )
    args = parser.parse_args()

    endpoint = _resolve_env_vars(args.endpoint)
    api_key = _resolve_env_vars(args.api_key)
    model = args.model.strip()
    if not endpoint or not api_key or not model:
        raise ValueError("endpoint/api-key/model must be non-empty")
    llm = LLMClient(endpoints=[endpoint], api_keys=[api_key], default_model=model)

    root = Path(args.results_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    files = _discover_trajectory_files(root)
    if not files:
        print(f"[warn] no trajectories.jsonl found under: {root}")
        return

    total_scanned = 0
    total_fixed = 0
    for fp in files:
        dataset_path = _resolve_dataset_path(dataset_root, fp)
        scanned, fixed = await _process_file(
            llm=llm,
            traj_path=fp,
            dataset_path=dataset_path,
            dry_run=bool(args.dry_run),
            max_fixes=args.max_fixes,
            judge_temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            max_python_times=max(1, int(args.max_python_times)),
        )
        total_scanned += scanned
        total_fixed += fixed
        print(f"[file] {fp} scanned={scanned} fixed={fixed}")

    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"[done] mode={mode} scanned={total_scanned} fixed={total_fixed}")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()

