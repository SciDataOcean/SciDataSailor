"""
ReAct-based trajectory sampler for scientific QA synthesis.

Implements the Thought-Action-Observation loop following the smolagents ReAct
framework, replacing the single-shot XML generation approach of SciTrajectorySampler.
"""

import asyncio
import bdb
import hashlib
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..prompt.react import REACT_SYSTEM_PROMPT, REACT_TASK_TEMPLATE
from ..prompt.tools import PYTHON_TOOL_SCHEMA
from .config import SynthesisConfig
from .models import Trajectory, TrajectoryNode

# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_THOUGHT_PATTERN = re.compile(
    r"Thought:\s*(.*?)(?=\nAction:|\Z)", re.DOTALL
)
_ACTION_BLOCK_PATTERN = re.compile(r"Action:\s*(.*)", re.DOTALL)
_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_first_json_object(text: str) -> str:
    """Extract the first complete JSON object from text."""
    if not text:
        return text
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _extract_action_json(action_block: str) -> Optional[Dict[str, Any]]:
    """Extract a complete JSON object from Action block text."""
    if not action_block:
        return None

    candidate = action_block.strip()
    fenced_match = _FENCED_JSON_PATTERN.search(candidate)
    if fenced_match:
        candidate = fenced_match.group(1).strip()

    json_text = _extract_first_json_object(candidate)
    try:
        action = json.loads(json_text)
        if isinstance(action, dict):
            return action
    except json.JSONDecodeError:
        pass

    # Fallback: tolerant cleanup for common JSON issues.
    cleaned = re.sub(r",\s*}", "}", json_text)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    try:
        action = json.loads(cleaned)
        if isinstance(action, dict):
            return action
    except json.JSONDecodeError:
        return None
    return None


def parse_react_output(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Parse a single ReAct step output into (thought, action_dict).

    Returns (thought_text, action_dict) where action_dict has keys
    ``tool_name`` and ``code``.  If no valid action is found, returns
    ``(thought_text, None)``.
    """
    thought = ""
    thought_match = _THOUGHT_PATTERN.search(text)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = _ACTION_BLOCK_PATTERN.search(text)
    if not action_match:
        return thought, None

    action_raw = action_match.group(1)
    action = _extract_action_json(action_raw)
    if action is None:
        return thought, None

    tool_name = action.get("tool_name", "")
    code = action.get("code", "")

    return thought, {"tool_name": tool_name, "code": code}


# ---------------------------------------------------------------------------
# ReAct Trajectory Sampler
# ---------------------------------------------------------------------------


class ReactTrajectorySampler:
    """Sample exploration trajectories using the ReAct paradigm.

    Each trajectory is an iterative Thought→Action→Observation chain,
    mirroring the smolagents ``MultiStepAgent._run_stream`` loop.
    """

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
        self.dataset_path = dataset_path

    # ----- public API -------------------------------------------------------

    async def sample_trajectories(
        self,
        seed_data: str,
        seed_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Trajectory]:
        """Sample ``n_trajectories`` independent ReAct chains for one seed.

        Returns a list of :class:`Trajectory` objects (one per chain).
        """
        if seed_kwargs is None:
            seed_kwargs = {}

        n_traj = getattr(self.config, "n_trajectories", 3)
        max_steps = getattr(self.config, "max_steps", 15)

        print(f"\n{'=' * 60}")
        print("Starting ReAct Trajectory Sampling")
        print(f"Seed: {seed_data[:100]}...")
        print(f"Dataset path: {self.dataset_path}")
        print(f"n_trajectories: {n_traj}, max_steps: {max_steps}")
        print(f"{'=' * 60}\n")

        tasks = [
            self._sample_single_trajectory(
                seed_data, max_steps, traj_idx, seed_kwargs
            )
            for traj_idx in range(n_traj)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        trajectories: List[Trajectory] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  Trajectory {idx} failed: {result}")
                continue
            if result is not None:
                trajectories.append(result)

        print(f"\nReAct sampling complete. Got {len(trajectories)} trajectories.")
        return trajectories

    # ----- single trajectory ------------------------------------------------

    async def _sample_single_trajectory(
        self,
        seed_data: str,
        max_steps: int,
        traj_idx: int,
        seed_kwargs: Dict[str, Any],
    ) -> Optional[Trajectory]:
        """Run one full ReAct loop, producing a single trajectory."""

        traj_id = f"react_{uuid.uuid4().hex[:8]}"
        print(f"\n--- Trajectory {traj_idx} (id={traj_id}) ---")

        system_prompt = self._build_system_prompt()
        task_prompt = self._build_task_prompt(seed_data, seed_kwargs)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        nodes: List[TrajectoryNode] = []
        used_code_hashes: set = set()

        for step in range(1, max_steps + 1):
            try:
                # ----- 1. Call LLM -----------------------------------------
                llm_output = await self._call_llm(messages, traj_idx, step)
                if not llm_output:
                    print(f"  [{traj_id}] Step {step}: empty LLM response, stopping.")
                    break

                # ----- 2. Parse Thought + Action ---------------------------
                thought, action = parse_react_output(llm_output)

                if not action:
                    print(f"  [{traj_id}] Step {step}: no action parsed, stopping.")
                    preview = llm_output[:800].replace("\n", "\\n")
                    print(f"  [{traj_id}] Raw LLM output preview: {preview}")
                    nodes.append(TrajectoryNode(
                        node_id=self._gen_node_id(),
                        thought=thought,
                        action=None,
                        observation="",
                        parent_id=nodes[-1].node_id if nodes else None,
                        children_ids=[],
                        depth=step,
                    ))
                    break

                tool_name = action.get("tool_name", "")
                code = action.get("code", "")

                # ----- 3. Check STOP signal --------------------------------
                if tool_name.upper() == "STOP":
                    print(f"  [{traj_id}] Step {step}: agent signaled STOP.")
                    nodes.append(TrajectoryNode(
                        node_id=self._gen_node_id(),
                        thought=thought,
                        action={"tool_name": "STOP", "parameters": {}},
                        observation="Agent stopped exploration.",
                        parent_id=nodes[-1].node_id if nodes else None,
                        children_ids=[],
                        depth=step,
                    ))
                    break

                # ----- 4. Dedup check --------------------------------------
                code_hash = hashlib.md5(code.encode()).hexdigest()
                if code_hash in used_code_hashes:
                    print(f"  [{traj_id}] Step {step}: duplicate code, stopping.")
                    break
                used_code_hashes.add(code_hash)

                # ----- 5. Execute action -----------------------------------
                observation = await self._execute_action(tool_name, code)

                # ----- 6. Create node & append to history ------------------
                node = TrajectoryNode(
                    node_id=self._gen_node_id(),
                    thought=thought,
                    action={
                        "tool_name": tool_name,
                        "parameters": {"code": code},
                    },
                    observation=observation,
                    parent_id=nodes[-1].node_id if nodes else None,
                    children_ids=[],
                    depth=step,
                )
                if nodes:
                    nodes[-1].children_ids.append(node.node_id)
                nodes.append(node)

                # ----- 7. Console log step ---------------------------------
                action_preview = code[:120] + "..." + code[-120:] if len(code) > 240 else code
                obs_preview = observation[:300] + "..." + observation[-300:] if len(observation) > 600 else observation
                print(
                    f"\033[36m[Traj {traj_idx} Step {step}]\033[0m\n"
                    f"  Thought: {thought[:150]}...\n"
                    f"  Action: {action_preview}\n"
                    f"  Observation: {obs_preview}\n"
                )

                # ----- 8. Append to conversation messages ------------------
                messages.append({
                    "role": "assistant",
                    "content": llm_output,
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

            except Exception as exc:
                if isinstance(exc, bdb.BdbQuit):
                    raise
                print(f"  [{traj_id}] Step {step} error: {exc}")
                continue

        if not nodes:
            print(f"  [{traj_id}] No nodes collected, skipping.")
            return None

        trajectory = Trajectory(
            trajectory_id=traj_id,
            nodes=nodes,
            seed_data=seed_data,
            total_depth=len(nodes),
        )
        print(f"  [{traj_id}] Completed with {len(nodes)} steps.")
        return trajectory

    # ----- LLM call ---------------------------------------------------------

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        traj_idx: int,
        step: int,
    ) -> str:
        """Call the LLM with the accumulated conversation messages."""
        # Vary temperature slightly across trajectories for diversity
        base_temp = 0.7
        temp = base_temp + 0.05 * traj_idx

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                result = await self.llm_client.chat_completion(
                    messages=messages,
                    temperature=min(temp, 1.0),
                )
                # LLMClient.chat_completion returns {"text": str, "usage": dict}
                if not result:
                    return ""
                if isinstance(result, dict):
                    text = result.get("text", "")
                else:
                    text = str(result)
                # Strip <think>...</think> reasoning wrapper if present
                text = re.sub(
                    r"<think>.*?</think>\s*", "", text, flags=re.DOTALL
                )
                return text
            except Exception as exc:
                if isinstance(exc, bdb.BdbQuit):
                    raise
                if attempt >= max_retries:
                    print(f"  LLM call failed after {max_retries + 1} attempts: {exc}")
                    return ""
                await asyncio.sleep(0.5 * (2 ** attempt))
        return ""

    # ----- Action execution -------------------------------------------------

    async def _execute_action(self, tool_name: str, code: str) -> str:
        """Execute a tool action and return the observation string."""
        if tool_name != "python_interpreter":
            return (
                f"Error: unknown tool '{tool_name}'. "
                "Only 'python_interpreter' is available."
            )
        if not code.strip():
            return "Error: empty code block."

        try:
            timeout = self.config.sandbox_timeout
            return await self.python_tool.execute(code, timeout=timeout)
        except Exception as exc:
            print(f"  Error executing python_interpreter: {exc}")
            return f"Error executing python_interpreter: {exc}"

    # ----- Prompt building --------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the ReAct system prompt."""
        return REACT_SYSTEM_PROMPT.replace(
            "{dataset_path}", self.dataset_path or "/data",
        )

    def _build_task_prompt(
        self, seed_data: str, seed_kwargs: Dict[str, Any]
    ) -> str:
        """Build the task prompt for the start of a ReAct chain."""

        seed_desc = ""
        if self.config.seed_description:
            seed_desc = f"Description: {self.config.seed_description}"

        tips = ""
        if self.config.sampling_tips:
            tips = f"[Exploration Strategy]\n{self.config.sampling_tips}"

        return (
            REACT_TASK_TEMPLATE
            .replace("{seed_data}", seed_data)
            .replace("{dataset_path}", self.dataset_path or "/data")
            .replace("{seed_description}", seed_desc)
            .replace("{sampling_tips}", tips)
        )

    # ----- Utilities --------------------------------------------------------

    @staticmethod
    def _gen_node_id() -> str:
        return f"node_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Inline self-test
# ---------------------------------------------------------------------------
def _self_test():
    """Quick sanity check for parse_react_output."""
    sample = """Thought: I need to list the directory to see what files exist.
Action:
{"tool_name": "python_interpreter", "code": "import os; print(os.listdir('/data'))"}
"""
    thought, action = parse_react_output(sample)
    assert "list the directory" in thought, f"Unexpected thought: {thought}"
    assert action is not None, "Action should not be None"
    assert action["tool_name"] == "python_interpreter"
    assert "os.listdir" in action["code"]

    # STOP signal
    stop_sample = """Thought: I have enough info.
Action:
{"tool_name": "STOP", "code": ""}
"""
    thought2, action2 = parse_react_output(stop_sample)
    assert action2["tool_name"] == "STOP"

    # No action
    no_action = "Thought: just thinking..."
    thought3, action3 = parse_react_output(no_action)
    assert action3 is None

    print("All self-tests passed!")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _self_test()
