"""
Data models for the scientific QA synthesis pipeline.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryNode:
    """Single node in a trajectory tree."""

    node_id: str
    thought: str = ""
    action: Optional[Dict[str, Any]] = None
    observation: str = ""
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0

    # Hierarchical MCTS: high-level strategy label (e.g., "list_dir",
    # "inspect_schema", "cross_file_join") chosen BEFORE generating the
    # concrete Python code. ``action`` still stores the low-level realization
    # (tool_name + parameters); ``action_type`` tags which strategy it belongs
    # to so siblings can be diversified at the strategy level.
    action_type: Optional[str] = None

    # ToolTree-style MCTS metadata (optional; used by tooltree_mcts sampler)
    mcts_visit_count: int = 0
    mcts_q_value: float = 0.0
    mcts_r_pre: float = 0.0
    mcts_r_post: Optional[float] = None
    mcts_expandable: bool = True

    # Optional post-exploration answer text. When set on a non-STOP terminal
    # leaf, ``build_tagged_logs`` appends an additional ``<answer>`` block so
    # the downstream training format always ends with a boxed final answer
    # (see ``tasks/convert_to_training_format.py`` / ``src/prompt_manager.py``).
    final_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def build_tagged_logs(self) -> List[str]:
        """Per-step tags: thought → ``<think>``, action → ``<python>``,
        observation → ``<result>``; terminal answer step → ``<answer>``.

        Rendering rules:

        - ``action is None`` + ``final_answer`` set: pure *answer* leaf. Emits
          ``<think>...</think><answer>...</answer>`` (no python/result).
        - ``action`` with ``tool_name == "STOP"``: emits
          ``<think>...</think><python>{STOP JSON}</python><answer>...</answer>``.
        - Regular exploration step: emits ``<think>...</think><python>...</python><result>...</result>``.
        """
        thought = self.thought or ""
        logs: List[str] = []
        if thought.strip():
            logs.append(f"<think>\n{thought}\n</think>")

        action = self.action
        final_answer = (self.final_answer or "").strip()

        if not action:
            # Pure terminal answer leaf: <think> then <answer>.
            if final_answer:
                logs.append(f"<answer>\n{final_answer}\n</answer>")
            return logs

        logs.append(
            f"<python>\n{json.dumps(action, ensure_ascii=False)}\n</python>"
        )

        tool_name = str(action.get("tool_name", "")).upper()
        if tool_name == "STOP":
            answer_body = thought.strip() or (self.observation or "")
            logs.append(f"<answer>\n{answer_body}\n</answer>")
        else:
            logs.append(f"<result>\n{self.observation or ''}\n</result>")
        return logs

    def to_export_dict(self) -> Dict[str, Any]:
        """Serialization shape for ``trajectories.jsonl`` / pipeline output."""
        out: Dict[str, Any] = {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "depth": self.depth,
            "logs": self.build_tagged_logs(),
        }
        if self.action_type:
            out["action_type"] = self.action_type
        if (
            self.mcts_visit_count
            or self.mcts_q_value
            or self.mcts_r_pre
            or self.mcts_r_post is not None
            or not self.mcts_expandable
        ):
            out["mcts_stats"] = {
                "visit_count": self.mcts_visit_count,
                "q_value": self.mcts_q_value,
                "r_pre": self.mcts_r_pre,
                "r_post": self.mcts_r_post,
                "expandable": self.mcts_expandable,
            }
        return out


@dataclass
class Trajectory:
    """Complete trajectory chain."""

    trajectory_id: str
    nodes: List[TrajectoryNode]
    seed_data: str
    total_depth: int
    source_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        # A Trajectory is a linear root→leaf *chain* extracted from a possibly
        # branching exploration tree. Each TrajectoryNode may be shared across
        # trajectories, and its .children_ids still references ALL siblings
        # from the full tree (including nodes not present in this chain).
        # Rewrite children_ids / parent_id here so the exported trajectory is
        # internally consistent: each node points only to the next node in
        # the path, and parent_id only references the previous node in path
        # (or None for the root of the chain).
        path_ids = [node.node_id for node in self.nodes]
        path_set = set(path_ids)
        next_in_path: Dict[str, List[str]] = {
            nid: ([path_ids[i + 1]] if i + 1 < len(path_ids) else [])
            for i, nid in enumerate(path_ids)
        }

        serialized_nodes: List[Dict[str, Any]] = []
        for i, node in enumerate(self.nodes):
            node_dict = node.to_export_dict()
            node_dict["children_ids"] = list(next_in_path.get(node.node_id, []))
            # Keep parent_id consistent with the chain: only reference the
            # previous node in path, else None. Any ancestor outside this
            # chain would be an "unknown id" to downstream validators.
            if i == 0:
                node_dict["parent_id"] = None
            else:
                prev_id = path_ids[i - 1]
                # If the original parent_id matches the previous node in path
                # keep it; otherwise rewrite for internal consistency.
                if node_dict.get("parent_id") not in path_set:
                    node_dict["parent_id"] = prev_id
            serialized_nodes.append(node_dict)

        return {
            "trajectory_id": self.trajectory_id,
            "source_id": self.source_id,
            "seed_data": self.seed_data,
            "total_depth": len(self.nodes),
            "nodes": serialized_nodes,
        }


@dataclass
class SynthesizedQA:
    """Synthesized question-answer pair."""

    question: str
    answer: str
    trajectory_id: str
    reasoning_steps: List[str]
    source_id: str = ""
    qa_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesizedQA":
        data = dict(data or {})
        data.pop("negative_aspect", None)
        return cls(**data)
