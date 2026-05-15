"""
Trajectory selector for scientific QA synthesis.
"""

from typing import Dict, List, Optional

from .config import SynthesisConfig
from .models import Trajectory, TrajectoryNode
from .utils import create_openai_client


class TrajectorySelector:
    """Select high-quality trajectories from the exploration tree."""

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.available_tool_total: int = 0
        self.client = create_openai_client(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def select_trajectories(
        self,
        nodes: Dict[str, TrajectoryNode],
        root_id: str,
        seed_data: str,
        source_id: str,
        max_selected_traj: Optional[int] = None,
    ) -> List[Trajectory]:
        if max_selected_traj is None:
            max_selected_traj = self.config.max_selected_traj

        print(f"\n{'=' * 60}")
        print(f"Selecting Trajectories (max: {max_selected_traj})")
        print(f"{'=' * 60}\n")

        leaf_nodes = [node for node in nodes.values() if not node.children_ids]
        print(f"Found {len(leaf_nodes)} leaf nodes")

        valid_leaves = [
            node for node in leaf_nodes if node.depth >= self.config.min_depth
        ]
        print(f"Valid leaves (depth >= {self.config.min_depth}): {len(valid_leaves)}")

        if not valid_leaves:
            print("No trajectories meet depth requirement")
            return []

        candidate_paths = []
        for leaf in valid_leaves:
            path = self._build_path_to_root(leaf, nodes, root_id)
            if path:
                candidate_paths.append(path)

        print(f"Built {len(candidate_paths)} candidate paths")
        selected = self._score_and_select(
            candidate_paths, seed_data, source_id, max_selected_traj
        )
        print(f"\nSelected {len(selected)} trajectories")
        return selected

    def _build_path_to_root(
        self,
        leaf: TrajectoryNode,
        nodes: Dict[str, TrajectoryNode],
        root_id: str,
    ) -> List[TrajectoryNode]:
        root = nodes.get(root_id)
        path: List[TrajectoryNode] = []
        current = leaf

        while current.node_id != root_id:
            path.append(current)
            if current.parent_id is None:
                break
            current = nodes[current.parent_id]

        path.reverse()

        if not root:
            return path
        if leaf.node_id == root_id:
            return [root]
        if path and path[0].parent_id == root_id:
            return [root] + path
        return path

    def _score_and_select(
        self,
        paths: List[List[TrajectoryNode]],
        seed_data: str,
        source_id: str,
        max_selected: int,
    ) -> List[Trajectory]:
        all_avg_lengths = []
        for path in paths:
            avg_length = (
                sum(len(node.observation) for node in path) / len(path) if path else 0
            )
            all_avg_lengths.append(avg_length)

        min_length = min(all_avg_lengths) if all_avg_lengths else 0
        max_length = max(all_avg_lengths) if all_avg_lengths else 1
        length_range = max_length - min_length if max_length > min_length else 1

        scored_paths = []
        for idx, path in enumerate(paths):
            score = self._score_path(path, all_avg_lengths[idx], min_length, length_range)
            scored_paths.append((score, idx, path))

        scored_paths.sort(reverse=True, key=lambda item: item[0])

        selected_trajectories = []
        selected_path_sets = []
        similarity_threshold = getattr(
            self.config, "path_similarity_threshold", 0.7
        )

        for score, idx, path in scored_paths:
            if len(selected_trajectories) >= max_selected:
                break

            current_path_set = {node.node_id for node in path}
            is_too_similar = False

            for selected_set in selected_path_sets:
                intersection = len(current_path_set & selected_set)
                union = len(current_path_set | selected_set)
                jaccard_similarity = intersection / union if union > 0 else 0.0
                if jaccard_similarity > similarity_threshold:
                    is_too_similar = True
                    print(
                        f"  Trajectory {idx} too similar "
                        f"({jaccard_similarity:.2f} > {similarity_threshold:.2f}), skipping"
                    )
                    break

            if not is_too_similar:
                trajectory = Trajectory(
                    trajectory_id=f"{source_id}_traj_{idx}",
                    nodes=path,
                    seed_data=seed_data,
                    source_id=source_id,
                    total_depth=len(path),
                )
                selected_trajectories.append(trajectory)
                selected_path_sets.append(current_path_set)
                print(
                    f"  Selected Trajectory {len(selected_trajectories)}: "
                    f"ID={trajectory.trajectory_id}, depth={len(path)}, score={score:.2f}"
                )

        return selected_trajectories

    def _score_path(
        self,
        path: List[TrajectoryNode],
        avg_obs_length: Optional[float] = None,
        min_length: float = 0,
        length_range: float = 1,
    ) -> float:
        depth_score = min(len(path) / 20.0, 1.0) * 40

        if avg_obs_length is None:
            avg_obs_length = (
                sum(len(node.observation) for node in path) / len(path) if path else 0
            )
        normalized_length = (
            (avg_obs_length - min_length) / length_range if length_range > 0 else 0
        )
        info_score = normalized_length * 30

        tool_names = set()
        for node in path:
            if node.action:
                tool_names.add(node.action.get("tool_name", ""))
        total_tools = (
            len(self.config.available_tools)
            if self.config.available_tools
            else self.available_tool_total
        )
        diversity_score = len(tool_names) / max(total_tools, 1) * 30

        # ToolTree MCTS: prefer leaves with higher backpropagated Q (react paths keep 0).
        leaf_q = path[-1].mcts_q_value if path else 0.0
        mcts_bonus = max(leaf_q, 0.0) * 25.0

        return depth_score + info_score + diversity_score + mcts_bonus
