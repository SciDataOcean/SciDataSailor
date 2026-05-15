#!/usr/bin/env python3
"""
Visualize MCTS tree shape from synthesized trajectories JSON.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class NodeInfo:
    node_id: str
    parent_id: Optional[str]
    depth: int
    label: str
    r_post: Optional[float]


def _iter_trajectories(path: Path) -> Iterable[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array in input file.")
        return data

    out: List[dict] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        out.append(json.loads(raw))
    return out


def _node_label(node: dict) -> str:
    nid = str(node.get("node_id", "unknown"))
    short = nid[-6:]
    depth = int(node.get("depth", 0))
    mcts_stats = node.get("mcts_stats") or {}
    r_post = mcts_stats.get("r_post", None)
    if isinstance(r_post, (int, float)):
        return f"{short}\\nd={depth}, post={float(r_post):.2f}"
    return f"{short}\\nd={depth}"


def build_graph(trajectories: Iterable[dict]) -> Tuple[Dict[str, NodeInfo], Set[Tuple[str, str]]]:
    nodes: Dict[str, NodeInfo] = {}
    edges: Set[Tuple[str, str]] = set()

    for traj in trajectories:
        t_nodes = traj.get("nodes") or []
        if not isinstance(t_nodes, list):
            continue

        for n in t_nodes:
            if not isinstance(n, dict):
                continue
            nid = str(n.get("node_id", "")).strip()
            if not nid:
                continue
            parent_id = n.get("parent_id")
            parent_id = str(parent_id) if parent_id is not None else None
            depth = int(n.get("depth", 0))
            mcts_stats = n.get("mcts_stats") or {}
            r_post = mcts_stats.get("r_post", None)
            if not isinstance(r_post, (int, float)):
                r_post = None

            if nid not in nodes:
                nodes[nid] = NodeInfo(
                    node_id=nid,
                    parent_id=parent_id,
                    depth=depth,
                    label=_node_label(n),
                    r_post=r_post,
                )
            else:
                # Keep a stable parent if already known.
                if nodes[nid].parent_id is None and parent_id is not None:
                    nodes[nid].parent_id = parent_id
                nodes[nid].depth = min(nodes[nid].depth, depth)

            if parent_id:
                edges.add((parent_id, nid))

            for cid in n.get("children_ids") or []:
                cid = str(cid)
                if cid:
                    edges.add((nid, cid))

    # Backfill unknown nodes found only from children_ids
    unknown_ids = {v for _, v in edges if v not in nodes}
    for nid in unknown_ids:
        nodes[nid] = NodeInfo(
            node_id=nid,
            parent_id=None,
            depth=999,
            label=f"{nid[-6:]}\\nd=?",
            r_post=None,
        )
    return nodes, edges


def _children_map(edges: Set[Tuple[str, str]]) -> Dict[str, List[str]]:
    cmap: Dict[str, List[str]] = defaultdict(list)
    for p, c in sorted(edges):
        cmap[p].append(c)
    return cmap


def _roots(nodes: Dict[str, NodeInfo], edges: Set[Tuple[str, str]]) -> List[str]:
    indeg: Dict[str, int] = {nid: 0 for nid in nodes}
    for p, c in edges:
        if p not in indeg:
            indeg[p] = 0
        indeg[c] = indeg.get(c, 0) + 1
    roots = [nid for nid, d in indeg.items() if d == 0]
    roots.sort()
    return roots


def write_ascii_tree(nodes: Dict[str, NodeInfo], edges: Set[Tuple[str, str]], out_path: Path) -> None:
    cmap = _children_map(edges)
    roots = _roots(nodes, edges)
    lines: List[str] = []

    def dfs(nid: str, prefix: str, is_last: bool) -> None:
        marker = "└── " if is_last else "├── "
        node = nodes[nid]
        rp = "" if node.r_post is None else f" post={node.r_post:.2f}"
        lines.append(f"{prefix}{marker}{nid} (d={node.depth}{rp})")
        next_prefix = prefix + ("    " if is_last else "│   ")
        children = cmap.get(nid, [])
        for i, cid in enumerate(children):
            dfs(cid, next_prefix, i == len(children) - 1)

    if not roots:
        lines.append("(empty tree)")
    else:
        for i, rid in enumerate(roots):
            lines.append(rid)
            for j, cid in enumerate(cmap.get(rid, [])):
                dfs(cid, "", j == len(cmap.get(rid, [])) - 1)
            if i != len(roots) - 1:
                lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dot(nodes: Dict[str, NodeInfo], edges: Set[Tuple[str, str]], out_path: Path) -> None:
    depths: Dict[int, List[str]] = defaultdict(list)
    for nid, info in nodes.items():
        depths[info.depth].append(nid)

    lines: List[str] = [
        "digraph MCTSTree {",
        "  rankdir=TB;",
        "  node [shape=box, fontsize=10, fontname=\"Courier\"];",
    ]

    for depth in sorted(depths):
        lines.append("  { rank = same;")
        for nid in sorted(depths[depth]):
            label = nodes[nid].label.replace('"', '\\"')
            lines.append(f"    \"{nid}\" [label=\"{label}\"];")
        lines.append("  }")

    for p, c in sorted(edges):
        lines.append(f"  \"{p}\" -> \"{c}\";")

    lines.append("}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(nodes: Dict[str, NodeInfo], edges: Set[Tuple[str, str]], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    cmap = _children_map(edges)
    roots = _roots(nodes, edges)
    level_nodes: Dict[int, List[str]] = defaultdict(list)

    # BFS order for a cleaner left-to-right layout within each depth.
    seen: Set[str] = set()
    q: deque[Tuple[str, int]] = deque((r, 0) for r in roots)
    while q:
        nid, d = q.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        level_nodes[d].append(nid)
        for cid in cmap.get(nid, []):
            q.append((cid, d + 1))

    # Include disconnected/leftover nodes.
    for nid in nodes:
        if nid not in seen:
            d = max(nodes[nid].depth, 0)
            level_nodes[d].append(nid)

    pos: Dict[str, Tuple[float, float]] = {}
    for d in sorted(level_nodes):
        row = level_nodes[d]
        n = max(len(row), 1)
        for i, nid in enumerate(row):
            x = (i + 1) / (n + 1)
            y = -float(d)
            pos[nid] = (x, y)

    fig_w = max(10.0, max((len(v) for v in level_nodes.values()), default=1) * 1.8)
    fig_h = max(6.0, len(level_nodes) * 1.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # edges
    for p, c in edges:
        if p in pos and c in pos:
            x1, y1 = pos[p]
            x2, y2 = pos[c]
            ax.plot([x1, x2], [y1, y2], color="#888888", linewidth=1.0, zorder=1)

    # nodes
    xs = [pos[nid][0] for nid in pos]
    ys = [pos[nid][1] for nid in pos]
    colors = [nodes[nid].depth for nid in pos]
    ax.scatter(xs, ys, c=colors, cmap="viridis", s=450, zorder=2, edgecolors="black")

    for nid in pos:
        x, y = pos[nid]
        label = nodes[nid].label.replace("\\n", "\n")
        ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=3)

    ax.set_title("MCTS Tree Shape")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw MCTS tree from trajectories JSON.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to trajectories file (JSON array or JSONL).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output path prefix (default: input basename without suffix).",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG generation even if matplotlib is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    prefix = (
        Path(args.output_prefix).resolve()
        if args.output_prefix
        else in_path.with_suffix("")
    )

    trajectories = _iter_trajectories(in_path)
    nodes, edges = build_graph(trajectories)
    if not nodes:
        raise RuntimeError("No nodes were parsed from input.")

    ascii_path = Path(str(prefix) + "_tree.txt")
    dot_path = Path(str(prefix) + "_tree.dot")
    png_path = Path(str(prefix) + "_tree.png")

    write_ascii_tree(nodes, edges, ascii_path)
    write_dot(nodes, edges, dot_path)

    png_done = False
    if not args.no_png:
        png_done = write_png(nodes, edges, png_path)

    print(f"Parsed nodes: {len(nodes)}, edges: {len(edges)}")
    print(f"ASCII tree: {ascii_path}")
    print(f"DOT file:   {dot_path}")
    if png_done:
        print(f"PNG file:   {png_path}")
    else:
        print("PNG file:   skipped (matplotlib unavailable or --no-png)")


if __name__ == "__main__":
    main()
