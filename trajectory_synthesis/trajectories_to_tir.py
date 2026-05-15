from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as a plain script from repo root.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from trajectory_synthesis.prompt.tooltree_mcts import AGENT_ROLE_PREAMBLE  # noqa: E402

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

ROOT_PATH_RE = re.compile(r"""root_path\s*=\s*['\"]([^'\"]+)['\"]""")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
TAG_RE_CACHE: Dict[str, re.Pattern] = {}

# Long decorative separator lines (e.g. "=" * 80) printed inside <result> blocks
_SEPARATOR_RUN_RE = re.compile(r"([=\-_])\1{5,}")


def _shorten_decorative_separators(text: str) -> str:
    if not text:
        return text
    return _SEPARATOR_RUN_RE.sub(lambda m: m.group(1) * 3, text)


def _tag_regex(tag: str) -> re.Pattern:
    if tag not in TAG_RE_CACHE:
        TAG_RE_CACHE[tag] = re.compile(fr"<{tag}>(.*?)</{tag}>", re.DOTALL)
    return TAG_RE_CACHE[tag]


def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = _tag_regex(tag).search(text or "")
    return match.group(1) if match else None


def _extract_outer_tag_block(text: str, tag: str) -> Optional[str]:
    raw = text or ""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = raw.find(open_tag)
    if start == -1:
        return None
    end = raw.rfind(close_tag)
    if end == -1 or end < start:
        return None
    end += len(close_tag)
    return raw[start:end].strip()


def _extract_outer_tag_content(text: str, tag: str) -> Optional[str]:
    block = _extract_outer_tag_block(text, tag)
    if not block:
        return None
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    inner = block[len(open_tag) : -len(close_tag)]
    return inner.strip()


def _has_tag(text: str, tag: str) -> bool:
    return f"<{tag}>" in (text or "")


# ---------------------------------------------------------------------------
# Loading & tree traversal
# ---------------------------------------------------------------------------


def load_trajectories(path: Path) -> List[Dict[str, Any]]:
    """Load trees from ``trajectories.jsonl`` (JSON array preferred, JSONL fallback)."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    out: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def infer_dataset_path(trees: List[Dict[str, Any]]) -> Optional[str]:
    """Best-effort recovery of the dataset root path from python snippets."""
    for tree in trees:
        for node in tree.get("nodes", []) or []:
            for log in node.get("logs", []) or []:
                match = ROOT_PATH_RE.search(log)
                if match:
                    return match.group(1)
    return None


def _index_by_id(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {node["node_id"]: node for node in nodes if "node_id" in node}


def _find_root_id(nodes: List[Dict[str, Any]]) -> Optional[str]:
    for node in nodes:
        if node.get("parent_id") is None:
            return node.get("node_id")
    return nodes[0].get("node_id") if nodes else None


def _root_to_leaf_paths(
    nodes_by_id: Dict[str, Dict[str, Any]],
    root_id: str,
) -> List[List[Dict[str, Any]]]:
    paths: List[List[Dict[str, Any]]] = []
    stack: List[Dict[str, Any]] = []

    def _dfs(node_id: str) -> None:
        node = nodes_by_id.get(node_id)
        if node is None:
            return
        stack.append(node)
        children = node.get("children_ids") or []
        if not children:
            paths.append(list(stack))
        else:
            for child_id in children:
                _dfs(child_id)
        stack.pop()

    _dfs(root_id)
    return paths


# ---------------------------------------------------------------------------
# Per-path rendering
# ---------------------------------------------------------------------------


def _node_is_root_placeholder(node: Dict[str, Any]) -> bool:
    logs = node.get("logs") or []
    if len(logs) != 1:
        return False
    inner = _extract_tag(logs[0], "think") or ""
    return inner.strip().lower().startswith("root: start exploration")


def _pick_first(logs: List[str], tag: str) -> Optional[str]:
    for log in logs:
        block = _extract_outer_tag_block(log, tag)
        if block is not None:
            return block
    return None


def build_logs_for_path(path_nodes: List[Dict[str, Any]]) -> List[str]:
    """Collapse per-node logs into the alternating shape used by results_tir.json.

    Normal step node → ``"<think>…</think> <python>…</python>"`` + ``<result>…</result>``.
    Answer leaf node → ``"<think>…</think> <answer>…</answer>"`` (single entry).
    Root placeholder → dropped.
    """
    out: List[str] = []
    for node in path_nodes:
        if _node_is_root_placeholder(node):
            continue
        logs = node.get("logs") or []
        think_log = _pick_first(logs, "think")
        python_log = _pick_first(logs, "python")
        result_log = _pick_first(logs, "result")
        answer_log = _pick_first(logs, "answer")

        if answer_log:
            if think_log:
                out.append(f"{think_log} {answer_log}")
            else:
                out.append(answer_log)
            continue

        if think_log and python_log:
            out.append(f"{think_log} {python_log}")
        elif python_log:
            out.append(python_log)
        elif think_log:
            out.append(think_log)

        if result_log:
            out.append(_shorten_decorative_separators(result_log))

    return out


def extract_prediction(path_nodes: List[Dict[str, Any]]) -> str:
    for node in reversed(path_nodes):
        for log in node.get("logs") or []:
            answer_body = _extract_outer_tag_content(log, "answer")
            if answer_body:
                return answer_body
    return ""


# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------


def convert_tree(
    tree: Dict[str, Any],
    task_id: str,
    dataset_path: str,
) -> List[Dict[str, Any]]:
    nodes = tree.get("nodes") or []
    if not nodes:
        return []
    nodes_by_id = _index_by_id(nodes)
    root_id = _find_root_id(nodes)
    if not root_id:
        return []

    paths = _root_to_leaf_paths(nodes_by_id, root_id)
    seed_data = tree.get("seed_data", "") or ""

    entries: List[Dict[str, Any]] = []
    for path_nodes in paths:
        logs = build_logs_for_path(path_nodes)
        prediction = extract_prediction(path_nodes)
        entries.append(
            {
                "task_id": task_id,
                "instruction": AGENT_ROLE_PREAMBLE,
                "input": seed_data,
                "input_path": dataset_path,
                "prediction": prediction,
                "output": "".join(logs),
                "logs": logs,
            }
        )
    return entries


def convert_file(
    input_path: Path,
    output_path: Path,
    task_id: Optional[str],
    dataset_path: Optional[str],
) -> List[Dict[str, Any]]:
    trees = load_trajectories(input_path)
    if not trees:
        print(f"[warn] no trajectories loaded from {input_path}")

    resolved_task_id = task_id or input_path.parent.name
    resolved_dataset_path = dataset_path or infer_dataset_path(trees) or ""
    if not resolved_dataset_path:
        print(
            "[warn] dataset path could not be inferred; "
            "output 'input_path' will be empty. Pass --dataset-path to override."
        )

    entries: List[Dict[str, Any]] = []
    for tree in trees:
        entries.extend(convert_tree(tree, resolved_task_id, resolved_dataset_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(entries)} TIR record(s) for task_id='{resolved_task_id}' "
        f"from {len(trees)} tree(s) → {output_path}"
    )
    return entries


# ---------------------------------------------------------------------------
# Optional quality-check delegation
# ---------------------------------------------------------------------------


def run_quality_check(
    tir_file: Path,
    output_dir: Path,
    extra_args: Optional[List[str]] = None,
) -> int:
    script = _PKG_ROOT / "tasks" / "quality_check_v2.py"
    if not script.is_file():
        raise FileNotFoundError(f"quality_check_v2.py not found at {script}")

    cmd: List[str] = [
        sys.executable,
        str(script),
        "--tir_file",
        str(tir_file),
        "--output_dir",
        str(output_dir),
    ]

    extras = list(extra_args or [])
    def _has(flag: str) -> bool:
        return any(a == flag or a.startswith(flag + "=") for a in extras)
    if not _has("--sft_threshold"):
        cmd.extend(["--sft_threshold", "30"])
    if not _has("--rl_threshold"):
        cmd.extend(["--rl_threshold", "12"])
    if not _has("--sft_min_grounding"):
        cmd.extend(["--sft_min_grounding", "0"])
    if not _has("--sft_min_coverage"):
        cmd.extend(["--sft_min_coverage", "0"])
    if not _has("--sft_min_answer_quality"):
        cmd.extend(["--sft_min_answer_quality", "0"])
    if extras:
        cmd.extend(extras)

    print("\n$ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(_PKG_ROOT), check=False)
    return completed.returncode

DEFAULT_SFT_SYSTEM_PROMPT = AGENT_ROLE_PREAMBLE


def _qc_record_to_sharegpt(
    record: Dict[str, Any],
    system_prompt: str,
    fallback_sample_id: str,
) -> Dict[str, Any]:
    sample_id = record.get("sample_id") or fallback_sample_id
    human_value = record.get("input", "") or ""
    gpt_value = record.get("output") or ""
    if not gpt_value:
        logs = record.get("logs")
        if isinstance(logs, list):
            gpt_value = "".join(logs)
    if not gpt_value:
        gpt_value = record.get("prediction", "") or ""

    return {
        "sample_id": sample_id,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value},
        ],
        "system": system_prompt,
    }


def export_training_data(
    qc_output_dir: Path,
    training_dir: Path,
    system_prompt: str,
    fallback_sample_id: str,
) -> None:
    """Convert quality_check_v2 outputs into ShareGPT ``*_train.jsonl`` files."""
    mapping = {
        "sft_data.json": "sft_train.jsonl",
        "rl_data.json": "rl_train.jsonl",
    }
    training_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in mapping.items():
        src = qc_output_dir / src_name
        if not src.is_file():
            print(f"[warn] {src} not found; skipping {dst_name}")
            continue
        try:
            records = json.loads(src.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[warn] could not parse {src}: {exc}; skipping")
            continue
        if not isinstance(records, list):
            print(f"[warn] {src} is not a list; skipping")
            continue

        dst = training_dir / dst_name
        with dst.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(
                    json.dumps(
                        _qc_record_to_sharegpt(rec, system_prompt, fallback_sample_id),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Wrote {len(records)} ShareGPT record(s) → {dst}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, required=True, help="trajectories.jsonl path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output results_tir.json path (default: <input_dir>/results_tir.json)",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="task_id prefix (default: parent folder name of --input)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Dataset root path (auto-inferred from python snippets when omitted)",
    )
    parser.add_argument(
        "--run-quality-check",
        action="store_true",
        help="Also invoke tasks/quality_check_v2.py on the produced file.",
    )
    parser.add_argument(
        "--qc-output-dir",
        type=str,
        default=None,
        help="--output_dir forwarded to quality_check_v2.py (default: <output_dir>/quality_output_v2)",
    )
    parser.add_argument(
        "--qc-extra",
        nargs=argparse.REMAINDER,
        default=None,
        help="Extra args forwarded to quality_check_v2.py (e.g. --qc-extra -- --sft_threshold 75)",
    )
    parser.add_argument(
        "--export-training-data",
        action="store_true",
        help=(
            "After quality control, convert sft_data.json / rl_data.json into "
            "ShareGPT-style training_data/{sft,rl}_train.jsonl."
        ),
    )
    parser.add_argument(
        "--training-data-dir",
        type=str,
        default=None,
        help="Output directory for ShareGPT jsonl (default: <output_dir>/training_data)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override the 'system' field written into ShareGPT records (default: AGENT_ROLE_PREAMBLE).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.parent / "results_tir.json"
    )

    convert_file(
        input_path=input_path,
        output_path=output_path,
        task_id=args.task_id,
        dataset_path=args.dataset_path,
    )

    if args.run_quality_check:
        qc_output_dir = (
            Path(args.qc_output_dir).expanduser().resolve()
            if args.qc_output_dir
            else output_path.parent / "quality_output_v2"
        )
        extra = list(args.qc_extra or [])
        if extra and extra[0] == "--":
            extra = extra[1:]
        exit_code = run_quality_check(output_path, qc_output_dir, extra)
        if exit_code != 0:
            sys.exit(exit_code)

        if args.export_training_data:
            training_dir = (
                Path(args.training_data_dir).expanduser().resolve()
                if args.training_data_dir
                else output_path.parent / "training_data"
            )
            system_prompt = args.system_prompt or DEFAULT_SFT_SYSTEM_PROMPT
            fallback_sample_id = args.task_id or input_path.parent.name
            export_training_data(
                qc_output_dir=qc_output_dir,
                training_dir=training_dir,
                system_prompt=system_prompt,
                fallback_sample_id=fallback_sample_id,
            )
    elif args.export_training_data:
        print(
            "[warn] --export-training-data requires --run-quality-check to produce "
            "sft_data.json / rl_data.json first; skipping export."
        )


if __name__ == "__main__":
    main()
