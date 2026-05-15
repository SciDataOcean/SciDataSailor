from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# Allow running as a plain script from repo root or arbitrary cwd.
_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from trajectory_synthesis.core.config import SynthesisConfig
from trajectory_synthesis.core.models import Trajectory


@dataclass
class _LogOnlyNode:
    """Adapter to reuse stored node logs with QASynthesizer."""

    logs: List[str]

    def build_tagged_logs(self) -> List[str]:
        return list(self.logs or [])


def _load_trajectory_file(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        return data if isinstance(data, list) else []

    items: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _trajectory_from_record(record: Dict[str, Any], idx: int) -> Trajectory:
    node_objs = [
        _LogOnlyNode(logs=list((node or {}).get("logs") or []))
        for node in (record.get("nodes") or [])
    ]
    trajectory_id = str(record.get("trajectory_id") or f"traj_{idx}")
    source_id = str(record.get("source_id") or "")
    seed_data = str(record.get("seed_data") or "")
    total_depth = int(record.get("total_depth") or len(node_objs))

    return Trajectory(
        trajectory_id=trajectory_id,
        nodes=node_objs,  # type: ignore[arg-type]
        seed_data=seed_data,
        total_depth=total_depth,
        source_id=source_id,
    )


def _qa_to_conv(qa: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "conversations": [
            {"from": "human", "value": str(qa.get("question", "")).strip()},
            {"from": "gpt", "value": str(qa.get("answer", "")).strip()},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize QA from trajectories and export conversation JSONL"
    )
    parser.add_argument("--config", required=True, help="Path to synthesis config (.json/.yaml)")
    parser.add_argument("--trajectories", required=True, help="Path to trajectories file")
    parser.add_argument(
        "--qa-output",
        default=None,
        help="Raw QA output JSONL path (default: <trajectory_dir>/synthesized_qa.jsonl)",
    )
    parser.add_argument(
        "--train-output",
        default=None,
        help="Conversation output JSONL path (default: <trajectory_dir>/qa_train.jsonl)",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="sample_id for conversation records (default: trajectory parent dir name)",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="Optional cap on number of trajectories to synthesize (0 means all)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    traj_path = Path(args.trajectories).expanduser().resolve()
    traj_dir = traj_path.parent
    qa_output = (
        Path(args.qa_output).expanduser().resolve()
        if args.qa_output
        else traj_dir / "synthesized_qa.jsonl"
    )
    train_output = (
        Path(args.train_output).expanduser().resolve()
        if args.train_output
        else traj_dir / "qa_train.jsonl"
    )
    sample_id = args.sample_id or traj_dir.name

    if config_path.suffix == ".json":
        config = SynthesisConfig.from_json(str(config_path))
    elif config_path.suffix in {".yaml", ".yml"}:
        config = SynthesisConfig.from_yaml(str(config_path))
    else:
        raise ValueError("Configuration file must be .json or .yaml format")

    records = _load_trajectory_file(traj_path)
    if not records:
        raise ValueError(f"No trajectory records loaded from: {traj_path}")

    trajectories = [
        _trajectory_from_record(rec, i)
        for i, rec in enumerate(records)
    ]
    if args.max_trajectories > 0:
        trajectories = trajectories[: args.max_trajectories]

    try:
        from trajectory_synthesis.core.synthesizer import QASynthesizer
    except ModuleNotFoundError as exc:
        if exc.name == "openai":
            raise ModuleNotFoundError(
                "Missing dependency 'openai'. Install it in your current Python "
                "environment or run this script with the same conda env used by "
                "sci_pipeline.py."
            ) from exc
        raise

    synthesizer = QASynthesizer(config)
    qa_pairs: List[Dict[str, Any]] = []
    conv_pairs: List[Dict[str, Any]] = []

    print(f"Loaded {len(trajectories)} trajectories from {traj_path}")
    print("Step: Synthesizing QA pairs...")

    for qa_idx, trajectory in enumerate(trajectories):
        try:
            qa = synthesizer.synthesize_qa(trajectory, qa_idx)
            if qa:
                qa_dict = qa.to_dict()
                qa_pairs.append(qa_dict)
                conv_pairs.append(_qa_to_conv(qa_dict, sample_id))
        except Exception as exc:
            print(f"  QA synthesis failed for trajectory {qa_idx}: {exc}")

    qa_output.parent.mkdir(parents=True, exist_ok=True)
    with qa_output.open("w", encoding="utf-8") as fh:
        for row in qa_pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_output.parent.mkdir(parents=True, exist_ok=True)
    with train_output.open("w", encoding="utf-8") as fh:
        for row in conv_pairs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved raw QA ({len(qa_pairs)} records): {qa_output}")
    print(f"Saved conversation QA ({len(conv_pairs)} records): {train_output}")
    print(f"sample_id: {sample_id}")


if __name__ == "__main__":
    main()
