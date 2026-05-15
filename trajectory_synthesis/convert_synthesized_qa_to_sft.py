from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trajectory_synthesis.prompt.tooltree_mcts import AGENT_ROLE_PREAMBLE


DEFAULT_RESULTS_ROOT = "<RESULTS_ROOT>"
DEFAULT_START = "<START_DATASET>"
DEFAULT_END = "<END_DATASET>"
# Keep numeric type for argparse; <=0 means disabling strict count check.
DEFAULT_EXPECTED_DATASETS = -1


ANSWER_TAG_RE = re.compile(r"<answer>\s*.*?\s*</answer>", re.DOTALL | re.IGNORECASE)


def _list_dataset_dirs(results_root: Path) -> List[Path]:
    return sorted([p for p in results_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def _slice_dataset_range(dataset_dirs: List[Path], start_name: str, end_name: str) -> List[Path]:
    names = [p.name for p in dataset_dirs]
    if start_name not in names:
        raise ValueError(f"start dataset not found: {start_name}")
    if end_name not in names:
        raise ValueError(f"end dataset not found: {end_name}")
    i = names.index(start_name)
    j = names.index(end_name)
    if i > j:
        raise ValueError(f"start appears after end: {start_name} > {end_name}")
    return dataset_dirs[i : j + 1]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl_safe(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
    except (OSError, json.JSONDecodeError):
        return []
    return out


def _build_gpt_output(rec: Dict[str, Any]) -> str:
    steps = rec.get("reasoning_steps")
    if isinstance(steps, list) and steps:
        text = "\n".join(str(s) for s in steps if s is not None).strip()
    else:
        text = ""

    answer = str(rec.get("answer", "") or "").strip()
    if not text:
        if answer:
            return f"<think>Based on the evidence, I can answer directly.</think>\n<answer>{answer}</answer>"
        return "<think>No valid reasoning steps were provided.</think>\n<answer></answer>"

    if answer and not ANSWER_TAG_RE.search(text):
        text = f"{text}\n<answer>{answer}</answer>"
    return text


def _build_sample_id(
    rec: Dict[str, Any],
    fallback_sample_id: str,
    dataset_name: str,
    sample_id_mode: str,
) -> str:
    raw_sample_id = str(
        rec.get("qa_id")
        or rec.get("sample_id")
        or rec.get("trajectory_id")
        or fallback_sample_id
    )
    if sample_id_mode == "dataset":
        return dataset_name
    return f"{dataset_name}__{raw_sample_id}"


def _convert_record(
    rec: Dict[str, Any],
    fallback_sample_id: str,
    dataset_name: str,
    sample_id_mode: str,
    system_prompt: str,
) -> Dict[str, Any]:
    sample_id = _build_sample_id(
        rec,
        fallback_sample_id=fallback_sample_id,
        dataset_name=dataset_name,
        sample_id_mode=sample_id_mode,
    )
    human_value = str(rec.get("question", "") or "").strip()
    gpt_value = _build_gpt_output(rec)
    return {
        "sample_id": sample_id,
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value},
        ],
        "system": system_prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help="Default results root.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=AGENT_ROLE_PREAMBLE,
        help="System prompt written into each SFT record.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if a dataset has no synthesized_qa.jsonl.",
    )
    parser.add_argument("--start-dataset", type=str, default=DEFAULT_START)
    parser.add_argument("--end-dataset", type=str, default=DEFAULT_END)
    parser.add_argument(
        "--expected-datasets",
        type=int,
        default=DEFAULT_EXPECTED_DATASETS,
        help="Expected number of dataset dirs in [start, end] (set <=0 to disable check).",
    )
    parser.add_argument(
        "--sample-id-mode",
        type=str,
        choices=["dataset_prefixed", "dataset"],
        default="dataset_prefixed",
        help=(
            "How to generate sample_id: "
            "'dataset_prefixed' => <dataset>__<original_id>, "
            "'dataset' => <dataset>."
        ),
    )
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    if (root / "synthesized_qa.jsonl").is_file():
        dataset_dirs = [root]
    else:
        all_dirs = _list_dataset_dirs(root)
        dataset_dirs = _slice_dataset_range(all_dirs, args.start_dataset, args.end_dataset)
        if args.expected_datasets > 0 and len(dataset_dirs) != args.expected_datasets:
            raise ValueError(
                f"dataset count mismatch in range: got {len(dataset_dirs)}, "
                f"expected {args.expected_datasets}"
            )
    converted_datasets = 0
    total_records = 0
    missing: List[str] = []

    for ds in dataset_dirs:
        src = ds / "synthesized_qa.jsonl"
        if not src.is_file():
            missing.append(ds.name)
            continue

        rows = _load_jsonl(src)
        out_rows = [
            _convert_record(
                rec,
                fallback_sample_id=f"{ds.name}_{idx}",
                dataset_name=ds.name,
                sample_id_mode=args.sample_id_mode,
                system_prompt=args.system_prompt,
            )
            for idx, rec in enumerate(rows)
        ]
        dst = ds / "training_data" / "sft_train.jsonl"
        existing_rows = _load_jsonl_safe(dst)
        merged_rows = existing_rows + out_rows
        _dump_jsonl(dst, merged_rows)
        converted_datasets += 1
        total_records += len(out_rows)
        print(
            f"[ok] {ds.name}: existing={len(existing_rows)} appended={len(out_rows)} "
            f"total={len(merged_rows)} -> {dst}"
        )

    if missing:
        msg = f"missing synthesized_qa.jsonl in {len(missing)} dataset(s): {', '.join(missing)}"
        if args.strict_missing:
            raise FileNotFoundError(msg)
        print(f"[warn] {msg}")

    print(f"[done] converted_datasets={converted_datasets} total_sft_records={total_records}")


if __name__ == "__main__":
    main()

