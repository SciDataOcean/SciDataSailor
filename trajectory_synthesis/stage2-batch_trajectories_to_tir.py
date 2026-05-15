from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "trajectory_synthesis" / "results"
DOMAIN_DIRS = (
    RESULTS_ROOT / "earth_science",
    RESULTS_ROOT / "lifesci_data",
    RESULTS_ROOT / "physical_science",
)


def discover_trajectories(roots: Iterable[Path]) -> List[Path]:
    trajectories: List[Path] = []
    for root in roots:
        if not root.is_dir():
            print(f"[warn] skip missing directory: {root}")
            continue
        for p in sorted(root.glob("*/trajectories.jsonl")):
            if p.is_file():
                trajectories.append(p.resolve())
    return trajectories


def build_command(trajectory_file: Path, qc_extra: List[str]) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "trajectory_synthesis.trajectories_to_tir",
        "--input",
        str(trajectory_file),
        "--run-quality-check",
        "--qc-output-dir",
        str(trajectory_file.parent / "quality_output_v2"),
        "--export-training-data"
    ]
    if qc_extra:
        cmd.extend(["--qc-extra", *qc_extra])
    return cmd


def run_one(trajectory_file: Path, qc_extra: List[str]) -> int:
    cmd = build_command(trajectory_file, qc_extra)
    print(f"\n=== Processing: {trajectory_file} ===")
    print("$ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return completed.returncode


def parse_args_and_qc_extra() -> tuple[argparse.Namespace, List[str]]:
    argv = sys.argv[1:]
    qc_extra: List[str] = []
    if "--qc-extra" in argv:
        idx = argv.index("--qc-extra")
        qc_extra = argv[idx + 1 :]
        argv = argv[:idx]
        if qc_extra and qc_extra[0] == "--":
            qc_extra = qc_extra[1:]

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Extra QC flags: append them after --qc-extra, e.g.\n"
            "  --qc-extra --sft_threshold 30 --rl_threshold 12\n"
            "or\n"
            "  --qc-extra -- --sft_threshold 30 --rl_threshold 12"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately when one dataset fails.",
    )
    args = parser.parse_args(argv)
    return args, qc_extra


def main() -> None:
    args, qc_extra = parse_args_and_qc_extra()

    trajectory_files = discover_trajectories(DOMAIN_DIRS)
    if not trajectory_files:
        print("[warn] no trajectories.jsonl files found.")
        return

    print(f"Found {len(trajectory_files)} dataset(s).")

    failed: List[Path] = []
    for f in trajectory_files:
        code = run_one(f, qc_extra=qc_extra)
        if code != 0:
            failed.append(f)
            print(f"[error] failed ({code}): {f}")
            if args.stop_on_error:
                break

    if failed:
        print(f"\nFinished with {len(failed)} failure(s).")
        for f in failed:
            print(f" - {f}")
        sys.exit(1)

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":
    main()
