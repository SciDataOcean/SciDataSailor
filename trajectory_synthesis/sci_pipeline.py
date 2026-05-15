import argparse
import asyncio
import atexit
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import yaml

from trajectory_synthesis.core.config import SynthesisConfig
from trajectory_synthesis.core.react_sampler import ReactTrajectorySampler
from trajectory_synthesis.core.selector import TrajectorySelector
from trajectory_synthesis.core.synthesizer import QASynthesizer
from trajectory_synthesis.core.tooltree_mcts_sampler import ToolTreeMCTSSampler
from src.llm_client import LLMClient
from src.tools.python_tool import PythonTool

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "results"


class _TeeTextIO:
    """Duplicate writes to multiple text streams; flush every write for live logs."""

    __slots__ = ("_primary", "_extra")

    def __init__(self, primary: TextIO, extra: TextIO) -> None:
        self._primary = primary
        self._extra = extra

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._primary.flush()
        self._extra.write(s)
        self._extra.flush()
        return len(s)

    def flush(self) -> None:
        self._primary.flush()
        self._extra.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()

    def fileno(self) -> int:
        return self._primary.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8")


def _install_log_tee(log_file: str) -> None:
    """Mirror stdout/stderr to *log_file* (truncate each run, flush each write)."""
    path = Path(log_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeTextIO(sys.stdout, log_fp)
    sys.stderr = _TeeTextIO(sys.stderr, log_fp)

    def _close_log() -> None:
        try:
            log_fp.close()
        except Exception:
            pass

    atexit.register(_close_log)


def generate_source_id(seed_content: str, seed_idx: int) -> str:
    content_hash = hashlib.md5(seed_content.encode("utf-8")).hexdigest()[:8]
    return f"src_{seed_idx:04d}_{content_hash}"


class SciSynthesisPipeline:
    """Scientific data QA synthesis pipeline."""

    def __init__(
        self,
        config: SynthesisConfig,
        llm_client: LLMClient,
        python_tool: PythonTool,
        output_dir: Optional[str] = None,
        dataset_path: str = "",
    ):
        self.config = config
        self.llm_client = llm_client
        self.python_tool = python_tool
        self.dataset_path = dataset_path

        resolved_output_dir = Path(output_dir or DEFAULT_RESULTS_DIR).resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(resolved_output_dir)

        self.qa_file_path = str(resolved_output_dir / "synthesized_qa.jsonl")
        self.traj_file_path = str(resolved_output_dir / "trajectories.jsonl")

        print("Output files:")
        print(f"   QA: {self.qa_file_path}")
        print(f"   Trajectories: {self.traj_file_path}")

    async def run_async(self, seeds: List[Dict[str, Any]]):
        if self.config.number_of_seed is not None:
            seeds = seeds[: self.config.number_of_seed]

        print(f"\n{'=' * 80}")
        print("Scientific Data QA Synthesis Pipeline")
        print(f"{'=' * 80}")
        print(f"Total seeds: {len(seeds)}")
        print(f"Model: {self.config.model_name}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"{'=' * 80}\n")

        synthesizer = QASynthesizer(self.config)

        for seed_idx, seed in enumerate(seeds, 1):
            seed_content = seed.get("content", "")
            seed_kwargs = seed.get("kwargs", {})
            source_id = generate_source_id(seed_content, seed_idx)

            print(f"\n{'#' * 60}")
            print(f"Processing Seed {seed_idx}/{len(seeds)}")
            print(f"Source ID: {source_id}")
            print(f"Seed: {seed_content[:100]}...")
            if seed_kwargs:
                print(f"Kwargs: {seed_kwargs}")
            print(f"{'#' * 60}\n")

            try:
                selected_trajectories = []

                mode = getattr(self.config, "sampler_mode", "react")
                if mode == "tooltree_mcts":
                    print(
                        "Step 1: ToolTree-style dual-feedback MCTS tree sampling..."
                    )
                    mcts_sampler = ToolTreeMCTSSampler(
                        self.llm_client,
                        self.python_tool,
                        self.config,
                        dataset_path=self.dataset_path,
                    )
                    nodes = await mcts_sampler.sample_trajectory_tree(
                        seed_content, seed_kwargs
                    )
                    if not mcts_sampler.root_id or len(nodes) <= 1:
                        print("No exploration beyond root, skipping seed")
                        continue

                    print("\nStep 2: Selecting trajectories from exploration tree...")
                    selector = TrajectorySelector(self.config)
                    selected_trajectories = selector.select_trajectories(
                        nodes,
                        mcts_sampler.root_id,
                        seed_content,
                        source_id,
                    )
                else:
                    print("Step 1: ReAct trajectory sampling...")
                    sampler = ReactTrajectorySampler(
                        self.llm_client,
                        self.python_tool,
                        self.config,
                        dataset_path=self.dataset_path,
                    )
                    trajectories = await sampler.sample_trajectories(
                        seed_content, seed_kwargs
                    )

                    if not trajectories:
                        print("No trajectories sampled, skipping seed")
                        continue

                    print(f"\nSampled {len(trajectories)} trajectory chains")

                    print("\nStep 2: Selecting trajectories...")
                    selected_path_sets: list = []
                    similarity_threshold = getattr(
                        self.config, "path_similarity_threshold", 0.7
                    )
                    for traj in trajectories:
                        current_set = {n.node_id for n in traj.nodes}
                        too_similar = False
                        for prev_set in selected_path_sets:
                            inter = len(current_set & prev_set)
                            union = len(current_set | prev_set)
                            if union and inter / union > similarity_threshold:
                                too_similar = True
                                break
                        if not too_similar:
                            traj.source_id = source_id
                            traj.trajectory_id = (
                                f"{source_id}_traj_{len(selected_trajectories)}"
                            )
                            selected_trajectories.append(traj)
                            selected_path_sets.append(current_set)
                        if len(selected_trajectories) >= self.config.max_selected_traj:
                            break

                if not selected_trajectories:
                    print("No trajectories selected, skipping seed")
                    continue

                print(f"Selected {len(selected_trajectories)} trajectories")

                print("\nStep 3: Synthesizing QA pairs...")
                qa_pairs = []
                for qa_idx, trajectory in enumerate(selected_trajectories):
                    try:
                        qa = synthesizer.synthesize_qa(trajectory, qa_idx)
                        if qa:
                            qa_pairs.append(qa.to_dict())
                    except Exception as exc:
                        print(f"  QA synthesis failed for trajectory {qa_idx}: {exc}")

                if qa_pairs:
                    self._save_qa_pairs(qa_pairs)
                    print(
                        f"\nSeed {seed_idx} complete! Generated {len(qa_pairs)} QA pairs"
                    )

                trajectories_data = [traj.to_dict() for traj in selected_trajectories]
                if trajectories_data:
                    self._save_trajectories(trajectories_data)

            except Exception as exc:
                print(f"\nError processing seed {seed_idx}: {exc}")
                import traceback

                traceback.print_exc()
                continue

        print(f"\n\n{'=' * 80}")
        print("Synthesis Complete!")
        print(f"{'=' * 80}")
        print(f"Total seeds processed: {len(seeds)}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 80}\n")

    def run(self, seeds: List[Dict[str, Any]]):
        asyncio.run(self.run_async(seeds))

    def _save_qa_pairs(self, qa_pairs: List[Dict[str, Any]]):
        with open(self.qa_file_path, "a", encoding="utf-8") as file:
            for qa in qa_pairs:
                file.write(json.dumps(qa, ensure_ascii=False) + "\n")

    @staticmethod
    def _load_trajectory_file(path: str) -> List[Dict[str, Any]]:
        """Load trajectory file: JSON array (preferred) or legacy JSONL (one object per line)."""
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return []
        with open(path, "r", encoding="utf-8") as file:
            raw = file.read().strip()
        if not raw:
            return []
        if raw.startswith("["):
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        # Legacy: one JSON object per line
        items: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
        return items

    def _save_trajectories(self, trajectories: List[Dict[str, Any]]) -> None:
        """Append trajectories to a single JSON array; pretty-print so each key is on its own line."""
        existing = self._load_trajectory_file(self.traj_file_path)
        existing.extend(trajectories)
        with open(self.traj_file_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(existing, indent=2, ensure_ascii=False))
            file.write("\n")


def _resolve_env_vars(value: str) -> str:
    if not isinstance(value, str):
        return value
    import re

    def _replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(r"\$\{(\w+)\}", _replace, value)


def _load_raw_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        if config_path.suffix == ".json":
            return json.load(file)
        if config_path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(file)
    raise ValueError("Configuration file must be .json or .yaml format")


def _resolve_path(path_value: Optional[str], base_dir: Path) -> Optional[str]:
    if not path_value:
        return path_value
    expanded = Path(_resolve_env_vars(path_value)).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str((base_dir / expanded).resolve())


def main():
    parser = argparse.ArgumentParser(description="Scientific Data QA Synthesis Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration file path (.json or .yaml)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seed data JSONL file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--endpoints",
        type=str,
        nargs="+",
        default=None,
        help="LLM API endpoint URL(s)",
    )
    parser.add_argument(
        "--api_keys",
        type=str,
        nargs="+",
        default=None,
        help="API key(s) for LLM endpoints",
    )
    parser.add_argument(
        "--conda_path",
        type=str,
        default=None,
        help="Path to conda installation (e.g. /home/user/miniconda3)",
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        default=None,
        help="Conda environment name for PythonTool",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Root path of the scientific dataset to explore",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Mirror stdout/stderr to this file each run (overwrites); line-buffered for live tail",
    )

    args = parser.parse_args()

    if args.log_file:
        _install_log_tee(args.log_file)

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    print(f"Loading configuration: {config_path}")

    if config_path.suffix == ".json":
        config = SynthesisConfig.from_json(str(config_path))
    elif config_path.suffix in {".yaml", ".yml"}:
        config = SynthesisConfig.from_yaml(str(config_path))
    else:
        raise ValueError("Configuration file must be .json or .yaml format")

    raw_config = _load_raw_config(config_path)

    endpoints = args.endpoints or raw_config.get("endpoints")
    if not endpoints:
        endpoints = [_resolve_env_vars(config.base_url)]
    else:
        endpoints = [_resolve_env_vars(endpoint) for endpoint in endpoints]

    api_keys = args.api_keys or raw_config.get("api_keys")
    if not api_keys:
        api_keys = [_resolve_env_vars(config.api_key)]
    else:
        api_keys = [_resolve_env_vars(api_key) for api_key in api_keys]

    conda_path = _resolve_env_vars(args.conda_path or raw_config.get("conda_path", ""))
    conda_env = args.conda_env or raw_config.get("conda_env", "")
    if not conda_path or not conda_env:
        raise ValueError(
            "conda_path and conda_env must be provided (via config or CLI)"
        )

    dataset_path = _resolve_path(
        args.dataset_path or raw_config.get("dataset_path", ""),
        config_dir,
    ) or ""

    seeds_value = args.seeds or config.seeds_file or os.environ.get("SEEDS_FILE")
    if not seeds_value:
        raise ValueError(
            "Missing seeds path: specify via --seeds, config file, or SEEDS_FILE env var"
        )
    seeds_path = _resolve_path(seeds_value, Path.cwd() if args.seeds else config_dir)

    output_dir = _resolve_path(
        args.output_dir or config.output_dir or os.environ.get("OUTPUT_DIR") or str(DEFAULT_RESULTS_DIR),
        Path.cwd() if args.output_dir else config_dir,
    )

    print(f"Endpoints: {endpoints}")
    print(f"Model: {config.model_name}")
    print(f"Conda: {conda_path}/envs/{conda_env}")
    print(f"Dataset path: {dataset_path}")
    print(f"Reading seeds from: {seeds_path}")
    print(f"Output directory: {output_dir}")

    seeds = []
    with open(seeds_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            seed = json.loads(line)
            if not isinstance(seed, dict):
                raise ValueError(
                    f"Each seed must be a dict, got: {type(seed).__name__}"
                )
            if "content" not in seed:
                raise ValueError(f"Seed missing 'content' field: {seed}")
            seed.setdefault("kwargs", {})
            seeds.append(seed)

    print(f"Loaded {len(seeds)} seeds")

    llm_client = LLMClient(
        endpoints=endpoints,
        api_keys=api_keys,
        default_model=config.model_name,
    )
    python_tool = PythonTool(
        conda_path=conda_path,
        conda_env=conda_env,
    )
    if conda_env == "base":
        python_tool.python_path = f"{conda_path}/bin/python"
        print(f"  (base env detected, python path: {python_tool.python_path})")

    pipeline = SciSynthesisPipeline(
        config=config,
        llm_client=llm_client,
        python_tool=python_tool,
        output_dir=output_dir,
        dataset_path=dataset_path,
    )
    pipeline.run(seeds)

    print("\nAll done!")


if __name__ == "__main__":
    main()
