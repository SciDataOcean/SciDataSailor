"""Core building blocks for scientific QA synthesis.

Keep this package import lightweight. Some modules (e.g. samplers) pull optional
runtime dependencies, so symbols are loaded lazily via ``__getattr__``.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

__all__ = [
    "SynthesisConfig",
    "SynthesizedQA",
    "Trajectory",
    "TrajectoryNode",
    "ReactTrajectorySampler",
    "SciTrajectorySampler",
    "ToolTreeMCTSSampler",
    "TrajectorySelector",
    "QASynthesizer",
]

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "SynthesisConfig": (".config", "SynthesisConfig"),
    "SynthesizedQA": (".models", "SynthesizedQA"),
    "Trajectory": (".models", "Trajectory"),
    "TrajectoryNode": (".models", "TrajectoryNode"),
    "ReactTrajectorySampler": (".react_sampler", "ReactTrajectorySampler"),
    "SciTrajectorySampler": (".sci_sampler", "SciTrajectorySampler"),
    "ToolTreeMCTSSampler": (".tooltree_mcts_sampler", "ToolTreeMCTSSampler"),
    "TrajectorySelector": (".selector", "TrajectorySelector"),
    "QASynthesizer": (".synthesizer", "QASynthesizer"),
}


def __getattr__(name: str) -> Any:
    entry = _LAZY_IMPORTS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = entry
    module = import_module(module_name, __name__)
    value = getattr(module, symbol)
    globals()[name] = value
    return value

