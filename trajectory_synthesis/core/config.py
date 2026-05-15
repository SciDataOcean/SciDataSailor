"""
Configuration management for the scientific QA synthesis pipeline.
"""

import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SynthesisConfig:
    """Scientific QA synthesis configuration."""

    seeds_file: Optional[str] = None
    output_dir: Optional[str] = None

    available_tools: List[str] = field(default_factory=list)
    qa_examples: List[Dict[str, str]] = field(default_factory=list)
    sampling_tips: str = ""
    synthesis_tips: str = ""
    seed_description: str = ""

    model_name: str = "gpt-4.1-2025-04-14"
    api_key: str = ""
    base_url: str = ""

    max_depth: int = 5 # maximum trajectory depth
    branching_factor: int = 2 # number of child branches sampled per node before the depth threshold
    depth_threshold: int = 3 # once depth reaches this threshold, branching narrows to one child per node

    min_depth: int = 2 # minimum leaf depth
    max_selected_traj: int = 3 # maximum number of trajectories kept per seed
    path_similarity_threshold: float = 0.7 # diversity control during selection

    # ReAct sampler parameters
    max_steps: int = 15 # maximum number of steps per trajectory
    n_trajectories: int = 3 # number of independent ReAct chains for one seed

    # Sampler backend: "react" (linear chains) or "tooltree_mcts" (dual-feedback tree)
    sampler_mode: str = "react"

    # ToolTree-inspired MCTS for dataset exploration (python_interpreter steps)
    mcts_rollouts: int = 24 # outer MCTS iterations (each may run up to mcts_branching python steps)
    mcts_lambda: float = 1.0 # UCT prior weight on r_pre (Eq. 1 style)
    mcts_tau_pre: float = 0.25 # drop candidate steps with r_pre below this before execution
    mcts_tau_post: float = 0.35 # mark executed step non-expandable if r_post below this
    mcts_num_candidates: int = 4 # LLM proposes this many candidates; pre-judge keeps survivors
    mcts_branching: int = 1 # how many top pre-scored survivors to execute and attach as siblings per expansion (1 => chain)
    mcts_max_depth: int = 8 # max depth from root (action steps); aligns with exploration budget
    mcts_judge_temperature: float = 0.2 # LLM judge temperature for pre/post scores
    mcts_fpu_reduction: float = 0.1 # DF-FPU: penalty subtracted from the sibling-calibrated r_pre used as the first-play Q of unvisited children (clipped at 0). 0 = aggressive first-play; 1 = near-legacy "visit every sibling first" behavior.
    mcts_depth_bonus: float = 0.05 # DF-FPU: per-depth-level reward added to every node's UCT score, to bias the search toward deeper trajectories. 0 disables; 0.05 is a sane default for rewards in [0,1].
    mcts_visit_decay: str = "sqrt" # DF-FPU: "sqrt" (1/sqrt(1+v), recommended) softens sibling rivalry; "linear" (1/(1+v), classic UCT) keeps textbook behavior.

    # Entropy-driven dynamic branching (Innovation: adaptive child count)
    mcts_dynamic_branching: bool = True # if True, the number of siblings created at each expansion scales with the entropy of the candidate r_pre distribution. Low entropy (LLM confident in one candidate) -> fewer children; high entropy -> up to mcts_branching.
    mcts_entropy_temperature: float = 0.3 # softmax temperature applied to priors before computing entropy. Lower = priors sharpened (entropy collapses to 1 child easier); higher = priors flattened.
    mcts_min_branching: int = 1 # floor on the dynamic-branching output. Set to 2+ if you always want at least some breadth even on confident steps.

    mcts_entropy_prior_weight: float = 0.5 # weight on the prior (r_pre softmax) entropy term
    mcts_entropy_token_weight: float = 0.5 # weight on the token-level (logprob) entropy term
    mcts_entropy_token_max_tokens: int = 20 # number of leading response tokens over which token-level entropy is aggregated (mirrors ARPO's 20-token window)
    mcts_entropy_top_logprobs: int = 5 # top-k logprobs requested per token for token-level entropy (mirrors ARPO's logprobs=10)
    mcts_entropy_use_token_logprobs: bool = True # if False, skip logprobs requests and fall back to prior-only entropy (useful for backends that don't support logprobs)

    # Hierarchical action space (Innovation: strategy-then-code MCTS)
    mcts_hierarchical_actions: bool = False # if True, expansion becomes two-phase: (1) LLM picks high-level action_types from a catalog, (2) LLM refines each chosen strategy into concrete Python code. Siblings are diversified at the strategy level.
    mcts_action_types: List[str] = field(default_factory=list) # override the default action catalog (see prompt/tooltree_mcts.DEFAULT_ACTION_TYPES). Empty = use built-in catalog.
    mcts_strategy_diversity: bool = True # try to pick siblings with DIFFERENT action_types first before revisiting the same type. Applies to hierarchical mode and (best-effort) flat mode when candidates are tagged.

    # Token/context budget controls for LLM calls in tooltree_mcts.
    # Upper bound per API call is 8192 output tokens.
    mcts_llm_max_tokens: int = 2048 # default max output tokens for proposal/judge/refine calls (<=8192)
    mcts_final_answer_max_tokens: int = 8192 # max output tokens for final answer call (<=8192)
    mcts_final_context_max_steps: int = 8 # keep only most recent N steps in final-answer context
    mcts_final_context_max_chars: int = 60000 # hard cap for final-answer trajectory_text chars

    # Observation (python tool output) truncation before feeding it back into prompts.
    mcts_obs_max_chars: int = 6000 # hard char cap per tool observation; longer outputs are truncated
    mcts_obs_head_chars: int = 4000 # how many leading chars to keep verbatim when truncating
    mcts_obs_tail_chars: int = 1500 # how many trailing chars to keep verbatim when truncating
    mcts_obs_truncate_notice: str = "... the response is too long to show ..." # marker inserted at truncation point

    number_of_seed: Optional[int] = None

    resource_types: List[str] = field(default_factory=list)
    resource_init_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    sandbox_server_url: str = "http://127.0.0.1:18890"
    sandbox_auto_start: bool = True
    sandbox_config_path: Optional[str] = None
    sandbox_timeout: int = 120

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SynthesisConfig":
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"config_dict must be dict, got: {type(config_dict).__name__}"
            )

        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}

        def _normalize_text_field(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "\n".join("" if item is None else str(item) for item in value)
            return str(value)

        if "sampling_tips" in filtered:
            filtered["sampling_tips"] = _normalize_text_field(
                filtered.get("sampling_tips")
            )
        if "synthesis_tips" in filtered:
            filtered["synthesis_tips"] = _normalize_text_field(
                filtered.get("synthesis_tips")
            )

        return cls(**filtered)

    @classmethod
    def from_json(cls, json_path: str) -> "SynthesisConfig":
        with open(json_path, "r", encoding="utf-8") as file:
            config_dict = json.load(file)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SynthesisConfig":
        with open(yaml_path, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available_tools": self.available_tools,
            "qa_examples": self.qa_examples,
            "sampling_tips": self.sampling_tips,
            "synthesis_tips": self.synthesis_tips,
            "seed_description": self.seed_description,
            "seeds_file": self.seeds_file,
            "output_dir": self.output_dir,
            "model_name": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_depth": self.max_depth,
            "branching_factor": self.branching_factor,
            "depth_threshold": self.depth_threshold,
            "min_depth": self.min_depth,
            "max_selected_traj": self.max_selected_traj,
            "number_of_seed": self.number_of_seed,
            "path_similarity_threshold": self.path_similarity_threshold,
            "resource_types": self.resource_types,
            "resource_init_configs": self.resource_init_configs,
            "sandbox_server_url": self.sandbox_server_url,
            "sandbox_auto_start": self.sandbox_auto_start,
            "sandbox_config_path": self.sandbox_config_path,
            "sandbox_timeout": self.sandbox_timeout,
        }

