"""Prompt templates for the scientific QA synthesis samplers.

Each submodule owns the prompts for a single sampler, keeping the templates
out of the ``core/`` module so they can be audited, versioned, or swapped
independently of the runtime logic:

- :mod:`trajectory_synthesis.prompt.tooltree_mcts` — ToolTree-style MCTS.
- :mod:`trajectory_synthesis.prompt.react` — ReAct loop.
- :mod:`trajectory_synthesis.prompt.sci` — single-shot XML agent.
- :mod:`trajectory_synthesis.prompt.tools` — shared tool schemas.
"""

from __future__ import annotations

from .react import REACT_SYSTEM_PROMPT, REACT_TASK_TEMPLATE
from .qa_synthesis import QA_SYNTHESIS_BASE_TEMPLATE, build_qa_synthesis_prompt
from .sci import (
    SCI_ALREADY_EXPLORED_BLOCK_TEMPLATE,
    SCI_EXPLORATION_GOAL_BLOCK,
    SCI_EXPLORATION_STRATEGY_BLOCK_TEMPLATE,
    SCI_OUTPUT_FORMAT_INSTRUCTIONS,
    SCI_SYSTEM_INSTRUCTION,
)
from .tools import PYTHON_TOOL_SCHEMA
from .tooltree_mcts import (
    AGENT_ROLE_PREAMBLE,
    CANDIDATES_SYSTEM,
    CANDIDATES_USER_TEMPLATE,
    FINAL_ANSWER_SYSTEM_TEMPLATE,
    FINAL_ANSWER_USER_TEMPLATE,
    POST_JUDGE_TEMPLATE,
    PRE_JUDGE_TEMPLATE,
)

__all__ = [
    # shared
    "PYTHON_TOOL_SCHEMA",
    # tooltree_mcts
    "AGENT_ROLE_PREAMBLE",
    "CANDIDATES_SYSTEM",
    "FINAL_ANSWER_SYSTEM_TEMPLATE",
    "CANDIDATES_USER_TEMPLATE",
    "PRE_JUDGE_TEMPLATE",
    "POST_JUDGE_TEMPLATE",
    "FINAL_ANSWER_USER_TEMPLATE",
    # react
    "REACT_SYSTEM_PROMPT",
    "REACT_TASK_TEMPLATE",
    # qa_synthesis
    "QA_SYNTHESIS_BASE_TEMPLATE",
    "build_qa_synthesis_prompt",
    # sci
    "SCI_SYSTEM_INSTRUCTION",
    "SCI_EXPLORATION_GOAL_BLOCK",
    "SCI_ALREADY_EXPLORED_BLOCK_TEMPLATE",
    "SCI_EXPLORATION_STRATEGY_BLOCK_TEMPLATE",
    "SCI_OUTPUT_FORMAT_INSTRUCTIONS",
]
