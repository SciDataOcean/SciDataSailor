from __future__ import annotations


SCI_SYSTEM_INSTRUCTION = (
    "You are a scientific data exploration agent. "
    "Use the python_interpreter tool to explore dataset folders, inspect file "
    "contents, read tabular data, compute statistics, and build understanding "
    "of scientific datasets."
)


SCI_EXPLORATION_GOAL_BLOCK = """

[Exploration Goal]:
Based on the starting point content and available tools, conduct systematic exploration to collect and reason about valuable information from this scientific dataset.
Finally, I will synthesize a question and answer based on your collected information. Therefore, you should explore sufficient information for me.
"""


SCI_ALREADY_EXPLORED_BLOCK_TEMPLATE = """
[Already Explored Actions - Do NOT Repeat]:
The following tool calls (tool_name + parameters) have ALREADY been executed for this seed.
You MUST propose a NEW action that is NOT in this list or similar to them to increase the diversity of the exploration. Repeating any of them is strictly forbidden.
{used_actions_block}
"""


SCI_EXPLORATION_STRATEGY_BLOCK_TEMPLATE = """[Exploration Strategy and Focus]:
{sampling_tips}

"""


SCI_OUTPUT_FORMAT_INSTRUCTIONS = """
Based on the current state and available tools, select an appropriate tool and parameters, and generate the next thought and action.

IMPORTANT: Return ONLY a valid XML block without other words or markdown.
Format:
<thought>...</thought>
<tool_name>tool name</tool_name>
<parameters>{"param": "value"}</parameters>
"""


__all__ = [
    "SCI_SYSTEM_INSTRUCTION",
    "SCI_EXPLORATION_GOAL_BLOCK",
    "SCI_ALREADY_EXPLORED_BLOCK_TEMPLATE",
    "SCI_EXPLORATION_STRATEGY_BLOCK_TEMPLATE",
    "SCI_OUTPUT_FORMAT_INSTRUCTIONS",
]
