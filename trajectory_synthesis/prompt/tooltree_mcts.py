from __future__ import annotations


AGENT_ROLE_PREAMBLE = """You are a scientific data exploration assistant that solves a \
question step by step with the help of a python interpreter tool. You first think about \
the reasoning process in your mind and then provide the final answer. During thinking, \
you can invoke the python interpreter to traverse folders, read files and compute facts \
about the dataset.

The reasoning process and final answer are enclosed within <think> </think> and \
<answer> </answer> tags respectively; python code and its result are enclosed within \
<python> </python> and <result> </result> tags respectively. After receiving a python \
result, you continue your reasoning from a new <think>. For example: \
<think> reasoning </think> <python> python code here </python> <result> tool output here </result> \
<think> more reasoning </think> <python> python code here </python> <result> tool output here </result> \
<think> final reasoning </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the <answer> block, the final exact answer MUST be enclosed within \\boxed{}.

Output-length discipline:
- Only inspect files/dirs under the provided dataset root path.
- Keep outputs concise: report key findings, not full file dumps.
- Never print huge recursive listings or thousands of matches/rows.
- For large outputs, print a small sample + aggregate counts/statistics."""


CANDIDATES_SYSTEM = (
    AGENT_ROLE_PREAMBLE
    + "\n\nYour current sub-task: propose distinct candidate <python> snippets for the "
    + "NEXT exploration step. Output ONLY a valid JSON array, no markdown."
)


STRATEGY_SYSTEM = (
    AGENT_ROLE_PREAMBLE
    + "\n\nYour current sub-task is STRATEGY PLANNING (no code yet). You will pick which "
    + "high-level exploration strategies to try next from a fixed catalog of action types. "
    + "Output ONLY a valid JSON array, no markdown, no prose."
)


REFINE_SYSTEM = (
    AGENT_ROLE_PREAMBLE
    + "\n\nYour current sub-task is CODE REFINEMENT. A high-level strategy has already been "
    + "selected; your job is to realize it as one self-contained Python snippet. "
    + "Output ONLY a valid JSON object, no markdown, no prose."
)


FINAL_ANSWER_SYSTEM_TEMPLATE = """You are a data exploration assistant that can solve the given question step by step with the help of the python interpreter tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you should invoke the python interpreter tool to traverse all folders and files and inspect their content comprehensively for fact information about specific purposes. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, and the python code and result are enclosed within <python> </python> and <result> </result> tags respectively.\
**IMPORTANT**: After receiving the python result, you should continue your reasoning process begin with <think>. \
For example, <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <python> python code here </python> <result> python interpreter result here </result> <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>. \
**NO UNTAGGED TEXT**: Your entire reply must consist only of content inside <think>, <python>, <result>, and <answer> blocks. Do not output any words or sentences outside those tags (only whitespace/newlines between adjacent tags is allowed). \
After you have gathered enough metadata, Do NOT exceed {max_python_times} python calls — use your remaining calls to write the metadata summary \
In the last part of the answer, the final exact answer is enclosed within \\boxed{{}}.

Dataset root path: {input_path}"""


CANDIDATES_USER_TEMPLATE = """[Task]
Propose distinct candidate Python snippets for the NEXT step of exploring this \
dataset. Each snippet must be self-contained.

[Seed / focus]
{seed_data}

[Dataset root]
{dataset_path}

[Exploration context]
{context_summary}

[Requirements]
- Return a JSON array of exactly {k} objects.
- Each object MUST be: {{"thought": "<brief reasoning>", "code": "<python code>", "action_type": "<one label from the catalog below>"}}
- Code must only use the dataset at path: {dataset_path!r}
- Vary strategies (list dirs, read headers, stats, plots data sample, etc.)
- Do NOT repeat the same code or trivial renames.

[Action-type catalog (pick ONE per candidate)]
{action_catalog}

Output ONLY the JSON array."""


STRATEGY_USER_TEMPLATE = """[Task]
You are planning the NEXT exploration step at a HIGH LEVEL. Propose distinct \
strategies (no code yet) from the catalog below. Each strategy should move the \
investigation meaningfully forward given the evidence already gathered.

[Seed / focus]
{seed_data}

[Dataset root]
{dataset_path}

[Exploration context so far]
{context_summary}

[Action-type catalog]
{action_catalog}

[Requirements]
- Return a JSON array of up to {k} objects.
- Each object MUST be: \
{{"action_type": "<exact label from catalog>", \
"thought": "<what this strategy would uncover, 1-2 sentences>", \
"prior": <number in [0,1] = your self-rated probability this step yields new, useful evidence>}}
- Strongly prefer DIVERSE action_types across the {k} proposals (do not propose \
the same type twice unless one is clearly dominant).
- Do NOT include actual Python code at this stage.
- Low prior (<=0.2) means "likely redundant / noisy / unlikely to help".
- High prior (>=0.7) means "almost certainly yields new, parseable facts".

Output ONLY the JSON array."""


REFINE_USER_TEMPLATE = """[Task]
Realize the chosen HIGH-LEVEL strategy as ONE self-contained Python snippet. \

[Seed / focus]
{seed_data}

[Dataset root]
{dataset_path}

[Exploration context so far]
{context_summary}

[Chosen strategy]
- action_type: {action_type}
- intent:      {strategy_thought}

[Requirements]
- Return ONE JSON object: {{"code": "<python code>", "confidence": <number in [0,1]>}}
- Code must be faithful to the chosen action_type (e.g. an "inspect_schema" step \
must actually read a header / dtypes, not just list a directory).
- Code must only use the dataset at path: {dataset_path!r}
- "confidence" is your belief the code correctly realizes the strategy.

Output ONLY the JSON object."""


PRE_JUDGE_TEMPLATE = """You score a PROPOSED next exploration step BEFORE it runs.
Return JSON: {{"score": <number 0-1>, "reason": "<short>"}}

[Seed]
{seed_data}

[Recent path summary]
{path_summary}

[Proposed code]
{code}

Score higher if the step is likely to reveal NEW structured facts about the data \
(structure, schema, key statistics, non-trivial patterns). Score lower if redundant, \
irrelevant to the seed, unsafe, or likely empty/error.)"""


POST_JUDGE_TEMPLATE = """You score an EXECUTED exploration observation AFTER it runs.
Return JSON: {{"score": <number 0-1>, "reason": "<short>"}}

[Seed]
{seed_data}

[Code that was run]
{code}

[Observation / tool output]
{observation}

Score higher if the output adds concrete, parseable information useful for later \
QA authoring. Score lower for tracebacks, empty output, or generic noise."""


FINAL_ANSWER_USER_TEMPLATE = """[Question]
{seed_data}

[Exploration trajectory so far]
The following tagged fragments are YOUR own prior turns in this conversation, \
produced step by step while exploring the dataset. Treat them as the authoritative \
evidence gathered so far:

{trajectory_text}

[Your next (and FINAL) turn]
You have already finished all python exploration you need. Continue the trajectory \
and produce ONLY the FINAL two tags — nothing else, no new <python> or <result> — \
in this exact order:

1. One <think>...</think> block: a compact, self-contained summary that ties \
together the key concrete facts surfaced in the <result> blocks above and explains \
how they lead to the final answer.
2. One <answer>...</answer> block: a comprehensive, well-structured metadata \
description that directly answers the question, grounded strictly in the evidence \
shown above. End the <answer> block with a final sentence of the form:
    The final answer is \\[ \\boxed{{<concise final answer>}} \\]
where the boxed content is a single concise statement (a number, file path, short \
phrase, or one-sentence summary).

Do NOT emit any text outside the <think> and <answer> tags."""


DEFAULT_ACTION_TYPES = [
    "list_dir",
    "inspect_schema",
    "inspect_content",
    "stats",
    "cross_file_join",
    "validate_claim",
    "visualize",
    "debug",
]


ACTION_TYPE_DESCRIPTIONS = {
    "list_dir": "Enumerate files/subdirectories; report names, sizes, extensions.",
    "inspect_schema": "Examine schema/columns/dtypes of a tabular file (csv/tsv/parquet/h5ad/jsonl).",
    "inspect_content": "Peek a small sample of rows or a metadata/JSON/README to understand semantics.",
    "stats": "Compute aggregate statistics (counts, unique values, min/max/mean, value distributions).",
    "cross_file_join": "Cross-reference / join columns across multiple files; verify key overlap or record counts.",
    "validate_claim": "Re-check a specific numeric fact or assertion surfaced by earlier steps.",
    "visualize": "Summarize a distribution via an in-text histogram / printed binning (no plotting libs if unavailable).",
    "debug": "Recover from a prior error; adjust encoding / delimiter / path to make the previous step succeed.",
}


def format_action_catalog(action_types=None) -> str:
    """Render the action-type catalog as a human-readable bullet list for prompts."""
    types = list(action_types) if action_types else list(DEFAULT_ACTION_TYPES)
    lines = []
    for t in types:
        desc = ACTION_TYPE_DESCRIPTIONS.get(t, "(custom action)")
        lines.append(f"- {t}: {desc}")
    return "\n".join(lines)


__all__ = [
    "AGENT_ROLE_PREAMBLE",
    "CANDIDATES_SYSTEM",
    "STRATEGY_SYSTEM",
    "REFINE_SYSTEM",
    "FINAL_ANSWER_SYSTEM_TEMPLATE",
    "CANDIDATES_USER_TEMPLATE",
    "STRATEGY_USER_TEMPLATE",
    "REFINE_USER_TEMPLATE",
    "PRE_JUDGE_TEMPLATE",
    "POST_JUDGE_TEMPLATE",
    "FINAL_ANSWER_USER_TEMPLATE",
    "DEFAULT_ACTION_TYPES",
    "ACTION_TYPE_DESCRIPTIONS",
    "format_action_catalog",
]
