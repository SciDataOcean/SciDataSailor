"""Prompt templates for QA synthesis from selected trajectories."""

from __future__ import annotations

from typing import Any, Dict, List


QA_SYNTHESIS_BASE_TEMPLATE = """You are a data synthesis expert. Based on the following Agent's ReAct exploration trajectory (Thought→Action→Observation chain), synthesize a high-quality Q&A pair.

【Starting Point Information】
Content: {seed_data}{seed_description_block}

【Complete ReAct Exploration Trajectory】
{traj_description}

{synthesis_tips_block}{qa_examples_block}Please synthesize a high-quality Q&A pair based on the trajectory:

## Question Requirements:
- The target answer must be a specific fact (name, date, location, count, yes/no).
- DO NOT ask "How", "Why", or "Describe" questions requiring long explanations.
- Anti-shortcut: Question MUST NOT contain the answer text.
- Low-entrance, deep-reasoning: Keep question to <=2 sentences; depth from multi-hop dependency chain.
- Deep multi-hop (required): Question must require >=3 dependent hops to solve.
- Question should be natural and self-contained (don't mention "agent", "trajectory", "search results").

## Answer Requirements:
- Extreme Brevity: Answer MUST be <=1 sentence, ideally just a short phrase.
- No Fluff: No filler words like "According to..." or "The answer is...".
- Groundedness: Must be strictly derived from trajectory observations.

Return JSON EXACTLY in this schema:
{{
  "question": "question text",
  "answer": "short phrase or single sentence",
  "reasoning_steps": [
    "<think>...</think> <python>...</python> <result>...</result>",
    "<think>...</think> <python>...</python> <result>...</result>",
    "...",
    "<think>...</think> <answer>...</answer>"
  ]
}}

Format constraints for reasoning_steps:
- Each item MUST be a tagged string, not a JSON object/dict.
- Intermediate steps use think + python + result tags.
- Final step uses think + answer tags.
- Keep final step answer text consistent with top-level "answer".{regen_block}
"""


def build_qa_synthesis_prompt(
    seed_data: str,
    seed_description: str,
    traj_description: str,
    synthesis_tips: str,
    qa_examples: List[Dict[str, Any]],
    last_failure_reason: str,
) -> str:
    seed_description_block = (
        f"\nDescription: {seed_description}" if seed_description else ""
    )
    synthesis_tips_block = (
        f"Data Synthesis Guidance:\n{synthesis_tips}\n\n" if synthesis_tips else ""
    )

    qa_examples_block = ""
    if qa_examples:
        qa_examples_block = "Refer to the style and quality of the following examples:\n\n"
        for i, example in enumerate(qa_examples, 1):
            qa_examples_block += (
                f"Example {i}:\n"
                f"Question: {example.get('question', '')}\n"
                f"Answer: {example.get('answer', '')}\n\n"
            )

    regen_block = ""
    if last_failure_reason:
        regen_block = (
            "\n\n[Regeneration Required - Previous Output Rejected]\n"
            f"Reason: {last_failure_reason}\n"
            "You MUST regenerate a NEW question/answer that fully satisfies the guidance."
        )

    return QA_SYNTHESIS_BASE_TEMPLATE.format(
        seed_data=seed_data,
        seed_description_block=seed_description_block,
        traj_description=traj_description,
        synthesis_tips_block=synthesis_tips_block,
        qa_examples_block=qa_examples_block,
        regen_block=regen_block,
    )


__all__ = ["QA_SYNTHESIS_BASE_TEMPLATE", "build_qa_synthesis_prompt"]
