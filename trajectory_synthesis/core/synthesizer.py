"""
QA synthesizer for scientific QA synthesis.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import SynthesisConfig
from .models import SynthesizedQA, Trajectory
from .utils import chat_completion, create_openai_client, extract_json_object
from ..prompt.qa_synthesis import build_qa_synthesis_prompt


class QASynthesizer:
    """Synthesize QA pairs from selected trajectories."""

    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.client = create_openai_client(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def synthesize_qa(
        self, trajectory: Trajectory, qa_index: int = 0
    ) -> Optional[SynthesizedQA]:
        print(f"\nSynthesizing QA pair - Trajectory: {trajectory.trajectory_id}")

        traj_description = self._format_trajectory(trajectory)
        max_attempts = 3
        last_failure_reason = ""

        for attempt in range(1, max_attempts + 1):
            prompt = self._build_prompt(
                trajectory, traj_description, last_failure_reason
            )

            try:
                response = chat_completion(
                    self.client,
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 + 0.1 * (attempt - 1),
                    response_format={"type": "json_object"},
                )
                result = json.loads(
                    extract_json_object(response.choices[0].message.content)
                )
            except Exception as exc:
                last_failure_reason = f"Synthesis exception: {exc}"
                continue

            qa_obj, failure_reason = self._build_qa_from_result(
                result, trajectory, qa_index, attempt
            )
            if failure_reason or qa_obj is None:
                last_failure_reason = failure_reason or "Unknown validation failure."
                continue

            print("  Successfully synthesized QA pair")
            print(f"    QA ID: {qa_obj.qa_id}")
            print(f"    Question: {qa_obj.question}...")
            print(f"    Answer: {qa_obj.answer}...")
            return qa_obj

        print(
            f"  Synthesis failed after {max_attempts} attempts. "
            f"Last reason: {last_failure_reason}"
        )
        return None

    def _build_qa_from_result(
        self,
        result: Dict[str, Any],
        trajectory: Trajectory,
        qa_index: int,
        attempt: int,
    ) -> Tuple[Optional[SynthesizedQA], Optional[str]]:
        del attempt
        if not isinstance(result, dict):
            return None, "Invalid response format."

        question = str(result.get("question", "")).strip()
        answer = str(result.get("answer", "")).strip()
        reasoning_steps = self._normalize_reasoning_steps(
            result.get("reasoning_steps", [])
        )

        if not question or not answer:
            return None, "Empty question or answer."
        if self._answer_leaks_into_question(question, answer):
            return None, "Answer leakage: answer appears in question."
        if self._question_too_verbose(question):
            return None, "Question too verbose."
        if len(reasoning_steps) < 2:
            return None, "Missing/too-short reasoning_steps. Provide >=2 ReAct steps."
        react_err = self._validate_reasoning_steps_structure(reasoning_steps)
        if react_err:
            return None, react_err
        final_step_answer = self._extract_tag_text(reasoning_steps[-1], "answer")
        if final_step_answer != answer:
            return None, 'Top-level "answer" must match final reasoning_steps answer.'

        qa_id = f"{trajectory.trajectory_id}_qa_{qa_index}"
        qa = SynthesizedQA(
            question=question,
            answer=answer,
            trajectory_id=trajectory.trajectory_id,
            source_id=trajectory.source_id,
            qa_id=qa_id,
            reasoning_steps=reasoning_steps,
            metadata={
                "seed_data": trajectory.seed_data,
                "seed_description": self.config.seed_description,
                "trajectory_depth": trajectory.total_depth,
                "synthesis_date": datetime.now().isoformat(),
            },
        )
        return qa, None

    def _normalize_text(self, text: str) -> str:
        normalized = str(text or "").strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _answer_leaks_into_question(self, question: str, answer: str) -> bool:
        question_text = self._normalize_text(question)
        answer_text = self._normalize_text(answer)
        if not question_text or not answer_text:
            return False
        return answer_text in question_text

    def _question_too_verbose(
        self, question: str, max_words: int = 85, max_chars: int = 500
    ) -> bool:
        question_text = str(question or "").strip()
        if len(question_text) > max_chars:
            return True
        words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", question_text)
        return len(words) > max_words

    def _build_prompt(
        self,
        trajectory: Trajectory,
        traj_description: str,
        last_failure_reason: str,
    ) -> str:
        return build_qa_synthesis_prompt(
            seed_data=trajectory.seed_data,
            seed_description=self.config.seed_description,
            traj_description=traj_description,
            synthesis_tips=self.config.synthesis_tips,
            qa_examples=self.config.qa_examples,
            last_failure_reason=last_failure_reason,
        )

    def _format_trajectory(self, trajectory: Trajectory) -> str:
        """Join per-node tagged logs (``<think>``, ``<python>``, ``<result>``, etc.)."""
        parts: List[str] = []
        for node in trajectory.nodes:
            parts.extend(node.build_tagged_logs())
        return "\n\n".join(parts)

    def _normalize_reasoning_steps(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        normalized: List[str] = []
        for step in value:
            if isinstance(step, str):
                step_text = step.strip()
                if step_text:
                    normalized.append(step_text)
                continue
            if isinstance(step, dict):
                serialized = self._serialize_reasoning_step_dict(step)
                if serialized:
                    normalized.append(serialized)
        return normalized

    def _validate_reasoning_steps_structure(
        self, reasoning_steps: List[str]
    ) -> Optional[str]:
        if len(reasoning_steps) < 2:
            return (
                "reasoning_steps must include at least one <think><python><result> "
                "step and one final <think><answer> step."
            )

        for idx, step in enumerate(reasoning_steps[:-1], start=1):
            has_think = bool(self._extract_tag_text(step, "think"))
            has_python = bool(self._extract_tag_text(step, "python"))
            has_result = bool(self._extract_tag_text(step, "result"))
            has_answer = bool(self._extract_tag_text(step, "answer"))
            if not (has_think and has_python and has_result) or has_answer:
                return (
                    "Invalid reasoning_steps structure: each intermediate step "
                    f"must contain think/python/result only (step {idx})."
                )

        final_step = reasoning_steps[-1]
        has_think = bool(self._extract_tag_text(final_step, "think"))
        has_answer = bool(self._extract_tag_text(final_step, "answer"))
        has_python = bool(self._extract_tag_text(final_step, "python"))
        has_result = bool(self._extract_tag_text(final_step, "result"))
        if not (has_think and has_answer) or has_python or has_result:
            return (
                "Invalid reasoning_steps structure: final step must contain "
                "exactly think/answer."
            )
        return None

    @staticmethod
    def _extract_tag_text(step: str, tag: str) -> str:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        m = re.search(pattern, str(step or ""), flags=re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def _serialize_reasoning_step_dict(self, step: Dict[str, Any]) -> str:
        think = str(step.get("think", "")).strip()
        python = str(step.get("python", "")).strip()
        result = str(step.get("result", "")).strip()
        answer = str(step.get("answer", "")).strip()

        parts: List[str] = []
        if think:
            parts.append(f"<think>\n{think}\n</think>")
        if python:
            parts.append(f"<python>\n{python}\n</python>")
        if result:
            parts.append(f"<result>\n{result}\n</result>")
        if answer:
            parts.append(f"<answer>\n{answer}\n</answer>")
        return " ".join(parts).strip()
