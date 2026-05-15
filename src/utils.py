import re
from typing import Optional


def extract_answer(text: str) -> str:
    """
    Extract the best available final answer.

    Priority:
    1. The last complete <answer>...</answer> block.
    2. A trailing unclosed <answer> block.
    3. Any remaining plain text after removing tool/thinking blocks.
    """
    if not text:
        return ""

    last_answer_open = text.rfind("<answer>")
    last_answer_close = text.rfind("</answer>")
    if last_answer_open != -1 and last_answer_open > last_answer_close:
        trailing_answer = text[last_answer_open + len("<answer>"):].strip()
        if trailing_answer:
            return trailing_answer

    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    cleaned = text
    for tag in ("think", "python", "result"):
        cleaned = re.sub(rf"<{tag}>.*?</{tag}>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"</?(think|python|result|answer)>", "", cleaned)
    return cleaned.strip()


def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}."""
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def count_valid_tags(text: str, tag: str) -> int:
    """Count valid paired <tag>...</tag> occurrences."""
    count = 0
    current_pos = 0
    while True:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start_pos = text.find(start_tag, current_pos)
        if start_pos == -1:
            break
        end_pos = text.find(end_tag, start_pos + len(start_tag))
        if end_pos == -1:
            break
        count += 1
        current_pos = end_pos + len(end_tag)
    return count


def remove_result_tags(text: str) -> str:
    """Remove content inside <result>...</result> and <r>...</r> tags."""
    if not text:
        return ""
    cleaned = re.sub(r"<r>.*?</r>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<result>.*?</result>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()
