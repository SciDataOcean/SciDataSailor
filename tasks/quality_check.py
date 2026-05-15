from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from src.utils import extract_answer, extract_boxed
except Exception:
    def extract_answer(text: str) -> str:
        if not text:
            return ""
        pattern = r"<answer>(.*?)</answer>"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return text.strip()

    def extract_boxed(text: str) -> Optional[str]:
        matches = re.findall(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)
        return matches[-1].strip() if matches else None


TAG_PATTERN = re.compile(r"<(/?)([A-Za-z_][A-Za-z0-9_-]*)>")
DEFAULT_ID_KEYS = ("sample_id", "task_id", "id", "uid", "question_id")
DEFAULT_TRAJECTORY_KEYS = ("output", "trajectory", "response", "logs")
DEFAULT_PREDICTION_KEYS = (
    "prediction",
    "pred",
    "model_answer",
    "final_answer",
)
DEFAULT_REFERENCE_KEYS = (
    "answer",
    "reference",
    "reference_answer",
    "gold",
    "gold_answer",
    "target",
    "label",
    "expected_answer",
    "answer_key",
)
SEARCH_LIKE_TAGS = {
    "search",
    "browser",
    "webbrowser",
    "webbrowseragent",
    "google",
    "bing",
    "retrieve",
}
CORE_TAGS = {"think", "result", "answer"}
KNOWN_TOOL_TAGS = {
    "python",
    "search",
    "browser",
    "webbrowser",
    "webbrowseragent",
    "google",
    "bing",
    "retrieve",
    "calculator",
    "code",
    "bash",
    "sql",
    "codedebugger",
    "backtracer",
    "refiner",
}
EXECUTION_TOOL_TAGS = {"python", "bash", "sql", "code", "calculator"}
TAG_ALIASES = {
    "thought": "think",
    "reasoning": "think",
    "reflection": "think",
    "tool_result": "result",
    "feedback": "result",
    "observation": "result",
    "obs": "result",
    "r": "result",
    "final": "answer",
    "finalanswer": "answer",
    "final_answer": "answer",
}
SPECIAL_TAGS = CORE_TAGS | KNOWN_TOOL_TAGS | set(TAG_ALIASES.keys()) | set(TAG_ALIASES.values())
SPECIAL_TAG_PATTERN = re.compile(
    r"<(/?)(" + "|".join(sorted((re.escape(tag) for tag in SPECIAL_TAGS), key=len, reverse=True)) + r")>",
    flags=re.IGNORECASE,
)
RESULT_ERROR_PATTERNS = (
    "traceback",
    "exception",
    "error",
    "failed",
    "not found",
    "timeout",
    "syntaxerror",
    "nameerror",
    "typeerror",
    "valueerror",
    "maximum python call limit is exceeded",
)
ANSWER_PREFIX_RE = re.compile(
    r"^(the\s+)?(final\s+)?answer\s*(is|:)\s*",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check trajectory quality and split TIR data into SFT/RL sets."
    )
    parser.add_argument("--tir_file", type=str, required=True, help="Tool-integrated results JSON/JSONL.")
    parser.add_argument("--dr_file", type=str, default=None, help="Direct-reasoning results JSON/JSONL.")
    parser.add_argument(
        "--reference_file",
        type=str,
        default=None,
        help="Optional reference answers JSON/JSONL. If omitted, references are searched inside tir/dr records.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="quality_output",
        help="Directory for reports and split datasets.",
    )
    parser.add_argument(
        "--tool_frequency_threshold",
        type=int,
        default=None,
        help="Maximum allowed number of tool calls per trajectory. If omitted, use an auto threshold tuned for SciDataCrawler.",
    )
    parser.add_argument(
        "--trajectory_keys",
        nargs="+",
        default=list(DEFAULT_TRAJECTORY_KEYS),
        help="Keys checked in order to locate the trajectory text.",
    )
    parser.add_argument(
        "--prediction_keys",
        nargs="+",
        default=list(DEFAULT_PREDICTION_KEYS),
        help="Keys checked in order to locate model predictions.",
    )
    parser.add_argument(
        "--reference_keys",
        nargs="+",
        default=list(DEFAULT_REFERENCE_KEYS),
        help="Keys checked in order to locate gold answers.",
    )
    parser.add_argument(
        "--id_keys",
        nargs="+",
        default=list(DEFAULT_ID_KEYS),
        help="Keys checked in order to align TIR / DR / reference records.",
    )
    parser.add_argument(
        "--save_normalized_trajectory",
        action="store_true",
        help="Save normalized trajectory text into filtered/SFT/RL outputs.",
    )

    scoring_group = parser.add_argument_group("Composite Scoring")
    scoring_group.add_argument(
        "--sft_threshold",
        type=float,
        default=80.0,
        help="Minimum composite score to qualify as SFT data (default 80).",
    )
    scoring_group.add_argument(
        "--rl_threshold",
        type=float,
        default=40.0,
        help="Minimum composite score to qualify as RL data; below this is discarded (default 40).",
    )
    return parser.parse_args()


def load_records(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "results", "data", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unsupported file format for {path}")


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON array from *path*, returning ``[]`` on missing / corrupt file."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _incremental_merge(
    existing: List[Dict[str, Any]],
    new: List[Dict[str, Any]],
    processed_ids: set,
    id_key: str,
) -> List[Dict[str, Any]]:
    """Keep existing records whose *id_key* was NOT re-processed, then append *new*."""
    kept = [r for r in existing if r.get(id_key) not in processed_ids]
    return kept + new


def first_nonempty(record: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value:
            return value
        if value not in ("", [], {}):
            return value
    return None


def normalize_free_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = html.unescape(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sample_id(record: Dict[str, Any], idx: int, id_keys: Sequence[str]) -> str:
    value = first_nonempty(record, id_keys)
    if value is not None:
        return str(value)

    query = first_nonempty(record, ("input", "question", "query", "task", "prompt"))
    if isinstance(query, str) and query.strip():
        return hashlib.md5(normalize_free_text(query).encode("utf-8")).hexdigest()

    return f"sample_{idx}"


def build_index(records: Sequence[Dict[str, Any]], id_keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for idx, record in enumerate(records):
        sid = sample_id(record, idx, id_keys)
        if sid not in index:
            index[sid] = record
    return index


def extract_trajectory(record: Dict[str, Any], trajectory_keys: Sequence[str]) -> str:
    value = first_nonempty(record, trajectory_keys)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None)
    return str(value or "")


def canonical_tag(tag_name: str) -> str:
    lowered = tag_name.lower()
    return TAG_ALIASES.get(lowered, lowered)


def normalize_special_tokens(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        slash = match.group(1)
        tag_name = canonical_tag(match.group(2))
        return f"<{slash}{tag_name}>"

    return TAG_PATTERN.sub(repl, text or "")


def parse_top_level_blocks(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    blocks: List[Dict[str, Any]] = []
    errors: List[str] = []
    stack: List[Tuple[str, int, int, int]] = []
    last_consumed = 0

    for match in SPECIAL_TAG_PATTERN.finditer(text):
        is_closing = bool(match.group(1))
        tag_name = canonical_tag(match.group(2))

        # Tolerant mode for noisy final answers:
        if stack and stack[-1][0] == "answer":
            if is_closing and tag_name == "answer":
                open_tag, open_start, open_end, top_start = stack.pop()
                if not stack:
                    blocks.append(
                        {
                            "tag": open_tag,
                            "content": text[open_end:match.start()],
                            "start": top_start,
                            "end": match.end(),
                        }
                    )
                    last_consumed = match.end()
            # Everything else inside <answer> is treated as plain text.
            continue

        if not is_closing:
            if not stack:
                outside = text[last_consumed:match.start()]
                if outside.strip():
                    errors.append("non-whitespace text exists outside tagged blocks")
            top_start = match.start() if not stack else stack[-1][3]
            stack.append((tag_name, match.start(), match.end(), top_start))
            continue

        if not stack:
            errors.append(f"unexpected closing tag </{tag_name}>")
            last_consumed = match.end()
            continue

        open_tag, open_start, open_end, top_start = stack.pop()
        if tag_name != open_tag:
            errors.append(f"mismatched closing tag </{tag_name}> for <{open_tag}>")

        if not stack:
            blocks.append(
                {
                    "tag": open_tag,
                    "content": text[open_end:match.start()],
                    "start": top_start,
                    "end": match.end(),
                }
            )
            last_consumed = match.end()

    if stack:
        errors.append("unclosed tags: " + ", ".join(f"<{tag}>" for tag, *_ in stack))

    trailing = text[last_consumed:]
    if trailing.strip():
        errors.append("non-whitespace trailing text exists after the last tagged block")

    return blocks, errors


def analyze_format(normalized_trajectory: str) -> Dict[str, Any]:
    blocks, parse_errors = parse_top_level_blocks(normalized_trajectory)
    errors = list(parse_errors)

    think_count = 0
    result_count = 0
    answer_count = 0
    tool_blocks: List[Dict[str, Any]] = []
    last_tag: Optional[str] = None
    saw_answer = False
    saw_think = False

    for idx, block in enumerate(blocks):
        tag = block["tag"]
        if tag == "think":
            think_count += 1
            saw_think = True
            if saw_answer:
                errors.append("think block appears after answer")
        elif tag == "result":
            result_count += 1
            if not saw_think:
                errors.append("result block appears before any think block")
            if saw_answer:
                errors.append("result block appears after answer")
        elif tag == "answer":
            answer_count += 1
            if idx != len(blocks) - 1:
                errors.append("answer block must be the final top-level block")
            saw_answer = True
        else:
            tool_blocks.append(block)
            if not saw_think:
                errors.append(f"tool block <{tag}> appears before any think block")
            if saw_answer:
                errors.append(f"tool block <{tag}> appears after answer")
        last_tag = tag

    if not blocks:
        errors.append("no valid tagged blocks found")
    if answer_count != 1:
        errors.append("trajectory must contain exactly one <answer>...</answer> block")
    if think_count == 0:
        errors.append("trajectory must contain at least one <think>...</think> block")
    if last_tag != "answer":
        errors.append("trajectory must end with <answer>...</answer>")
    if tool_blocks and result_count != len(tool_blocks):
        errors.append("number of <result> blocks must match number of tool-call blocks")
    if not tool_blocks and result_count > think_count:
        errors.append("result blocks exceed reasoning blocks")

    return {
        "normalized_trajectory": normalized_trajectory.strip(),
        "blocks": blocks,
        "errors": list(dict.fromkeys(errors)),
        "format_valid": len(errors) == 0,
        "think_count": think_count,
        "tool_call_count": len(tool_blocks),
        "result_count": result_count,
        "answer_count": answer_count,
        "tool_tags": [block["tag"] for block in tool_blocks],
    }


def canonicalize_tool_call(tag: str, content: str) -> str:
    payload = unicodedata.normalize("NFKC", content or "")
    payload = html.unescape(payload).strip()
    if tag in SEARCH_LIKE_TAGS:
        payload = re.sub(r"\s+", " ", payload).casefold()
    else:
        lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in payload.splitlines() if line.strip()]
        payload = "\n".join(lines)
    return f"{tag}:{payload}"


def result_contains_error(text: str) -> bool:
    lowered = unicodedata.normalize("NFKC", text or "").casefold()
    return any(pattern in lowered for pattern in RESULT_ERROR_PATTERNS)


COMPOSITE_WEIGHTS = {
    "format": 0.20,
    "execution": 0.35,
    "completion": 0.25,
    "efficiency": 0.10,
    "coherence": 0.10,
}


def _compute_format_score(format_info: Dict[str, Any]) -> float:
    if format_info["format_valid"]:
        return 100.0
    return max(0.0, 100.0 - 20.0 * len(format_info["errors"]))


def _compute_execution_score(
    result_blocks: List[Dict[str, Any]], result_error_count: int
) -> float:
    if not result_blocks:
        return 50.0
    success = len(result_blocks) - result_error_count
    return (success / len(result_blocks)) * 100.0


def _compute_completion_score(answer_nonempty: bool) -> float:
    return 100.0 if answer_nonempty else 0.0


def _compute_efficiency_score(
    tool_call_count: int,
    duplicate_tool_call_count: int,
    tool_frequency_threshold: int,
) -> float:
    score = 100.0
    if tool_call_count > tool_frequency_threshold:
        score -= min(50.0, 10.0 * (tool_call_count - tool_frequency_threshold))
    if duplicate_tool_call_count > 0:
        score -= min(40.0, 15.0 * duplicate_tool_call_count)
    return max(0.0, score)


def _compute_coherence_score(think_blocks: List[Dict[str, Any]]) -> float:
    if not think_blocks:
        return 0.0
    non_empty = sum(1 for b in think_blocks if b["content"].strip())
    return (non_empty / len(think_blocks)) * 100.0


def quality_assessment(format_info: Dict[str, Any], tool_frequency_threshold: int) -> Dict[str, Any]:
    blocks = format_info["blocks"]
    tool_blocks = [block for block in blocks if block["tag"] not in CORE_TAGS]
    result_blocks = [block for block in blocks if block["tag"] == "result"]
    think_blocks = [block for block in blocks if block["tag"] == "think"]
    answer_blocks = [block for block in blocks if block["tag"] == "answer"]

    tool_signatures = [canonicalize_tool_call(block["tag"], block["content"]) for block in tool_blocks]
    tool_counter = Counter(tool_signatures)
    duplicate_tool_call_count = sum(count - 1 for count in tool_counter.values() if count > 1)
    duplicate_tool_calls = [
        {"call": signature, "count": count}
        for signature, count in tool_counter.items()
        if count > 1
    ]

    empty_think_count = sum(1 for block in think_blocks if not block["content"].strip())
    empty_result_count = sum(1 for block in result_blocks if not block["content"].strip())
    result_error_count = sum(1 for block in result_blocks if result_contains_error(block["content"]))
    answer_text = answer_blocks[-1]["content"].strip() if answer_blocks else ""
    answer_nonempty = bool(answer_text)
    tool_call_count = len(tool_blocks)

    format_score = _compute_format_score(format_info)
    execution_score = _compute_execution_score(result_blocks, result_error_count)
    completion_score = _compute_completion_score(answer_nonempty)
    efficiency_score = _compute_efficiency_score(
        tool_call_count, duplicate_tool_call_count, tool_frequency_threshold,
    )
    coherence_score = _compute_coherence_score(think_blocks)

    composite_score = (
        COMPOSITE_WEIGHTS["format"] * format_score
        + COMPOSITE_WEIGHTS["execution"] * execution_score
        + COMPOSITE_WEIGHTS["completion"] * completion_score
        + COMPOSITE_WEIGHTS["efficiency"] * efficiency_score
        + COMPOSITE_WEIGHTS["coherence"] * coherence_score
    )
    composite_score = round(max(0.0, min(100.0, composite_score)), 2)

    warnings: List[str] = []
    if not format_info["format_valid"]:
        warnings.append("format_invalid")
    if not answer_nonempty:
        warnings.append("empty_answer")
    if tool_call_count > tool_frequency_threshold:
        warnings.append("tool_frequency_exceeded")
    if duplicate_tool_call_count > 0:
        warnings.append("duplicate_tool_calls")

    return {
        "composite_score": composite_score,
        "format_score": round(format_score, 2),
        "execution_score": round(execution_score, 2),
        "completion_score": round(completion_score, 2),
        "efficiency_score": round(efficiency_score, 2),
        "coherence_score": round(coherence_score, 2),
        "reward": round(composite_score / 100.0, 4),
        "warnings": warnings,
        "answer_nonempty": answer_nonempty,
        "tool_call_count": tool_call_count,
        "duplicate_tool_call_count": duplicate_tool_call_count,
        "duplicate_tool_calls": duplicate_tool_calls,
        "result_error_count": result_error_count,
        "empty_think_count": empty_think_count,
        "empty_result_count": empty_result_count,
    }


def strip_tag_markup(text: str) -> str:
    return TAG_PATTERN.sub("", text or "")


def normalize_answer_text(text: str) -> str:
    if not text:
        return ""

    text = extract_answer(text) if "<answer>" in text else text
    boxed = extract_boxed(text)
    if boxed:
        text = boxed

    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = strip_tag_markup(text)
    text = text.replace("$", " ")
    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text, flags=re.DOTALL)
    text = ANSWER_PREFIX_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text)
    text = text.strip().strip(".,;:!?\"'`")
    return text.casefold()


def try_parse_number(text: str) -> Optional[float]:
    cleaned = text.replace(",", "").strip()
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", cleaned):
        return float(cleaned)
    if re.fullmatch(r"[-+]?\d+/\d+", cleaned):
        numerator, denominator = cleaned.split("/")
        denominator_value = float(denominator)
        if denominator_value != 0:
            return float(numerator) / denominator_value
    return None


def answers_match(prediction: Optional[str], reference: Optional[str]) -> Optional[bool]:
    if not prediction or not reference:
        return None

    pred_norm = normalize_answer_text(prediction)
    ref_norm = normalize_answer_text(reference)
    if not pred_norm or not ref_norm:
        return None

    pred_num = try_parse_number(pred_norm)
    ref_num = try_parse_number(ref_norm)
    if pred_num is not None and ref_num is not None:
        return abs(pred_num - ref_num) <= 1e-8 * max(1.0, abs(ref_num))

    if pred_norm == ref_norm:
        return True

    pred_simple = re.sub(r"[^\w\s./-]", "", pred_norm)
    ref_simple = re.sub(r"[^\w\s./-]", "", ref_norm)
    return pred_simple == ref_simple


def extract_prediction(record: Dict[str, Any], trajectory: str, prediction_keys: Sequence[str]) -> str:
    value = first_nonempty(record, prediction_keys)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return extract_answer(trajectory)


def extract_reference(
    sample_id_value: str,
    primary_record: Dict[str, Any],
    reference_index: Optional[Dict[str, Dict[str, Any]]],
    reference_keys: Sequence[str],
) -> Optional[str]:
    record_value = first_nonempty(primary_record, reference_keys)
    if isinstance(record_value, str) and record_value.strip():
        return record_value.strip()

    if reference_index and sample_id_value in reference_index:
        reference_record = reference_index[sample_id_value]
        indexed_value = first_nonempty(reference_record, reference_keys)
        if isinstance(indexed_value, str) and indexed_value.strip():
            return indexed_value.strip()

    return None


def compact_errors(errors: Iterable[str]) -> List[str]:
    seen = set()
    compact: List[str] = []
    for error in errors:
        if error not in seen:
            compact.append(error)
            seen.add(error)
    return compact


def classify_sample(dr_correct: Optional[bool], tir_correct: Optional[bool]) -> Optional[str]:
    if dr_correct is None or tir_correct is None:
        return None
    if dr_correct and tir_correct:
        return "category_1_dr_correct_tir_correct"
    if dr_correct and not tir_correct:
        return "category_2_dr_correct_tir_wrong"
    if not dr_correct and tir_correct:
        return "category_3_dr_wrong_tir_correct"
    return "category_4_dr_wrong_tir_wrong"


def base_prompt(record: Dict[str, Any]) -> str:
    prompt = first_nonempty(record, ("input", "question", "query", "task", "prompt"))
    return str(prompt or "")


def summarize_counts(values: Iterable[Optional[str]]) -> Dict[str, int]:
    counter = Counter(value for value in values if value is not None)
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def summarize_reason_counts(samples: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in samples:
        for reason in item.get("hard_fail_reasons", []):
            counter[reason] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def infer_tool_frequency_threshold(precomputed_samples: Sequence[Dict[str, Any]]) -> Tuple[int, str]:
    tool_counts = [item["format_info"]["tool_call_count"] for item in precomputed_samples]
    observed_tags = {
        tag
        for item in precomputed_samples
        for tag in item["format_info"]["tool_tags"]
    }

    only_execution_tools = bool(observed_tags) and observed_tags.issubset(EXECUTION_TOOL_TAGS)
    base_threshold = 10 if only_execution_tools else 7

    nonzero_counts = [count for count in tool_counts if count > 0]
    if not nonzero_counts:
        return base_threshold, "auto"

    if len(nonzero_counts) <= 3:
        anchor = max(nonzero_counts)
    else:
        sorted_counts = sorted(nonzero_counts)
        anchor = sorted_counts[int(round(0.8 * (len(sorted_counts) - 1)))]

    threshold = max(base_threshold, min(15, anchor))
    return threshold, "auto"


def build_sft_record(
    sample_id_value: str,
    tir_record: Dict[str, Any],
    quality_info: Dict[str, Any],
    save_normalized_trajectory: bool,
    reference: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    output_record = {
        "sample_id": sample_id_value,
        "input": base_prompt(tir_record),
        "output": tir_record.get("output", ""),
        "prediction": tir_record.get("prediction", ""),
        "composite_score": quality_info["composite_score"],
        "split": "sft",
    }
    if reference is not None:
        output_record["reference"] = reference
    if category is not None:
        output_record["difficulty_category"] = category
    if save_normalized_trajectory:
        output_record["normalized_output"] = tir_record.get("normalized_output", "")
    return output_record


def build_rl_record(
    sample_id_value: str,
    tir_record: Dict[str, Any],
    quality_info: Dict[str, Any],
    save_normalized_trajectory: bool,
    reference: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    output_record = {
        "sample_id": sample_id_value,
        "input": base_prompt(tir_record),
        "output": tir_record.get("output", ""),
        "prediction": tir_record.get("prediction", ""),
        "composite_score": quality_info["composite_score"],
        "reward": quality_info["reward"],
        "quality_detail": {
            "format_score": quality_info["format_score"],
            "execution_score": quality_info["execution_score"],
            "completion_score": quality_info["completion_score"],
            "efficiency_score": quality_info["efficiency_score"],
            "coherence_score": quality_info["coherence_score"],
        },
        "split": "rl",
    }
    if reference is not None:
        output_record["reference"] = reference
    if category is not None:
        output_record["difficulty_category"] = category
    if save_normalized_trajectory:
        output_record["normalized_output"] = tir_record.get("normalized_output", "")
    return output_record


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tir_records = load_records(args.tir_file)
    dr_records = load_records(args.dr_file) if args.dr_file else []
    reference_records = load_records(args.reference_file) if args.reference_file else []

    dr_index = build_index(dr_records, args.id_keys) if dr_records else {}
    reference_index = build_index(reference_records, args.id_keys) if reference_records else {}

    precomputed_samples: List[Dict[str, Any]] = []
    for idx, raw_tir_record in enumerate(tir_records):
        sid = sample_id(raw_tir_record, idx, args.id_keys)
        trajectory = extract_trajectory(raw_tir_record, args.trajectory_keys)
        normalized_trajectory = normalize_special_tokens(trajectory)
        format_info = analyze_format(normalized_trajectory)
        precomputed_samples.append(
            {
                "sample_id": sid,
                "raw_tir_record": raw_tir_record,
                "normalized_trajectory": normalized_trajectory,
                "format_info": format_info,
            }
        )

    if args.tool_frequency_threshold is None:
        effective_tool_frequency_threshold, threshold_mode = infer_tool_frequency_threshold(precomputed_samples)
    else:
        effective_tool_frequency_threshold, threshold_mode = args.tool_frequency_threshold, "manual"

    sample_reports: List[Dict[str, Any]] = []
    filtered_tir: List[Dict[str, Any]] = []
    sft_data: List[Dict[str, Any]] = []
    rl_data: List[Dict[str, Any]] = []
    num_samples_with_reference = 0
    num_samples_with_dr = 0
    num_samples_classified = 0
    num_discarded = 0
    score_accumulator: List[float] = []

    for precomputed in precomputed_samples:
        sid = precomputed["sample_id"]
        raw_tir_record = precomputed["raw_tir_record"]
        normalized_trajectory = precomputed["normalized_trajectory"]
        format_info = precomputed["format_info"]

        quality_info = quality_assessment(
            format_info, effective_tool_frequency_threshold,
        )

        tir_prediction = extract_prediction(raw_tir_record, normalized_trajectory, args.prediction_keys)
        reference = extract_reference(sid, raw_tir_record, reference_index, args.reference_keys)
        tir_correct = answers_match(tir_prediction, reference)

        enriched_tir_record = dict(raw_tir_record)
        enriched_tir_record["sample_id"] = sid
        enriched_tir_record["prediction"] = tir_prediction
        enriched_tir_record["quality_check"] = {
            "format_valid": format_info["format_valid"],
            "format_errors": compact_errors(format_info["errors"]),
            "think_count": format_info["think_count"],
            "tool_call_count": format_info["tool_call_count"],
            "result_count": format_info["result_count"],
            "answer_count": format_info["answer_count"],
            "tool_tags": format_info["tool_tags"],
            **quality_info,
        }
        if args.save_normalized_trajectory:
            enriched_tir_record["normalized_output"] = format_info["normalized_trajectory"]

        dr_record = dr_index.get(sid)
        dr_prediction = None
        dr_correct = None
        category = None

        if dr_record is not None:
            num_samples_with_dr += 1
            dr_output = extract_trajectory(dr_record, args.trajectory_keys)
            dr_prediction = extract_prediction(dr_record, dr_output, args.prediction_keys)
            reference = reference or extract_reference(sid, dr_record, reference_index, args.reference_keys)
            dr_correct = answers_match(dr_prediction, reference)
            category = classify_sample(dr_correct, tir_correct)
            if category is not None:
                num_samples_classified += 1

        if reference:
            num_samples_with_reference += 1

        composite = quality_info["composite_score"]
        score_accumulator.append(composite)

        if composite >= args.sft_threshold:
            split = "sft"
        elif composite >= args.rl_threshold:
            split = "rl"
        else:
            split = "discard"

        sample_report = {
            "sample_id": sid,
            "input_preview": normalize_free_text(base_prompt(raw_tir_record))[:240],
            "format_valid": format_info["format_valid"],
            "format_errors": compact_errors(format_info["errors"]),
            "composite_score": composite,
            "format_score": quality_info["format_score"],
            "execution_score": quality_info["execution_score"],
            "completion_score": quality_info["completion_score"],
            "efficiency_score": quality_info["efficiency_score"],
            "coherence_score": quality_info["coherence_score"],
            "reward": quality_info["reward"],
            "split": split,
            "warnings": quality_info["warnings"],
            "tool_call_count": quality_info["tool_call_count"],
            "duplicate_tool_call_count": quality_info["duplicate_tool_call_count"],
            "result_error_count": quality_info["result_error_count"],
            "tir_correct": tir_correct,
            "dr_correct": dr_correct,
            "difficulty_category": category,
            "has_reference": bool(reference),
            "has_dr": dr_record is not None,
        }
        sample_reports.append(sample_report)

        if split == "sft":
            filtered_tir.append(enriched_tir_record)
            sft_data.append(
                build_sft_record(
                    sample_id_value=sid,
                    tir_record=enriched_tir_record,
                    quality_info=quality_info,
                    save_normalized_trajectory=args.save_normalized_trajectory,
                    reference=reference,
                    category=category,
                )
            )
        elif split == "rl":
            filtered_tir.append(enriched_tir_record)
            rl_data.append(
                build_rl_record(
                    sample_id_value=sid,
                    tir_record=enriched_tir_record,
                    quality_info=quality_info,
                    save_normalized_trajectory=args.save_normalized_trajectory,
                    reference=reference,
                    category=category,
                )
            )
        else:
            num_discarded += 1

    avg_score = sum(score_accumulator) / len(score_accumulator) if score_accumulator else 0.0
    summary = {
        "num_tir_records": len(tir_records),
        "num_filtered_tir_records": len(filtered_tir),
        "num_sft_records": len(sft_data),
        "num_rl_records": len(rl_data),
        "num_discarded": num_discarded,
        "sft_threshold": args.sft_threshold,
        "rl_threshold": args.rl_threshold,
        "composite_score_mean": round(avg_score, 2),
        "split_distribution": summarize_counts(item["split"] for item in sample_reports),
        "num_format_valid": sum(1 for item in sample_reports if item["format_valid"]),
        "warning_distribution": summarize_reason_counts(
            [{"hard_fail_reasons": item["warnings"]} for item in sample_reports]
        ),
        "difficulty_distribution": summarize_counts(item["difficulty_category"] for item in sample_reports),
        "num_samples_with_reference": num_samples_with_reference,
        "num_samples_with_dr": num_samples_with_dr,
        "num_samples_classified": num_samples_classified,
        "tool_frequency_threshold": effective_tool_frequency_threshold,
        "tool_frequency_threshold_mode": threshold_mode,
        "composite_weights": COMPOSITE_WEIGHTS,
        "has_dr_file": bool(args.dr_file),
        "has_reference_file": bool(args.reference_file),
    }

    # --- Incremental merge with existing outputs ---
    processed_ids = {s["sample_id"] for s in sample_reports}

    filtered_tir = _incremental_merge(
        _load_json_list(output_dir / "filtered_tir.json"),
        filtered_tir, processed_ids, "sample_id",
    )
    sft_data = _incremental_merge(
        _load_json_list(output_dir / "sft_data.json"),
        sft_data, processed_ids, "sample_id",
    )
    rl_data = _incremental_merge(
        _load_json_list(output_dir / "rl_data.json"),
        rl_data, processed_ids, "sample_id",
    )

    existing_report_samples: List[Dict[str, Any]] = []
    report_path = output_dir / "quality_report.json"
    if report_path.exists():
        try:
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            existing_report_samples = existing_report.get("samples", [])
        except (json.JSONDecodeError, OSError):
            pass
    merged_samples = _incremental_merge(
        existing_report_samples, sample_reports, processed_ids, "sample_id",
    )

    all_scores = [s["composite_score"] for s in merged_samples]
    summary["num_tir_records"] = len(merged_samples)
    summary["num_filtered_tir_records"] = len(filtered_tir)
    summary["num_sft_records"] = len(sft_data)
    summary["num_rl_records"] = len(rl_data)
    summary["num_discarded"] = len(merged_samples) - len(sft_data) - len(rl_data)
    summary["composite_score_mean"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0
    summary["split_distribution"] = summarize_counts(s["split"] for s in merged_samples)
    summary["num_format_valid"] = sum(1 for s in merged_samples if s["format_valid"])
    summary["warning_distribution"] = summarize_reason_counts(
        [{"hard_fail_reasons": s["warnings"]} for s in merged_samples]
    )
    summary["difficulty_distribution"] = summarize_counts(
        s["difficulty_category"] for s in merged_samples
    )
    summary["num_samples_with_reference"] = sum(1 for s in merged_samples if s.get("has_reference"))
    summary["num_samples_with_dr"] = sum(1 for s in merged_samples if s.get("has_dr"))
    summary["num_samples_classified"] = sum(
        1 for s in merged_samples if s.get("difficulty_category") is not None
    )

    report = {
        "config": {
            "tir_file": args.tir_file,
            "dr_file": args.dr_file,
            "reference_file": args.reference_file,
            "sft_threshold": args.sft_threshold,
            "rl_threshold": args.rl_threshold,
            "tool_frequency_threshold": effective_tool_frequency_threshold,
            "tool_frequency_threshold_mode": threshold_mode,
            "composite_weights": COMPOSITE_WEIGHTS,
            "trajectory_keys": args.trajectory_keys,
            "prediction_keys": args.prediction_keys,
            "reference_keys": args.reference_keys,
            "id_keys": args.id_keys,
        },
        "summary": summary,
        "samples": merged_samples,
    }

    dump_json(output_dir / "quality_report.json", report)
    dump_json(output_dir / "filtered_tir.json", filtered_tir)
    dump_json(output_dir / "sft_data.json", sft_data)
    dump_json(output_dir / "rl_data.json", rl_data)

    num_discarded = summary["num_discarded"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if num_discarded > 0:
        print(f"Note: {num_discarded} trajectory(s) discarded (composite score < {args.rl_threshold}).")
    if len(sft_data) == 0:
        print(f"Note: no SFT data produced. Try lowering --sft_threshold (current: {args.sft_threshold}).")
    if args.dr_file:
        print(f"Note: 4-Category enrichment available ({summary['num_samples_classified']} classified).")
    print(f"Saved report to {output_dir / 'quality_report.json'}")
    print(f"Saved filtered TIR data ({len(filtered_tir)} records) to {output_dir / 'filtered_tir.json'}")
    print(f"Saved SFT data ({len(sft_data)} records) to {output_dir / 'sft_data.json'}")
    print(f"Saved RL data ({len(rl_data)} records, reward=composite/100) to {output_dir / 'rl_data.json'}")


if __name__ == "__main__":
    main()
