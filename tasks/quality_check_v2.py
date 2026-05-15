from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import os as _os
import sys as _sys

_sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from tasks import quality_check as qc


V2_WEIGHTS = {
    "grounding": 0.30,
    "coverage": 0.25,
    "execution": 0.20,
    "answer_quality": 0.15,
    "efficiency": 0.10,
}

DEFAULT_SFT_COMPONENT_FLOORS = {
    "grounding": 60.0,
    "coverage": 55.0,
    "answer_quality": 55.0,
}

ROOT_LISTING_OPS = {"walk", "listdir", "glob", "scan_dir"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml"}
COUNT_LABELS = {
    "rows": "row_count",
    "row": "row_count",
    "nrows": "row_count",
    "row_count": "row_count",
    "records": "row_count",
    "record": "row_count",
    "files": "file_count",
    "file": "file_count",
    "file_count": "file_count",
    "directories": "dir_count",
    "directory": "dir_count",
    "dirs": "dir_count",
    "dir": "dir_count",
    "dir_count": "dir_count",
    "folders": "dir_count",
    "folder": "dir_count",
    "columns": "column_count",
    "column": "column_count",
    "fields": "column_count",
    "field": "column_count",
    "column_count": "column_count",
}
SECTION_KEYWORDS = {
    "inventory": ("file", "directory", "folder", "extension", "subdir", "subdirectory"),
    "schema": ("schema", "column", "field", "row", "record", "dtype"),
    "summary": ("summary", "metadata", "dataset", "structure"),
    "calibration": ("uncertain", "unable", "not inspected", "not available", "could not", "missing"),
}

PATH_RE = re.compile(r"(?:[A-Za-z]:)?(?:[/\\][^\s<>'\"`]+)+")
FILE_LIKE_RE = re.compile(r"\b[\w.\-]+\.[A-Za-z0-9]{1,8}\b")
EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}\b")
COUNT_RE = re.compile(
    r"\b(rows?|nrows?|records?|files?|directories|directory|dirs?|folders?|columns?|fields?)\b"
    r"[^\n:]{0,24}[:= ]\s*(\d+)",
    flags=re.IGNORECASE,
)
SHAPE_RE = re.compile(
    r"\bshape\b[^\n:]{0,24}[:= ]\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?",
    flags=re.IGNORECASE,
)
COLUMN_BRACKET_RE = re.compile(
    r"\b(?:columns?|schema|fields?)\b[^\n:]{0,12}[:= ]\s*(\[[^\]]{1,500}\])",
    flags=re.IGNORECASE,
)
QUOTED_TOKEN_RE = re.compile(r"[\"']([^\"']{1,200})[\"']")
UNCERTAINTY_RE = re.compile(
    r"\b(?:uncertain|unsure|unable|not inspected|not available|could not|may contain|likely|appears to)\b",
    flags=re.IGNORECASE,
)
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


def _parse_top_level_blocks_tolerant(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse top-level tagged blocks with uniform tolerant handling.

    Once parser enters any <tag> block, only the first matching </tag> closes
    it. Any nested/interleaved tag-like text while inside a block is treated as
    plain text.
    """
    blocks: List[Dict[str, Any]] = []
    errors: List[str] = []
    stack: List[Tuple[str, int, int, int]] = []
    last_consumed = 0

    for match in qc.SPECIAL_TAG_PATTERN.finditer(text):
        is_closing = bool(match.group(1))
        tag_name = qc.canonical_tag(match.group(2))

        # Uniform tolerance: once inside any tag, ignore all inner tags until
        # the same closing tag appears.
        if stack:
            current_open_tag = stack[-1][0]
            if is_closing and tag_name == current_open_tag:
                open_tag, _open_start, open_end, top_start = stack.pop()
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

        open_tag, _open_start, open_end, top_start = stack.pop()
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


def _analyze_format_tolerant(normalized_trajectory: str) -> Dict[str, Any]:
    blocks, parse_errors = _parse_top_level_blocks_tolerant(normalized_trajectory)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check trajectory quality with a deterministic v2 scorer."
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
        default="quality_output_v2",
        help="Directory for reports and split datasets.",
    )
    parser.add_argument(
        "--tool_frequency_threshold",
        type=int,
        default=None,
        help="Soft warning threshold for unusually high tool use. If omitted, infer automatically.",
    )
    parser.add_argument(
        "--trajectory_keys",
        nargs="+",
        default=list(qc.DEFAULT_TRAJECTORY_KEYS),
        help="Keys checked in order to locate the trajectory text.",
    )
    parser.add_argument(
        "--prediction_keys",
        nargs="+",
        default=list(qc.DEFAULT_PREDICTION_KEYS),
        help="Keys checked in order to locate model predictions.",
    )
    parser.add_argument(
        "--reference_keys",
        nargs="+",
        default=list(qc.DEFAULT_REFERENCE_KEYS),
        help="Keys checked in order to locate gold answers.",
    )
    parser.add_argument(
        "--id_keys",
        nargs="+",
        default=list(qc.DEFAULT_ID_KEYS),
        help="Keys checked in order to align TIR / DR / reference records.",
    )
    parser.add_argument(
        "--save_normalized_trajectory",
        action="store_true",
        help="Save normalized trajectory text into filtered/SFT/RL outputs.",
    )

    manifest_group = parser.add_argument_group("Manifest")
    manifest_group.add_argument(
        "--manifest_max_files",
        type=int,
        default=2000,
        help="Maximum number of files to scan while building the dataset manifest.",
    )
    manifest_group.add_argument(
        "--manifest_top_extensions",
        type=int,
        default=5,
        help="How many dominant extensions to keep in the manifest checklist.",
    )
    manifest_group.add_argument(
        "--manifest_top_subtrees",
        type=int,
        default=4,
        help="How many dominant top-level subtrees to keep in the checklist.",
    )
    manifest_group.add_argument(
        "--manifest_table_limit",
        type=int,
        default=3,
        help="How many representative tabular files to summarize for schema-aware coverage.",
    )

    scoring_group = parser.add_argument_group("Composite Scoring")
    scoring_group.add_argument(
        "--sft_threshold",
        type=float,
        default=80.0,
        help="Minimum v2 composite score to qualify as SFT data.",
    )
    scoring_group.add_argument(
        "--rl_threshold",
        type=float,
        default=40.0,
        help="Minimum v2 composite score to qualify as RL data; below this is discarded.",
    )
    scoring_group.add_argument(
        "--sft_min_grounding",
        type=float,
        default=DEFAULT_SFT_COMPONENT_FLOORS["grounding"],
        help="Minimum grounding score for SFT promotion.",
    )
    scoring_group.add_argument(
        "--sft_min_coverage",
        type=float,
        default=DEFAULT_SFT_COMPONENT_FLOORS["coverage"],
        help="Minimum coverage score for SFT promotion.",
    )
    scoring_group.add_argument(
        "--sft_min_answer_quality",
        type=float,
        default=DEFAULT_SFT_COMPONENT_FLOORS["answer_quality"],
        help="Minimum answer-quality score for SFT promotion.",
    )
    return parser.parse_args()


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _manifest_filename(input_path: str) -> str:
    """Build a stable, filesystem-safe filename for one input_path manifest."""
    candidate = re.sub(r"[^a-zA-Z0-9._-]+", "_", input_path.strip())[:80] or "empty_input_path"
    digest = hashlib.sha1(input_path.encode("utf-8")).hexdigest()[:10]
    return f"{candidate}__{digest}.json"


def _weighted_geometric_mean(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total_weight = sum(weights.values()) or 1.0
    log_sum = 0.0
    for name, weight in weights.items():
        value = max(1.0, scores.get(name, 0.0)) / 100.0
        log_sum += (weight / total_weight) * math.log(value)
    return round(math.exp(log_sum) * 100.0, 2)


def _safe_resolve(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    try:
        return Path(path_str).expanduser().resolve()
    except OSError:
        try:
            return Path(path_str).expanduser()
        except OSError:
            return None


def _normalize_path_token(token: str, root: Optional[Path] = None) -> str:
    text = str(token or "").strip().strip("\"'`")
    if not text:
        return ""
    text = text.replace("\\", "/").rstrip(".,;:")
    if text.startswith("file://"):
        text = text[7:]

    if root is not None:
        try:
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = (root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            return candidate.relative_to(root).as_posix()
        except Exception:
            pass

    return text.lstrip("./")


def _extract_json_candidate(text: str) -> Optional[Any]:
    stripped = (text or "").strip()
    if not stripped:
        return None

    candidates = [stripped]
    fenced = JSON_FENCE_RE.findall(stripped)
    candidates.extend(fenced)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate[0] not in "[{":
            continue
        try:
            return json.loads(candidate)
        except Exception:
            try:
                return ast.literal_eval(candidate)
            except Exception:
                continue
    return None


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _normalize_python_signature(code: str) -> str:
    normalized = qc.normalize_free_text(code)
    normalized = re.sub(r"([\"']).*?\1", "__STR__", normalized)
    normalized = re.sub(r"\b\d+(\.\d+)?\b", "__NUM__", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized[:4000]


def _extract_path_tokens(text: str, root: Optional[Path] = None) -> List[str]:
    tokens: Set[str] = set()
    if not text:
        return []

    for match in PATH_RE.findall(text):
        normalized = _normalize_path_token(match, root)
        if normalized:
            tokens.add(normalized)

    for match in FILE_LIKE_RE.findall(text):
        normalized = _normalize_path_token(match, root)
        if normalized:
            tokens.add(normalized)

    return sorted(tokens)


def _extract_extension_tokens(text: str) -> List[str]:
    extensions = {
        token.lower()
        for token in EXT_RE.findall(text or "")
        if 1 <= len(token) <= 10
    }
    return sorted(extensions)


def _normalize_count_label(raw_label: str) -> Optional[str]:
    key = qc.normalize_free_text(raw_label).casefold()
    return COUNT_LABELS.get(key)


def _extract_count_facts(text: str, rel_path: Optional[str] = None) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for label, value in COUNT_RE.findall(text or ""):
        normalized = _normalize_count_label(label)
        if normalized is None:
            continue
        facts.append({"kind": "count", "label": normalized, "value": int(value), "rel_path": rel_path})

    for rows, cols in SHAPE_RE.findall(text or ""):
        facts.append({"kind": "count", "label": "row_count", "value": int(rows), "rel_path": rel_path})
        facts.append({"kind": "count", "label": "column_count", "value": int(cols), "rel_path": rel_path})

    return facts


def _extract_column_tokens(text: str, rel_path: Optional[str] = None) -> List[Dict[str, Any]]:
    columns: Set[str] = set()
    for raw in COLUMN_BRACKET_RE.findall(text or ""):
        parsed = _extract_json_candidate(raw)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str) and item.strip():
                    columns.add(item.strip())
        else:
            for item in QUOTED_TOKEN_RE.findall(raw):
                if item.strip():
                    columns.add(item.strip())

    return [{"kind": "column", "value": value, "rel_path": rel_path} for value in sorted(columns)]


def _extract_structured_facts(data: Any, rel_path: Optional[str] = None, depth: int = 0) -> List[Dict[str, Any]]:
    if depth > 4:
        return []

    facts: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            key_text = qc.normalize_free_text(str(key)).casefold()
            normalized_label = _normalize_count_label(key_text)
            if normalized_label and isinstance(value, (int, float)):
                facts.append(
                    {"kind": "count", "label": normalized_label, "value": int(value), "rel_path": rel_path}
                )
            elif any(token in key_text for token in ("column", "schema", "field")) and isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        facts.append({"kind": "column", "value": item.strip(), "rel_path": rel_path})
            else:
                facts.extend(_extract_structured_facts(value, rel_path=rel_path, depth=depth + 1))
    elif isinstance(data, list):
        for item in data[:20]:
            facts.extend(_extract_structured_facts(item, rel_path=rel_path, depth=depth + 1))
    elif isinstance(data, str):
        for path_token in _extract_path_tokens(data):
            facts.append({"kind": "path", "value": path_token})
        for ext in _extract_extension_tokens(data):
            facts.append({"kind": "extension", "value": ext})
    return facts


def _dedupe_fact_records(facts: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[Any, ...]] = set()
    deduped: List[Dict[str, Any]] = []
    for fact in facts:
        key = (
            fact.get("kind"),
            fact.get("label"),
            fact.get("value"),
            fact.get("rel_path"),
            fact.get("source"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _summarize_csv_like(path: Path, delimiter: str) -> Dict[str, Any]:
    columns: List[str] = []
    row_count: Optional[int] = 0
    truncated = False
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            header = next(reader, None)
            if header is None:
                return {"columns": [], "row_count": 0, "truncated": False}
            columns = [str(item).strip() for item in header[:100]]
            for idx, _ in enumerate(reader, start=1):
                row_count = idx
                if idx >= 100000:
                    truncated = True
                    row_count = None
                    break
    except Exception as exc:
        return {"columns": [], "row_count": None, "truncated": False, "error": str(exc)}
    return {"columns": columns, "row_count": row_count, "truncated": truncated}


def _summarize_json_file(path: Path) -> Dict[str, Any]:
    try:
        if path.stat().st_size > 5 * 1024 * 1024:
            return {"columns": [], "row_count": None, "truncated": True}
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            columns: Set[str] = set()
            for item in data[:50]:
                if isinstance(item, dict):
                    columns.update(str(key) for key in item.keys())
            return {"columns": sorted(columns)[:100], "row_count": len(data), "truncated": False}
        if isinstance(data, dict):
            return {"columns": sorted(str(key) for key in data.keys())[:100], "row_count": 1, "truncated": False}
    except Exception as exc:
        return {"columns": [], "row_count": None, "truncated": False, "error": str(exc)}
    return {"columns": [], "row_count": None, "truncated": False}


def _summarize_jsonl_file(path: Path) -> Dict[str, Any]:
    columns: Set[str] = set()
    count = 0
    truncated = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip():
                    continue
                count += 1
                if count <= 50:
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            columns.update(str(key) for key in item.keys())
                    except Exception:
                        pass
                if count >= 100000:
                    truncated = True
                    count = 0
                    break
    except Exception as exc:
        return {"columns": [], "row_count": None, "truncated": False, "error": str(exc)}
    return {"columns": sorted(columns)[:100], "row_count": count or None, "truncated": truncated}


def _summarize_parquet_file(path: Path) -> Dict[str, Any]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet_file = pq.ParquetFile(path)
        return {
            "columns": list(parquet_file.schema.names[:100]),
            "row_count": parquet_file.metadata.num_rows if parquet_file.metadata else None,
            "truncated": False,
        }
    except Exception as exc:
        return {"columns": [], "row_count": None, "truncated": False, "error": str(exc)}


def _summarize_table_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _summarize_csv_like(path, ",")
    if suffix == ".tsv":
        return _summarize_csv_like(path, "\t")
    if suffix == ".json":
        return _summarize_json_file(path)
    if suffix == ".jsonl":
        return _summarize_jsonl_file(path)
    if suffix == ".parquet":
        return _summarize_parquet_file(path)
    return {"columns": [], "row_count": None, "truncated": False}


def build_dataset_manifest(
    input_path: str,
    max_files: int,
    top_extensions: int,
    top_subtrees: int,
    table_limit: int,
) -> Dict[str, Any]:
    root = _safe_resolve(input_path)
    if root is None or not root.exists():
        return {
            "available": False,
            "input_path": input_path,
            "reason": "missing_input_path",
            "expected_tool_budget": 4,
            "major_extensions": [],
            "dominant_subtrees": [],
            "representative_files": [],
            "table_summaries": [],
            "truncated": False,
            "count_exact": False,
        }

    extension_counter: Counter[str] = Counter()
    subtree_counter: Counter[str] = Counter()
    subtree_bytes: Counter[str] = Counter()
    representative_by_ext: Dict[str, Dict[str, Any]] = {}
    table_candidates: List[Tuple[int, str]] = []
    num_dirs = 0
    num_files = 0
    total_bytes = 0
    max_depth = 0
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root):
        num_dirs += 1
        rel_dir = Path(dirpath).relative_to(root)
        depth = 0 if str(rel_dir) == "." else len(rel_dir.parts)
        max_depth = max(max_depth, depth)

        for filename in sorted(filenames):
            if num_files >= max_files:
                truncated = True
                break

            file_path = Path(dirpath) / filename
            rel_path = file_path.relative_to(root).as_posix()
            suffix = file_path.suffix.lower() or "<no_ext>"

            try:
                size = file_path.stat().st_size
            except OSError:
                size = 0

            extension_counter[suffix] += 1
            total_bytes += size
            num_files += 1

            top_level = Path(rel_path).parts[0] if "/" in rel_path else "<root>"
            subtree_counter[top_level] += 1
            subtree_bytes[top_level] += size

            best = representative_by_ext.get(suffix)
            if best is None or size > best["size"]:
                representative_by_ext[suffix] = {"rel_path": rel_path, "size": size, "ext": suffix}

            if suffix in TABLE_EXTENSIONS:
                table_candidates.append((size, rel_path))

        if truncated:
            break

    major_extensions = []
    for ext, count in extension_counter.most_common(top_extensions):
        major_extensions.append(
            {
                "ext": ext,
                "count": count,
                "share": round(_safe_ratio(count, num_files), 4),
            }
        )

    dominant_subtrees = []
    for name, count in subtree_counter.most_common(top_subtrees):
        dominant_subtrees.append(
            {
                "name": name,
                "file_count": count,
                "byte_count": subtree_bytes[name],
                "share": round(_safe_ratio(count, num_files), 4),
            }
        )

    representative_files = sorted(representative_by_ext.values(), key=lambda item: (item["ext"], -item["size"]))
    representative_files = representative_files[: max(8, top_extensions + table_limit)]

    table_summaries = []
    for _, rel_path in sorted(table_candidates, reverse=True)[: table_limit * 3]:
        if any(item["rel_path"] == rel_path for item in table_summaries):
            continue
        summary = _summarize_table_file(root / rel_path)
        table_summaries.append(
            {
                "rel_path": rel_path,
                "ext": Path(rel_path).suffix.lower() or "<no_ext>",
                "columns": summary.get("columns", []),
                "row_count": summary.get("row_count"),
                "truncated": bool(summary.get("truncated")),
                "error": summary.get("error"),
            }
        )
        if len(table_summaries) >= table_limit:
            break

    expected_tool_budget = 3
    expected_tool_budget += min(6, int(math.ceil(math.log2(num_files + 1)))) if num_files > 0 else 0
    expected_tool_budget += min(4, len(major_extensions))
    expected_tool_budget += min(4, len(dominant_subtrees))
    expected_tool_budget += min(4, len(table_summaries))
    if truncated:
        expected_tool_budget += 2
    expected_tool_budget = max(4, min(24, expected_tool_budget))

    return {
        "available": True,
        "input_path": str(root),
        "root_name": root.name or str(root),
        "num_files": num_files,
        "num_dirs": num_dirs,
        "max_depth": max_depth,
        "total_bytes": total_bytes,
        "truncated": truncated,
        "count_exact": not truncated,
        "major_extensions": major_extensions,
        "dominant_subtrees": dominant_subtrees,
        "representative_files": representative_files,
        "table_summaries": table_summaries,
        "expected_tool_budget": expected_tool_budget,
    }


def derive_coverage_targets(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not manifest.get("available"):
        return []

    targets: List[Dict[str, Any]] = [
        {
            "id": "inventory",
            "type": "inventory",
            "weight": 2.0,
        }
    ]

    for ext_info in manifest.get("major_extensions", []):
        targets.append(
            {
                "id": f"ext::{ext_info['ext']}",
                "type": "extension",
                "ext": ext_info["ext"],
                "weight": round(1.0 + min(1.5, 3.0 * ext_info.get("share", 0.0)), 2),
            }
        )

    for subtree in manifest.get("dominant_subtrees", []):
        if subtree["name"] == "<root>":
            continue
        targets.append(
            {
                "id": f"subtree::{subtree['name']}",
                "type": "subtree",
                "name": subtree["name"],
                "weight": round(1.0 + min(1.5, 3.0 * subtree.get("share", 0.0)), 2),
            }
        )

    for table in manifest.get("table_summaries", []):
        targets.append(
            {
                "id": f"schema::{table['rel_path']}",
                "type": "schema",
                "rel_path": table["rel_path"],
                "columns": table.get("columns", []),
                "weight": 1.75,
            }
        )

    return targets


def extract_python_accesses(code: str, root: Optional[Path]) -> Dict[str, Any]:
    path_candidates: Set[str] = set(_extract_path_tokens(code, root))
    operations: Set[str] = set()

    strict_signature = qc.canonicalize_tool_call("python", code)
    loose_signature = f"python:{_normalize_python_signature(code)}"

    try:
        tree = ast.parse(code)
    except Exception:
        return {
            "path_candidates": sorted(path_candidates),
            "primary_path": next(iter(sorted(path_candidates)), None),
            "operations": ["execute_code"],
            "strict_signature": strict_signature,
            "loose_signature": loose_signature,
        }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func).lower()
            if "walk" in name:
                operations.add("walk")
            elif name.endswith("listdir") or name.endswith("iterdir"):
                operations.add("listdir")
            elif name.endswith("glob") or name.endswith("rglob"):
                operations.add("glob")
            elif "read_csv" in name or "read_table" in name or "read_excel" in name:
                operations.add("read_table")
            elif "read_json" in name:
                operations.add("read_json")
            elif "read_parquet" in name:
                operations.add("read_parquet")
            elif name == "open":
                operations.add("open_file")
            elif "dump" in name or "write" in name or "to_json" in name or "to_csv" in name:
                operations.add("write_file")
            elif "head" in name or "sample" in name:
                operations.add("sample_rows")

            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    normalized = _normalize_path_token(arg.value, root)
                    if normalized:
                        path_candidates.add(normalized)

    primary_path = None
    for candidate in sorted(path_candidates):
        if Path(candidate).suffix.lower() in TABLE_EXTENSIONS:
            primary_path = candidate
            break
    if primary_path is None and path_candidates:
        primary_path = sorted(path_candidates)[0]

    if not operations:
        operations.add("execute_code")

    return {
        "path_candidates": sorted(path_candidates),
        "primary_path": primary_path,
        "operations": sorted(operations),
        "strict_signature": strict_signature,
        "loose_signature": loose_signature,
    }


def _analyze_tool_block(tag: str, content: str, root: Optional[Path]) -> Dict[str, Any]:
    if tag == "python":
        return extract_python_accesses(content, root)

    operations: Set[str] = set()
    if tag in qc.SEARCH_LIKE_TAGS:
        operations.add("search")
    elif tag in {"bash", "sql", "code", "calculator"}:
        operations.add("execute_code")
    else:
        operations.add(f"use_{tag}")

    path_candidates = _extract_path_tokens(content, root)
    primary_path = None
    for candidate in path_candidates:
        if Path(candidate).suffix.lower() in TABLE_EXTENSIONS:
            primary_path = candidate
            break
    if primary_path is None and path_candidates:
        primary_path = path_candidates[0]

    return {
        "path_candidates": path_candidates,
        "primary_path": primary_path,
        "operations": sorted(operations),
        "strict_signature": qc.canonicalize_tool_call(tag, content),
        "loose_signature": f"{tag}:{qc.normalize_free_text(content).casefold()[:4000]}",
    }


def extract_result_evidence(
    result_text: str,
    step_index: int,
    step_context: Dict[str, Any],
    root: Optional[Path],
) -> List[Dict[str, Any]]:
    rel_path = step_context.get("primary_path")
    facts: List[Dict[str, Any]] = []

    for path_token in _extract_path_tokens(result_text, root):
        facts.append({"kind": "path", "value": path_token, "source": f"R{step_index}"})

    for ext in _extract_extension_tokens(result_text):
        facts.append({"kind": "extension", "value": ext, "source": f"R{step_index}"})

    for fact in _extract_count_facts(result_text, rel_path=rel_path):
        fact["source"] = f"R{step_index}"
        facts.append(fact)

    for fact in _extract_column_tokens(result_text, rel_path=rel_path):
        fact["source"] = f"R{step_index}"
        facts.append(fact)

    structured = _extract_json_candidate(result_text)
    if structured is not None:
        for fact in _extract_structured_facts(structured, rel_path=rel_path):
            fact["source"] = f"R{step_index}"
            facts.append(fact)

    return _dedupe_fact_records(facts)


def build_trace_steps(blocks: List[Dict[str, Any]], root: Optional[Path]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    last_think = ""
    idx = 0

    while idx < len(blocks):
        block = blocks[idx]
        tag = block["tag"]

        if tag == "think":
            last_think = block["content"]
            idx += 1
            continue

        if tag in {"result", "answer"}:
            idx += 1
            continue

        result_block = None
        if idx + 1 < len(blocks) and blocks[idx + 1]["tag"] == "result":
            result_block = blocks[idx + 1]

        tool_info = _analyze_tool_block(tag, block["content"], root)
        result_text = result_block["content"] if result_block else ""
        result_error = bool(result_block) and qc.result_contains_error(result_text)
        evidence_items = extract_result_evidence(result_text, len(steps) + 1, tool_info, root) if result_block else []
        informative_result = bool(
            result_block and not result_error and (evidence_items or qc.normalize_free_text(result_text))
        )

        steps.append(
            {
                "step_index": len(steps) + 1,
                "think_before": last_think,
                "tool_tag": tag,
                "tool_content": block["content"],
                "result_content": result_text,
                "result_error": result_error,
                "informative_result": informative_result,
                "success": bool(result_block) and not result_error,
                "operations": tool_info["operations"],
                "path_candidates": tool_info["path_candidates"],
                "primary_path": tool_info["primary_path"],
                "strict_signature": tool_info["strict_signature"],
                "loose_signature": tool_info["loose_signature"],
                "evidence_items": evidence_items,
            }
        )

        idx += 2 if result_block else 1

    return steps


def extract_answer_claims(answer_text: str, root: Optional[Path]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []

    for path_token in _extract_path_tokens(answer_text, root):
        claims.append({"kind": "path", "value": path_token, "weight": 1.0, "critical": True})

    for ext in _extract_extension_tokens(answer_text):
        claims.append({"kind": "extension", "value": ext, "weight": 0.7, "critical": False})

    for fact in _extract_count_facts(answer_text):
        fact["weight"] = 1.4
        fact["critical"] = True
        claims.append(fact)

    for fact in _extract_column_tokens(answer_text):
        fact["weight"] = 1.1
        fact["critical"] = True
        claims.append(fact)

    structured = _extract_json_candidate(answer_text)
    if structured is not None:
        for fact in _extract_structured_facts(structured):
            if fact["kind"] == "count":
                fact["weight"] = 1.4
                fact["critical"] = True
            elif fact["kind"] == "column":
                fact["weight"] = 1.1
                fact["critical"] = True
            elif fact["kind"] == "path":
                fact["weight"] = 1.0
                fact["critical"] = True
            else:
                fact["weight"] = 0.7
                fact["critical"] = False
            claims.append(fact)

    return _dedupe_fact_records(claims)


def inspect_meta_output(meta_output_path: str) -> Dict[str, Any]:
    path = _safe_resolve(meta_output_path)
    if path is None:
        return {"path": meta_output_path, "exists": False, "nonempty": False, "valid_json": False}
    if not path.exists():
        return {"path": str(path), "exists": False, "nonempty": False, "valid_json": False}

    nonempty = False
    valid_json = False
    try:
        nonempty = path.stat().st_size > 0
    except OSError:
        pass

    if path.suffix.lower() == ".json" and nonempty:
        try:
            json.loads(path.read_text(encoding="utf-8"))
            valid_json = True
        except Exception:
            valid_json = False

    return {"path": str(path), "exists": True, "nonempty": nonempty, "valid_json": valid_json}


def _build_support_index(
    manifest: Dict[str, Any],
    evidence_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    evidence_paths = {item["value"] for item in evidence_items if item["kind"] == "path"}
    evidence_basenames = {Path(path).name for path in evidence_paths}
    evidence_exts = {item["value"] for item in evidence_items if item["kind"] == "extension"}
    evidence_columns = {item["value"] for item in evidence_items if item["kind"] == "column"}
    evidence_counts: Dict[str, Set[int]] = defaultdict(set)
    for item in evidence_items:
        if item["kind"] == "count":
            evidence_counts[item["label"]].add(int(item["value"]))

    manifest_paths: Set[str] = set()
    manifest_basenames: Set[str] = set()
    manifest_exts: Set[str] = set()
    manifest_columns: Set[str] = set()
    manifest_counts: Dict[str, Set[int]] = defaultdict(set)
    manifest_exact_counts: Dict[str, bool] = defaultdict(lambda: False)

    for item in manifest.get("representative_files", []):
        rel_path = item["rel_path"]
        manifest_paths.add(rel_path)
        manifest_basenames.add(Path(rel_path).name)
        manifest_exts.add(item["ext"])

    for item in manifest.get("table_summaries", []):
        rel_path = item["rel_path"]
        manifest_paths.add(rel_path)
        manifest_basenames.add(Path(rel_path).name)
        if item.get("ext"):
            manifest_exts.add(item["ext"])
        for column in item.get("columns", []):
            manifest_columns.add(column)
        if item.get("row_count") is not None:
            manifest_counts["row_count"].add(int(item["row_count"]))
            manifest_exact_counts["row_count"] = not item.get("truncated", False)

    for ext in manifest.get("major_extensions", []):
        manifest_exts.add(ext["ext"])

    if manifest.get("count_exact"):
        manifest_counts["file_count"].add(int(manifest.get("num_files", 0)))
        manifest_counts["dir_count"].add(int(manifest.get("num_dirs", 0)))
        manifest_exact_counts["file_count"] = True
        manifest_exact_counts["dir_count"] = True

    return {
        "evidence_paths": evidence_paths,
        "evidence_basenames": evidence_basenames,
        "evidence_exts": evidence_exts,
        "evidence_columns": evidence_columns,
        "evidence_counts": evidence_counts,
        "manifest_paths": manifest_paths,
        "manifest_basenames": manifest_basenames,
        "manifest_exts": manifest_exts,
        "manifest_columns": manifest_columns,
        "manifest_counts": manifest_counts,
        "manifest_exact_counts": manifest_exact_counts,
    }


def match_claims_to_support(
    claims: List[Dict[str, Any]],
    evidence_items: List[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    index = _build_support_index(manifest, evidence_items)
    matches: List[Dict[str, Any]] = []

    supported_weight = 0.0
    manifest_only_weight = 0.0
    contradicted_weight = 0.0
    unsupported_critical_weight = 0.0
    total_weight = 0.0

    for claim in claims:
        weight = float(claim.get("weight", 1.0))
        total_weight += weight
        evidence_support = False
        manifest_support = False
        contradiction = False

        if claim["kind"] == "path":
            value = claim["value"]
            basename = Path(value).name
            evidence_support = value in index["evidence_paths"] or basename in index["evidence_basenames"]
            manifest_support = value in index["manifest_paths"] or basename in index["manifest_basenames"]
        elif claim["kind"] == "extension":
            value = claim["value"]
            evidence_support = value in index["evidence_exts"]
            manifest_support = value in index["manifest_exts"]
        elif claim["kind"] == "column":
            value = claim["value"]
            evidence_support = value in index["evidence_columns"]
            manifest_support = value in index["manifest_columns"]
        elif claim["kind"] == "count":
            label = claim["label"]
            value = int(claim["value"])
            evidence_values = index["evidence_counts"].get(label, set())
            manifest_values = index["manifest_counts"].get(label, set())
            evidence_support = value in evidence_values
            manifest_support = value in manifest_values
            if not evidence_support and not manifest_support:
                if evidence_values:
                    contradiction = True
                elif index["manifest_exact_counts"].get(label, False) and manifest_values:
                    contradiction = True

        if evidence_support:
            supported_weight += weight
        elif manifest_support:
            manifest_only_weight += weight

        if contradiction:
            contradicted_weight += weight

        if claim.get("critical") and not evidence_support and not manifest_support:
            unsupported_critical_weight += weight

        matches.append(
            {
                "claim": claim,
                "supported_by_evidence": evidence_support,
                "supported_by_manifest": manifest_support,
                "contradiction": contradiction,
            }
        )

    supported_count = sum(1 for item in matches if item["supported_by_evidence"] or item["supported_by_manifest"])
    contradicted_count = sum(1 for item in matches if item["contradiction"])
    unsupported_count = sum(
        1
        for item in matches
        if not item["supported_by_evidence"] and not item["supported_by_manifest"]
    )

    return {
        "matches": matches,
        "claim_count": len(claims),
        "supported_claim_count": supported_count,
        "contradicted_claim_count": contradicted_count,
        "unsupported_claim_count": unsupported_count,
        "total_weight": total_weight,
        "supported_weight": supported_weight,
        "manifest_only_weight": manifest_only_weight,
        "contradicted_weight": contradicted_weight,
        "unsupported_critical_weight": unsupported_critical_weight,
    }


def evaluate_coverage(
    manifest: Dict[str, Any],
    targets: List[Dict[str, Any]],
    trace_steps: List[Dict[str, Any]],
    evidence_items: List[Dict[str, Any]],
    answer_claims: List[Dict[str, Any]],
    answer_text: str,
) -> Dict[str, Any]:
    trace_paths: Set[str] = set()
    trace_exts: Set[str] = set()
    trace_subtrees: Set[str] = set()
    schema_paths: Set[str] = set()
    inventory_seen = False

    for step in trace_steps:
        ops = set(step.get("operations", []))
        if ops & ROOT_LISTING_OPS:
            inventory_seen = True
        for path in step.get("path_candidates", []):
            trace_paths.add(path)
            if "/" in path:
                trace_subtrees.add(path.split("/", 1)[0])
            ext = Path(path).suffix.lower()
            if ext:
                trace_exts.add(ext)
        for item in step.get("evidence_items", []):
            if item["kind"] == "path":
                trace_paths.add(item["value"])
                if "/" in item["value"]:
                    trace_subtrees.add(item["value"].split("/", 1)[0])
            elif item["kind"] == "extension":
                trace_exts.add(item["value"])
            elif item["kind"] == "column" and item.get("rel_path"):
                schema_paths.add(item["rel_path"])
            elif item["kind"] == "count" and item.get("rel_path") and item["label"] in {"row_count", "column_count"}:
                schema_paths.add(item["rel_path"])

    answer_exts = {claim["value"] for claim in answer_claims if claim["kind"] == "extension"}
    answer_paths = {claim["value"] for claim in answer_claims if claim["kind"] == "path"}
    answer_columns = {claim["value"] for claim in answer_claims if claim["kind"] == "column"}
    answer_counts = {claim["label"] for claim in answer_claims if claim["kind"] == "count"}
    answer_lower = qc.normalize_free_text(answer_text).casefold()

    if not targets:
        diversity = len({op for step in trace_steps for op in step.get("operations", [])})
        score = _clamp(15.0 + 12.0 * diversity + 4.0 * len(trace_paths) + 3.0 * len(answer_claims))
        return {
            "coverage_score": round(score, 2),
            "coverage_target_count": 0,
            "trace_hit_count": 0,
            "answer_hit_count": 0,
            "trace_weighted_recall": 0.0,
            "answer_weighted_recall": 0.0,
            "target_hits": [],
        }

    total_weight = sum(target["weight"] for target in targets) or 1.0
    trace_weight = 0.0
    answer_weight = 0.0
    target_hits: List[Dict[str, Any]] = []

    for target in targets:
        trace_hit = False
        answer_hit = False
        target_type = target["type"]

        if target_type == "inventory":
            trace_hit = inventory_seen or bool({"file_count", "dir_count"} & answer_counts)
            answer_hit = bool({"file_count", "dir_count"} & answer_counts) or any(
                keyword in answer_lower for keyword in ("extension", "directory", "folder", "file")
            )
        elif target_type == "extension":
            trace_hit = target["ext"] in trace_exts
            answer_hit = target["ext"] in answer_exts or target["ext"] in answer_lower
        elif target_type == "subtree":
            trace_hit = target["name"] in trace_subtrees
            answer_hit = target["name"].casefold() in answer_lower or any(
                path.startswith(target["name"] + "/") or path == target["name"]
                for path in answer_paths
            )
        elif target_type == "schema":
            rel_path = target["rel_path"]
            basename = Path(rel_path).name
            trace_hit = rel_path in schema_paths or rel_path in trace_paths or basename in {Path(p).name for p in trace_paths}
            answer_hit = (
                rel_path in answer_paths
                or basename in {Path(p).name for p in answer_paths}
                or bool(set(target.get("columns", [])) & answer_columns)
            )

        if trace_hit:
            trace_weight += target["weight"]
        if answer_hit:
            answer_weight += target["weight"]

        target_hits.append(
            {
                "target_id": target["id"],
                "trace_hit": trace_hit,
                "answer_hit": answer_hit,
                "weight": target["weight"],
            }
        )

    trace_recall = _safe_ratio(trace_weight, total_weight)
    answer_recall = _safe_ratio(answer_weight, total_weight)
    score = _clamp(100.0 * (0.7 * trace_recall + 0.3 * answer_recall))
    return {
        "coverage_score": round(score, 2),
        "coverage_target_count": len(targets),
        "trace_hit_count": sum(1 for item in target_hits if item["trace_hit"]),
        "answer_hit_count": sum(1 for item in target_hits if item["answer_hit"]),
        "trace_weighted_recall": round(trace_recall, 4),
        "answer_weighted_recall": round(answer_recall, 4),
        "target_hits": target_hits,
    }


def _compute_grounding_score(match_info: Dict[str, Any], answer_claims: List[Dict[str, Any]]) -> float:
    total_weight = match_info["total_weight"]
    if total_weight <= 0:
        concrete_tokens = len(answer_claims)
        return _clamp(10.0 + 6.0 * concrete_tokens, 0.0, 35.0)

    score = 100.0 * (
        match_info["supported_weight"] + 0.4 * match_info["manifest_only_weight"]
    ) / total_weight
    score -= 35.0 * _safe_ratio(match_info["contradicted_weight"], total_weight)
    score -= 20.0 * _safe_ratio(match_info["unsupported_critical_weight"], total_weight)
    return _clamp(score)


def _count_recoveries(trace_steps: List[Dict[str, Any]]) -> int:
    recoveries = 0
    for idx, step in enumerate(trace_steps):
        if not step["result_error"]:
            continue
        current_paths = set(step.get("path_candidates", []))
        current_ops = set(step.get("operations", []))
        for later in trace_steps[idx + 1 : idx + 4]:
            if not later["success"]:
                continue
            later_paths = set(later.get("path_candidates", []))
            later_ops = set(later.get("operations", []))
            if current_paths & later_paths or current_ops & later_ops:
                recoveries += 1
                break
    return recoveries


def _compute_execution_score_v2(trace_steps: List[Dict[str, Any]]) -> Tuple[float, int, int, int]:
    if not trace_steps:
        return 0.0, 0, 0, 0

    total_calls = len(trace_steps)
    successful_calls = sum(1 for step in trace_steps if step["success"])
    informative_calls = sum(1 for step in trace_steps if step["informative_result"])
    result_error_count = sum(1 for step in trace_steps if step["result_error"])
    recovered_failures = _count_recoveries(trace_steps)

    success_rate = _safe_ratio(successful_calls, total_calls)
    informative_rate = _safe_ratio(informative_calls, total_calls)
    recovery_rate = _safe_ratio(recovered_failures, result_error_count) if result_error_count else 1.0
    score = 100.0 * (0.50 * success_rate + 0.35 * informative_rate + 0.15 * recovery_rate)
    return _clamp(score), informative_calls, result_error_count, recovered_failures


def _compute_efficiency_score_v2(
    trace_steps: List[Dict[str, Any]],
    expected_tool_budget: int,
    exact_duplicate_call_count: int,
    near_duplicate_call_count: int,
    informative_result_count: int,
    tool_frequency_threshold: int,
) -> float:
    total_calls = len(trace_steps)
    if total_calls == 0:
        return 0.0

    budget_ratio = _safe_ratio(total_calls, max(1, expected_tool_budget))
    duplicate_rate = _safe_ratio(exact_duplicate_call_count + near_duplicate_call_count, total_calls)
    low_yield_rate = _safe_ratio(total_calls - informative_result_count, total_calls)
    over_threshold = max(0, total_calls - tool_frequency_threshold)

    score = 100.0
    if budget_ratio > 1.0:
        score -= min(30.0, 25.0 * (budget_ratio - 1.0))
    score -= min(25.0, 35.0 * duplicate_rate)
    score -= min(35.0, 35.0 * low_yield_rate)
    if over_threshold > 0:
        score -= min(15.0, 2.0 * over_threshold)
    return _clamp(score)


def _compute_answer_quality_score(
    answer_text: str,
    answer_claims: List[Dict[str, Any]],
    match_info: Dict[str, Any],
    coverage_info: Dict[str, Any],
    meta_output_info: Dict[str, Any],
) -> float:
    answer_lower = qc.normalize_free_text(answer_text).casefold()
    answer_json = _extract_json_candidate(answer_text)
    paragraph_count = len([chunk for chunk in answer_text.split("\n\n") if chunk.strip()])
    section_hits = sum(
        1 for keywords in SECTION_KEYWORDS.values() if any(keyword in answer_lower for keyword in keywords)
    )

    structure = 0.15
    if isinstance(answer_json, (dict, list)):
        structure += 0.40
    structure += 0.20 * min(1.0, paragraph_count / 3.0)
    structure += 0.20 * min(1.0, section_hits / 3.0)
    if meta_output_info.get("valid_json"):
        structure += 0.05
    structure = min(1.0, structure)

    unique_paths = len({claim["value"] for claim in answer_claims if claim["kind"] == "path"})
    unique_exts = len({claim["value"] for claim in answer_claims if claim["kind"] == "extension"})
    unique_counts = len({(claim.get("label"), claim.get("value")) for claim in answer_claims if claim["kind"] == "count"})
    unique_columns = len({claim["value"] for claim in answer_claims if claim["kind"] == "column"})
    specificity = min(1.0, (unique_paths + unique_exts + 1.2 * unique_counts + 0.5 * unique_columns) / 8.0)

    total_weight = match_info["total_weight"]
    if total_weight > 0:
        consistency = 1.0
        consistency -= 0.9 * _safe_ratio(match_info["contradicted_weight"], total_weight)
        consistency -= 0.4 * _safe_ratio(match_info["unsupported_critical_weight"], total_weight)
        consistency = max(0.0, consistency)
    else:
        consistency = 0.35 if answer_text.strip() else 0.0

    coverage_score = coverage_info["coverage_score"]
    if coverage_score < 60.0:
        calibration = 1.0 if UNCERTAINTY_RE.search(answer_text or "") else 0.30
    else:
        calibration = 0.90 if match_info["contradicted_claim_count"] == 0 else 0.50

    score = 100.0 * (
        0.35 * structure
        + 0.25 * specificity
        + 0.25 * consistency
        + 0.15 * calibration
    )
    return _clamp(score)


def quality_assessment_v2(
    raw_record: Dict[str, Any],
    normalized_trajectory: str,
    format_info: Dict[str, Any],
    manifest: Dict[str, Any],
    coverage_targets: List[Dict[str, Any]],
    tool_frequency_threshold: int,
) -> Dict[str, Any]:
    blocks = format_info["blocks"]
    root = _safe_resolve(raw_record.get("input_path", ""))
    answer_text = qc.extract_answer(normalized_trajectory)
    answer_nonempty = bool(answer_text.strip())

    gate_fail_reasons: List[str] = []
    if not format_info["format_valid"]:
        gate_fail_reasons.append("format_invalid")
    if not answer_nonempty:
        gate_fail_reasons.append("empty_answer")

    split_gate_pass = bool(answer_nonempty) and int(format_info.get("answer_count", 0)) == 1

    trace_steps = build_trace_steps(blocks, root)
    evidence_items = _dedupe_fact_records(
        item
        for step in trace_steps
        for item in step.get("evidence_items", [])
    )
    answer_claims = extract_answer_claims(answer_text, root)
    match_info = match_claims_to_support(answer_claims, evidence_items, manifest)
    coverage_info = evaluate_coverage(manifest, coverage_targets, trace_steps, evidence_items, answer_claims, answer_text)
    execution_score, informative_result_count, result_error_count, recovered_failures = _compute_execution_score_v2(
        trace_steps
    )

    strict_counter = Counter(step["strict_signature"] for step in trace_steps)
    loose_counter = Counter(step["loose_signature"] for step in trace_steps)
    exact_duplicate_call_count = sum(count - 1 for count in strict_counter.values() if count > 1)
    near_duplicate_call_count = max(
        0,
        sum(count - 1 for count in loose_counter.values() if count > 1) - exact_duplicate_call_count,
    )

    expected_tool_budget = manifest.get("expected_tool_budget") if manifest.get("available") else max(
        4, tool_frequency_threshold
    )
    efficiency_score = _compute_efficiency_score_v2(
        trace_steps,
        expected_tool_budget=expected_tool_budget,
        exact_duplicate_call_count=exact_duplicate_call_count,
        near_duplicate_call_count=near_duplicate_call_count,
        informative_result_count=informative_result_count,
        tool_frequency_threshold=tool_frequency_threshold,
    )

    meta_output_info = inspect_meta_output(raw_record.get("meta_output_path", ""))
    grounding_score = _compute_grounding_score(match_info, answer_claims)
    answer_quality_score = _compute_answer_quality_score(
        answer_text,
        answer_claims,
        match_info,
        coverage_info,
        meta_output_info,
    )

    component_scores = {
        "grounding": grounding_score,
        "coverage": coverage_info["coverage_score"],
        "execution": execution_score,
        "efficiency": efficiency_score,
        "answer_quality": answer_quality_score,
    }
    composite_score = _weighted_geometric_mean(component_scores, V2_WEIGHTS)

    warnings: List[str] = []
    if not manifest.get("available"):
        warnings.append("manifest_unavailable")
    if len(trace_steps) == 0:
        warnings.append("no_tool_calls")
    if result_error_count > 0:
        warnings.append("result_errors")
    if exact_duplicate_call_count > 0:
        warnings.append("duplicate_tool_calls")
    if near_duplicate_call_count > 0:
        warnings.append("near_duplicate_tool_calls")
    if len(trace_steps) > tool_frequency_threshold:
        warnings.append("tool_frequency_exceeded")
    if match_info["unsupported_claim_count"] > 0:
        warnings.append("unsupported_answer_claims")
    if match_info["contradicted_claim_count"] > 0:
        warnings.append("answer_evidence_conflict")
    if coverage_info["coverage_score"] < 50.0:
        warnings.append("underexplored_dataset")

    return {
        "gate_pass": len(gate_fail_reasons) == 0,
        "gate_fail_reasons": gate_fail_reasons,
        "split_gate_pass": split_gate_pass,
        "composite_score": composite_score,
        "grounding_score": round(grounding_score, 2),
        "coverage_score": round(coverage_info["coverage_score"], 2),
        "execution_score": round(execution_score, 2),
        "efficiency_score": round(efficiency_score, 2),
        "answer_quality_score": round(answer_quality_score, 2),
        "reward": round(composite_score / 100.0, 4),
        "warnings": warnings,
        "answer_nonempty": answer_nonempty,
        "tool_call_count": len(trace_steps),
        "result_error_count": result_error_count,
        "informative_result_count": informative_result_count,
        "recovered_failure_count": recovered_failures,
        "exact_duplicate_call_count": exact_duplicate_call_count,
        "near_duplicate_call_count": near_duplicate_call_count,
        "claim_count": match_info["claim_count"],
        "supported_claim_count": match_info["supported_claim_count"],
        "contradicted_claim_count": match_info["contradicted_claim_count"],
        "unsupported_claim_count": match_info["unsupported_claim_count"],
        "coverage_target_count": coverage_info["coverage_target_count"],
        "coverage_trace_hit_count": coverage_info["trace_hit_count"],
        "coverage_answer_hit_count": coverage_info["answer_hit_count"],
        "coverage_trace_recall": coverage_info["trace_weighted_recall"],
        "coverage_answer_recall": coverage_info["answer_weighted_recall"],
        "manifest_available": bool(manifest.get("available")),
        "manifest_truncated": bool(manifest.get("truncated")),
        "expected_tool_budget": expected_tool_budget,
        "meta_output_written": bool(meta_output_info.get("exists")),
        "meta_output_valid_json": bool(meta_output_info.get("valid_json")),
    }


def build_sft_record_v2(
    sample_id_value: str,
    tir_record: Dict[str, Any],
    quality_info: Dict[str, Any],
    save_normalized_trajectory: bool,
    reference: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    record = {
        "sample_id": sample_id_value,
        "input": qc.base_prompt(tir_record),
        "output": tir_record.get("output", ""),
        "prediction": tir_record.get("prediction", ""),
        "composite_score": quality_info["composite_score"],
        "quality_detail": {
            "grounding_score": quality_info["grounding_score"],
            "coverage_score": quality_info["coverage_score"],
            "execution_score": quality_info["execution_score"],
            "efficiency_score": quality_info["efficiency_score"],
            "answer_quality_score": quality_info["answer_quality_score"],
        },
        "split": "sft",
    }
    if reference is not None:
        record["reference"] = reference
    if category is not None:
        record["difficulty_category"] = category
    if save_normalized_trajectory:
        record["normalized_output"] = tir_record.get("normalized_output", "")
    return record


def build_rl_record_v2(
    sample_id_value: str,
    tir_record: Dict[str, Any],
    quality_info: Dict[str, Any],
    save_normalized_trajectory: bool,
    reference: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    record = {
        "sample_id": sample_id_value,
        "input": qc.base_prompt(tir_record),
        "output": tir_record.get("output", ""),
        "prediction": tir_record.get("prediction", ""),
        "composite_score": quality_info["composite_score"],
        "reward": quality_info["reward"],
        "quality_detail": {
            "grounding_score": quality_info["grounding_score"],
            "coverage_score": quality_info["coverage_score"],
            "execution_score": quality_info["execution_score"],
            "efficiency_score": quality_info["efficiency_score"],
            "answer_quality_score": quality_info["answer_quality_score"],
        },
        "split": "rl",
    }
    if reference is not None:
        record["reference"] = reference
    if category is not None:
        record["difficulty_category"] = category
    if save_normalized_trajectory:
        record["normalized_output"] = tir_record.get("normalized_output", "")
    return record


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    tir_records = qc.load_records(args.tir_file)
    dr_records = qc.load_records(args.dr_file) if args.dr_file else []
    reference_records = qc.load_records(args.reference_file) if args.reference_file else []

    dr_index = qc.build_index(dr_records, args.id_keys) if dr_records else {}
    reference_index = qc.build_index(reference_records, args.id_keys) if reference_records else {}

    precomputed_samples: List[Dict[str, Any]] = []
    for idx, raw_tir_record in enumerate(tir_records):
        sid = qc.sample_id(raw_tir_record, idx, args.id_keys)
        trajectory = qc.extract_trajectory(raw_tir_record, args.trajectory_keys)
        normalized_trajectory = qc.normalize_special_tokens(trajectory)
        format_info = _analyze_format_tolerant(normalized_trajectory)
        precomputed_samples.append(
            {
                "sample_id": sid,
                "raw_tir_record": raw_tir_record,
                "normalized_trajectory": normalized_trajectory,
                "format_info": format_info,
            }
        )

    if args.tool_frequency_threshold is None:
        tool_frequency_threshold, threshold_mode = qc.infer_tool_frequency_threshold(precomputed_samples)
    else:
        tool_frequency_threshold, threshold_mode = args.tool_frequency_threshold, "manual"

    manifest_cache: Dict[str, Dict[str, Any]] = {}
    target_cache: Dict[str, List[Dict[str, Any]]] = {}

    sample_reports: List[Dict[str, Any]] = []
    filtered_tir: List[Dict[str, Any]] = []
    sft_data: List[Dict[str, Any]] = []
    rl_data: List[Dict[str, Any]] = []
    num_samples_with_reference = 0
    num_samples_with_dr = 0
    num_samples_classified = 0

    for precomputed in precomputed_samples:
        sid = precomputed["sample_id"]
        raw_tir_record = precomputed["raw_tir_record"]
        normalized_trajectory = precomputed["normalized_trajectory"]
        format_info = precomputed["format_info"]

        input_path = str(raw_tir_record.get("input_path", "") or "")
        if input_path not in manifest_cache:
            manifest_cache[input_path] = build_dataset_manifest(
                input_path=input_path,
                max_files=args.manifest_max_files,
                top_extensions=args.manifest_top_extensions,
                top_subtrees=args.manifest_top_subtrees,
                table_limit=args.manifest_table_limit,
            )
            target_cache[input_path] = derive_coverage_targets(manifest_cache[input_path])

        manifest = manifest_cache[input_path]
        manifest_path = manifest_dir / _manifest_filename(input_path)
        with manifest_path.open("w", encoding="utf-8") as mf:
            json.dump(manifest, mf, ensure_ascii=False, indent=2)
            
        coverage_targets = target_cache[input_path]
        quality_info = quality_assessment_v2(
            raw_record=raw_tir_record,
            normalized_trajectory=normalized_trajectory,
            format_info=format_info,
            manifest=manifest,
            coverage_targets=coverage_targets,
            tool_frequency_threshold=tool_frequency_threshold,
        )

        tir_prediction = qc.extract_prediction(raw_tir_record, normalized_trajectory, args.prediction_keys)
        reference = qc.extract_reference(sid, raw_tir_record, reference_index, args.reference_keys)
        tir_correct = qc.answers_match(tir_prediction, reference)

        enriched_tir_record = dict(raw_tir_record)
        enriched_tir_record["sample_id"] = sid
        enriched_tir_record["prediction"] = tir_prediction
        enriched_tir_record["quality_check_v2"] = {
            "format_valid": format_info["format_valid"],
            "format_errors": qc.compact_errors(format_info["errors"]),
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
            dr_output = qc.extract_trajectory(dr_record, args.trajectory_keys)
            dr_prediction = qc.extract_prediction(dr_record, dr_output, args.prediction_keys)
            reference = reference or qc.extract_reference(sid, dr_record, reference_index, args.reference_keys)
            dr_correct = qc.answers_match(dr_prediction, reference)
            category = qc.classify_sample(dr_correct, tir_correct)
            if category is not None:
                num_samples_classified += 1

        if reference:
            num_samples_with_reference += 1

        composite = quality_info["composite_score"]
        sft_eligible = (
            quality_info["split_gate_pass"]
            and composite >= args.sft_threshold
            and quality_info["grounding_score"] >= args.sft_min_grounding
            and quality_info["coverage_score"] >= args.sft_min_coverage
            and quality_info["answer_quality_score"] >= args.sft_min_answer_quality
        )

        if sft_eligible:
            split = "sft"
        elif quality_info["split_gate_pass"] and composite >= args.rl_threshold:
            split = "rl"
        else:
            split = "discard"

        sample_report = {
            "sample_id": sid,
            "input_preview": qc.normalize_free_text(qc.base_prompt(raw_tir_record))[:240],
            "format_valid": format_info["format_valid"],
            "format_errors": qc.compact_errors(format_info["errors"]),
            "gate_pass": quality_info["gate_pass"],
            "split_gate_pass": quality_info["split_gate_pass"],
            "gate_fail_reasons": quality_info["gate_fail_reasons"],
            "composite_score": composite,
            "grounding_score": quality_info["grounding_score"],
            "coverage_score": quality_info["coverage_score"],
            "execution_score": quality_info["execution_score"],
            "efficiency_score": quality_info["efficiency_score"],
            "answer_quality_score": quality_info["answer_quality_score"],
            "reward": quality_info["reward"],
            "split": split,
            "warnings": quality_info["warnings"],
            "tool_call_count": quality_info["tool_call_count"],
            "result_error_count": quality_info["result_error_count"],
            "informative_result_count": quality_info["informative_result_count"],
            "exact_duplicate_call_count": quality_info["exact_duplicate_call_count"],
            "near_duplicate_call_count": quality_info["near_duplicate_call_count"],
            "claim_count": quality_info["claim_count"],
            "supported_claim_count": quality_info["supported_claim_count"],
            "unsupported_claim_count": quality_info["unsupported_claim_count"],
            "coverage_target_count": quality_info["coverage_target_count"],
            "coverage_trace_hit_count": quality_info["coverage_trace_hit_count"],
            "coverage_answer_hit_count": quality_info["coverage_answer_hit_count"],
            "manifest_available": quality_info["manifest_available"],
            "expected_tool_budget": quality_info["expected_tool_budget"],
            "meta_output_written": quality_info["meta_output_written"],
            "meta_output_valid_json": quality_info["meta_output_valid_json"],
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
                build_sft_record_v2(
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
                build_rl_record_v2(
                    sample_id_value=sid,
                    tir_record=enriched_tir_record,
                    quality_info=quality_info,
                    save_normalized_trajectory=args.save_normalized_trajectory,
                    reference=reference,
                    category=category,
                )
            )

    processed_ids = {sample["sample_id"] for sample in sample_reports}

    filtered_tir_path = output_dir / "filtered_tir.json"
    sft_path = output_dir / "sft_data.json"
    rl_path = output_dir / "rl_data.json"
    report_path = output_dir / "quality_report.json"

    filtered_tir = qc._incremental_merge(qc._load_json_list(filtered_tir_path), filtered_tir, processed_ids, "sample_id")
    sft_data = qc._incremental_merge(qc._load_json_list(sft_path), sft_data, processed_ids, "sample_id")
    rl_data = qc._incremental_merge(qc._load_json_list(rl_path), rl_data, processed_ids, "sample_id")

    existing_report_samples: List[Dict[str, Any]] = []
    if report_path.exists():
        try:
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            existing_report_samples = existing_report.get("samples", [])
        except (json.JSONDecodeError, OSError):
            existing_report_samples = []
    merged_samples = qc._incremental_merge(existing_report_samples, sample_reports, processed_ids, "sample_id")

    all_scores = [sample["composite_score"] for sample in merged_samples]
    summary = {
        "num_tir_records": len(merged_samples),
        "num_filtered_tir_records": len(filtered_tir),
        "num_sft_records": len(sft_data),
        "num_rl_records": len(rl_data),
        "num_discarded": len(merged_samples) - len(sft_data) - len(rl_data),
        "sft_threshold": args.sft_threshold,
        "rl_threshold": args.rl_threshold,
        "sft_component_floors": {
            "grounding": args.sft_min_grounding,
            "coverage": args.sft_min_coverage,
            "answer_quality": args.sft_min_answer_quality,
        },
        "composite_score_mean": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0,
        "component_means": {
            "grounding": round(sum(sample["grounding_score"] for sample in merged_samples) / len(merged_samples), 2)
            if merged_samples else 0.0,
            "coverage": round(sum(sample["coverage_score"] for sample in merged_samples) / len(merged_samples), 2)
            if merged_samples else 0.0,
            "execution": round(sum(sample["execution_score"] for sample in merged_samples) / len(merged_samples), 2)
            if merged_samples else 0.0,
            "efficiency": round(sum(sample["efficiency_score"] for sample in merged_samples) / len(merged_samples), 2)
            if merged_samples else 0.0,
            "answer_quality": round(
                sum(sample["answer_quality_score"] for sample in merged_samples) / len(merged_samples), 2
            ) if merged_samples else 0.0,
        },
        "split_distribution": qc.summarize_counts(sample["split"] for sample in merged_samples),
        "num_gate_pass": sum(1 for sample in merged_samples if sample["gate_pass"]),
        "num_split_gate_pass": sum(1 for sample in merged_samples if sample.get("split_gate_pass")),
        "num_manifest_available": sum(1 for sample in merged_samples if sample["manifest_available"]),
        "warning_distribution": qc.summarize_reason_counts(
            [{"hard_fail_reasons": sample["warnings"]} for sample in merged_samples]
        ),
        "gate_fail_distribution": qc.summarize_reason_counts(
            [{"hard_fail_reasons": sample["gate_fail_reasons"]} for sample in merged_samples]
        ),
        "difficulty_distribution": qc.summarize_counts(sample["difficulty_category"] for sample in merged_samples),
        "num_samples_with_reference": sum(1 for sample in merged_samples if sample.get("has_reference")),
        "num_samples_with_dr": sum(1 for sample in merged_samples if sample.get("has_dr")),
        "num_samples_classified": sum(1 for sample in merged_samples if sample.get("difficulty_category") is not None),
        "tool_frequency_threshold": tool_frequency_threshold,
        "tool_frequency_threshold_mode": threshold_mode,
        "v2_weights": V2_WEIGHTS,
        "has_dr_file": bool(args.dr_file),
        "has_reference_file": bool(args.reference_file),
    }

    report = {
        "config": {
            "tir_file": args.tir_file,
            "dr_file": args.dr_file,
            "reference_file": args.reference_file,
            "sft_threshold": args.sft_threshold,
            "rl_threshold": args.rl_threshold,
            "tool_frequency_threshold": tool_frequency_threshold,
            "tool_frequency_threshold_mode": threshold_mode,
            "manifest_max_files": args.manifest_max_files,
            "manifest_top_extensions": args.manifest_top_extensions,
            "manifest_top_subtrees": args.manifest_top_subtrees,
            "manifest_table_limit": args.manifest_table_limit,
            "v2_weights": V2_WEIGHTS,
            "trajectory_keys": args.trajectory_keys,
            "prediction_keys": args.prediction_keys,
            "reference_keys": args.reference_keys,
            "id_keys": args.id_keys,
        },
        "summary": summary,
        "samples": merged_samples,
    }

    qc.dump_json(report_path, report)
    qc.dump_json(filtered_tir_path, filtered_tir)
    qc.dump_json(sft_path, sft_data)
    qc.dump_json(rl_path, rl_data)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved report to {report_path}")
    print(f"Saved filtered TIR data ({len(filtered_tir)} records) to {filtered_tir_path}")
    print(f"Saved SFT data ({len(sft_data)} records) to {sft_path}")
    print(f"Saved RL data ({len(rl_data)} records) to {rl_path}")


if __name__ == "__main__":
    main()
