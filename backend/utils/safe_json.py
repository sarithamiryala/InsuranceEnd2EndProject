# backend/utils/safe_json.py

from __future__ import annotations
import json
import re
import html
from typing import Any, Dict, Optional, Tuple, Iterable

# Fenced code blocks: ```<lang>\n ... \n```  or  ~~~<lang>\n ... \n~~~
FENCE_RE = re.compile(r"(?:```|~~~)\s*([a-zA-Z0-9_-]+)?\s*\n(.*?)(?:```|~~~)", re.DOTALL)

# Basic smart quotes normalization mapping
SMART_QUOTES = {
    "\u201c": '"',  # left double
    "\u201d": '"',  # right double
    "\u2018": "'",  # left single
    "\u2019": "'",  # right single
}

def _normalize_text(s: str) -> str:
    """
    Normalize common nuisances: BOM, HTML entities, smart quotes, stray NBSP.
    """
    if not s:
        return s
    # Strip BOM
    s = s.lstrip("\ufeff")
    # HTML entities (&quot;, &amp;, ...)
    s = html.unescape(s)
    # Replace smart quotes
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    # Normalize NBSP
    s = s.replace("\u00a0", " ")
    return s

def _extract_fenced_blocks(s: str) -> Iterable[Tuple[Optional[str], str]]:
    """
    Yield (language, content) for each fenced code block in order of appearance.
    """
    for m in FENCE_RE.finditer(s):
        lang = (m.group(1) or "").strip().lower() or None
        content = m.group(2).strip()
        yield (lang, content)

def _extract_first_balanced_json_object(s: str) -> Optional[str]:
    """
    Extract the first top-level balanced JSON object {...} honoring string/escape rules.
    """
    start_idx = None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    return s[start_idx:i+1]
    return None

def _extract_first_balanced_json_array(s: str) -> Optional[str]:
    """
    Extract the first top-level balanced JSON array [...] honoring string/escape rules.
    """
    start_idx = None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == '[':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == ']':
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    return s[start_idx:i+1]
    return None

def _try_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

def _is_expected_type(obj: Any, expect: str) -> bool:
    if expect == "any":
        return True
    if expect == "object":
        return isinstance(obj, dict)
    if expect == "array":
        return isinstance(obj, list)
    return False

def safe_json_parse(
    result: str,
    fallback: Any,
    *,
    expect: str = "object",    # "object" | "array" | "any"
    max_scan_chars: int = 200_000
) -> Any:
    """
    Safely extract JSON from a string and return as Python data.

    Strategy:
    1) Normalize nuisances (BOM, HTML entities, smart quotes).
    2) Direct json.loads if the whole string is JSON.
    3) Scan all fenced code blocks:
       - Prefer ones labeled 'json'
       - Then try unlabeled blocks
    4) Extract first balanced object {...}; if not found and expect != "object", try array [...]
    5) Return fallback if all attempts fail or type doesn't match `expect`.

    Params:
      - result: Raw LLM output (string).
      - fallback: Value to return on failure (type should match your expectation).
      - expect: Enforce JSON top-level type. One of "object" (default), "array", "any".
      - max_scan_chars: Hard cap on the amount of text scanned for performance safety.
    """
    if not isinstance(result, str) or not result.strip():
        print("Empty or non-string LLM response. Using fallback:", fallback)
        return fallback

    s = _normalize_text(result.strip())
    if len(s) > max_scan_chars:
        s = s[:max_scan_chars]

    # 1) Direct attempt on the whole payload
    obj = _try_json_loads(s)
    if obj is not None and _is_expected_type(obj, expect):
        return obj

    # 2) Fenced code blocks (scan ALL; prefer json-labeled ones)
    blocks = list(_extract_fenced_blocks(s))
    # First pass: lang == 'json'
    for lang, content in blocks:
        if lang == "json":
            content_norm = _normalize_text(content)
            obj = _try_json_loads(content_norm)
            if obj is not None and _is_expected_type(obj, expect):
                return obj
    # Second pass: any fenced block
    for lang, content in blocks:
        content_norm = _normalize_text(content)
        obj = _try_json_loads(content_norm)
        if obj is not None and _is_expected_type(obj, expect):
            return obj
        # Try balanced within the fenced content
        if expect in ("object", "any"):
            maybe_obj = _extract_first_balanced_json_object(content_norm)
            if maybe_obj:
                obj = _try_json_loads(maybe_obj)
                if obj is not None and _is_expected_type(obj, expect):
                    return obj
        if expect in ("array", "any"):
            maybe_arr = _extract_first_balanced_json_array(content_norm)
            if maybe_arr:
                obj = _try_json_loads(maybe_arr)
                if obj is not None and _is_expected_type(obj, expect):
                    return obj

    # 3) Balanced extraction on the whole string
    if expect in ("object", "any"):
        maybe_obj = _extract_first_balanced_json_object(s)
        if maybe_obj:
            obj = _try_json_loads(maybe_obj)
            if obj is not None and _is_expected_type(obj, expect):
                return obj

    if expect in ("array", "any"):
        maybe_arr = _extract_first_balanced_json_array(s)
        if maybe_arr:
            obj = _try_json_loads(maybe_arr)
            if obj is not None and _is_expected_type(obj, expect):
                return obj

    # 4) Fallback
    print("Failed to parse JSON from LLM response. Using fallback. Snippet:", repr(s[:300]))
    return fallback