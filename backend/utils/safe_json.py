import json
import re
from typing import Any, Dict, Optional, Tuple

FENCE_RE = re.compile(
    r"(?:```|~~~)\s*([a-zA-Z0-9_-]+)?\s*\n(.*?)(?:```|~~~)",
    re.DOTALL
)

def _extract_from_fence(s: str) -> Optional[str]:
    """
    Return the content of the FIRST fenced code block if any.
    Strips language tag (e.g., ```json).
    """
    m = FENCE_RE.search(s)
    if not m:
        return None
    content = m.group(2)
    return content.strip()

def _extract_first_balanced_json(s: str) -> Optional[str]:
    """
    Extract the first top-level balanced JSON object from a string
    using a simple stack. Safer than greedy regex like \{.*\}.
    """
    start_idx = None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(s):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
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

def _try_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def safe_json_parse(result: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely extract JSON from a string and return as dict.
    Strategy:
    1) If result already looks like a JSON object, try loads directly.
    2) Try fenced block extraction (``` / ~~~).
    3) Try balanced-brace extraction.
    4) Return fallback if all fail.
    """
    if not result or not isinstance(result, str):
        print("Empty or non-string LLM response. Using fallback:", fallback)
        return fallback

    stripped = result.strip()

    # 1) Direct JSON attempt
    direct = _try_json_loads(stripped)
    if direct is not None:
        return direct

    # 2) Fenced code block (```json ... ```)
    fenced = _extract_from_fence(stripped)
    if fenced:
        # a) direct try
        obj = _try_json_loads(fenced)
        if obj is not None:
            return obj
        # b) sometimes the fenced block includes leading 'json\n{...}'
        fenced_no_lang_line = re.sub(r"^\s*(json|javascript)\s*\n", "", fenced, flags=re.IGNORECASE)
        obj = _try_json_loads(fenced_no_lang_line)
        if obj is not None:
            return obj
        # c) try balanced inside fenced
        balanced = _extract_first_balanced_json(fenced)
        if balanced:
            obj = _try_json_loads(balanced)
            if obj is not None:
                return obj

    # 3) Balanced-brace extraction on the whole string
    balanced = _extract_first_balanced_json(stripped)
    if balanced:
        obj = _try_json_loads(balanced)
        if obj is not None:
            return obj

    # 4) Fallback
    print("Failed to parse JSON from LLM response. Using fallback.", "Snippet:", repr(stripped[:300]))
    return fallback
