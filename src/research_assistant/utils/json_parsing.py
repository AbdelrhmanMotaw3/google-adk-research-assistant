from __future__ import annotations

import ast
import json
import re
from typing import Any


def parse_jsonish(text: str) -> Any:
    """
    Parse model output that may include code fences or extra text.

    Returns dict/list/primitive on success; raises JSONDecodeError on failure.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise json.JSONDecodeError("Empty response", cleaned, 0)

    for candidate in _candidate_strings(cleaned):
        candidate = _sanitize(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return _literal_eval_fallback(candidate)
            except Exception:
                continue

    raise json.JSONDecodeError("Unable to parse JSON response", cleaned, 0)


def _candidate_strings(text: str) -> list[str]:
    candidates = [text]

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            fenced = "\n".join(lines[1:-1]).strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].lstrip()
            candidates.append(fenced)

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1].strip())

    return list(dict.fromkeys(c for c in candidates if c))


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _sanitize(s: str) -> str:
    s = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s.strip()


def _literal_eval_fallback(candidate: str) -> Any:
    py = (
        candidate.replace(": true", ": True")
        .replace(": false", ": False")
        .replace(": null", ": None")
        .replace(":true", ":True")
        .replace(":false", ":False")
        .replace(":null", ":None")
    )
    return ast.literal_eval(py)

