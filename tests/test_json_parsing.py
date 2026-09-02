import pytest

from research_assistant.utils.json_parsing import parse_jsonish


def test_parse_plain_json_object():
    assert parse_jsonish('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_fenced_json():
    text = """```json
{"k": ["v1", "v2"]}
```"""
    assert parse_jsonish(text) == {"k": ["v1", "v2"]}


def test_parse_jsonish_single_quotes_fallback():
    # Not strict JSON, but common model output.
    assert parse_jsonish("{'a': True, 'b': None}") == {"a": True, "b": None}


def test_parse_empty_raises():
    with pytest.raises(Exception):
        parse_jsonish("")

