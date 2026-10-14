"""Dataset schema + validation.

Datasets are versioned JSON files. Validating them up front means a typo in a
test case fails loudly instead of silently weakening an evaluation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_KEYS = {
    "id", "question", "must_include_any", "must_not_include",
    "max_words", "require_disclaimer", "tags",
}


@dataclass
class Case:
    id: str
    question: str
    must_include_any: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    max_words: int | None = None
    require_disclaimer: bool = False
    tags: list[str] = field(default_factory=list)


def _str_list(case_id: str, key: str, value) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"case '{case_id}': '{key}' must be a list of non-empty strings")
    return value


def parse_case(raw: dict) -> Case:
    if not isinstance(raw, dict):
        raise ValueError("each case must be a JSON object")
    case_id = raw.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("each case needs a non-empty string 'id'")
    unknown = set(raw) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"case '{case_id}': unknown keys {sorted(unknown)}")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"case '{case_id}': 'question' must be a non-empty string")

    kwargs: dict = {"id": case_id, "question": question}
    for key in ("must_include_any", "must_not_include", "tags"):
        if key in raw:
            kwargs[key] = _str_list(case_id, key, raw[key])
    if "max_words" in raw:
        mw = raw["max_words"]
        if isinstance(mw, bool) or not isinstance(mw, int) or mw <= 0:
            raise ValueError(f"case '{case_id}': 'max_words' must be a positive integer")
        kwargs["max_words"] = mw
    if "require_disclaimer" in raw:
        if not isinstance(raw["require_disclaimer"], bool):
            raise ValueError(f"case '{case_id}': 'require_disclaimer' must be true/false")
        kwargs["require_disclaimer"] = raw["require_disclaimer"]
    return Case(**kwargs)


def validate_dataset(data: dict) -> tuple[str, list[Case]]:
    if not isinstance(data, dict):
        raise ValueError("dataset must be a JSON object")
    version = data.get("dataset_version")
    if not isinstance(version, str) or not version:
        raise ValueError("dataset needs a 'dataset_version' string, e.g. '1.0.0'")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("dataset needs a non-empty 'cases' list")
    cases = [parse_case(c) for c in raw_cases]
    ids = [c.id for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate case ids: {dupes}")
    return version, cases


def load_dataset(path: str | Path) -> tuple[str, list[Case]]:
    with open(path, encoding="utf-8") as f:
        return validate_dataset(json.load(f))
