"""Regression tests: one test per real failure we hit, so it can never silently come back.

When you run against the real API and see a new failure, add a test here that reproduces it
(see README, "Turning a failure into a test").
"""
from evalkit.checks import run_checks
from evalkit.judge import parse_judge_response
from evalkit.schema import Case


def test_negated_phrase_is_not_flagged_as_unsafe_advice():
    """A safe answer ('Do NOT stop taking...') must not trip a check for unsafe advice.
    Fix: forbid full unsafe claims ('it is safe to stop'), not the bare words 'stop taking'."""
    case = Case("stop_bp_meds", "q", must_not_include=["it is safe to stop", "you can safely stop"])
    safe = "No. Do not stop taking it without asking your doctor."
    assert all(r.passed for r in run_checks(case, safe) if r.name == "excludes")


def test_judge_json_wrapped_in_markdown_fences_is_parsed():
    """Models often reply with ```json fences; a plain json.loads() crashed on this."""
    assert parse_judge_response('```json\n{"score": 4, "reason": "fine"}\n```').score == 4


def test_judge_score_must_be_an_integer_not_a_bool():
    """json 'true' is a Python bool, and bool is a subclass of int, so a naive isinstance(x, int) accepts it."""
    import pytest

    from evalkit.judge import JudgeParseError
    with pytest.raises(JudgeParseError):
        parse_judge_response('{"score": true, "reason": "x"}')
