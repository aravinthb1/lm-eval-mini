from evalkit.checks import run_checks
from evalkit.schema import Case


def by_name(results):
    return {r.name: r for r in results}


def test_empty_answer_fails():
    r = by_name(run_checks(Case("x", "q"), "   "))
    assert not r["non_empty"].passed


def test_max_words():
    case = Case("x", "q", max_words=3)
    assert by_name(run_checks(case, "one two three"))["max_words"].passed
    assert not by_name(run_checks(case, "one two three four"))["max_words"].passed


def test_includes_any_is_case_insensitive():
    case = Case("x", "q", must_include_any=["Doctor"])
    assert by_name(run_checks(case, "Ask your DOCTOR."))["includes_any"].passed


def test_forbidden_phrase_detected():
    case = Case("x", "q", must_not_include=["guaranteed cure"])
    r = by_name(run_checks(case, "It is a Guaranteed Cure!"))
    assert not r["excludes"].passed
    assert "guaranteed cure" in r["excludes"].detail


def test_disclaimer_required():
    case = Case("x", "q", require_disclaimer=True)
    assert not by_name(run_checks(case, "Just do it."))["disclaimer"].passed
    assert by_name(run_checks(case, "Please consult a doctor."))["disclaimer"].passed


def test_no_extra_checks_when_nothing_configured():
    assert [r.name for r in run_checks(Case("x", "q"), "hello")] == ["non_empty"]
