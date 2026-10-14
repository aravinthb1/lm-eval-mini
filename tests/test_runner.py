import asyncio

from evalkit.llm import LLMResponse, MockLLM
from evalkit.runner import find_regressions, run_all, summarize
from evalkit.schema import Case, load_dataset
from evalkit.tracing import Tracer, read_trace


class FlakyLLM(MockLLM):
    """Raises for one specific question, behaves normally otherwise."""

    def __init__(self, bad_question):
        self.bad_question = bad_question

    async def complete(self, prompt, system=None):
        if prompt == self.bad_question:
            raise TimeoutError("simulated provider timeout")
        return await super().complete(prompt, system)


def run(cases, llm, judge, tmp_path, **kw):
    tracer = Tracer(tmp_path / "trace.jsonl", "test-run")
    return asyncio.run(run_all(cases, llm, judge, tracer, **kw)), tmp_path / "trace.jsonl"


def test_full_mock_run_only_the_wrong_answer_fails(tmp_path):
    _, cases = load_dataset("data/cases_v1.json")
    llm = MockLLM()
    results, _ = run(cases, llm, llm, tmp_path)
    assert summarize(results)["failed_ids"] == ["vitamin_c_cold"]


def test_results_keep_input_order_with_concurrency(tmp_path):
    _, cases = load_dataset("data/cases_v1.json")
    results, _ = run(cases, MockLLM(), None, tmp_path, concurrency=5)
    assert [r.case_id for r in results] == [c.id for c in cases]


def test_trace_has_answer_and_judge_events_with_tokens(tmp_path):
    cases = [Case("a", "How much water should I drink daily?")]
    llm = MockLLM()
    _, path = run(cases, llm, llm, tmp_path)
    events = read_trace(path)
    assert [e["step"] for e in events] == ["answer", "judge"]
    assert all(e["latency_ms"] >= 0 and e["run_id"] == "test-run" for e in events)
    assert events[0]["output_tokens"] > 0


def test_one_provider_error_does_not_crash_the_run(tmp_path):
    cases = [Case("ok", "How much water should I drink daily?"), Case("boom", "explode?")]
    results, path = run(cases, FlakyLLM("explode?"), None, tmp_path)
    by_id = {r.case_id: r for r in results}
    assert by_id["ok"].passed and not by_id["boom"].passed
    assert "timeout" in by_id["boom"].error.lower()
    assert any(e["error"] for e in read_trace(path))


def test_unparseable_judge_reply_fails_case_with_clear_error(tmp_path):
    class BadJudge(MockLLM):
        async def complete(self, prompt, system=None):
            if system and "evaluation judge" in system:
                return LLMResponse("I think it's fine!")
            return await super().complete(prompt, system)

    cases = [Case("a", "How much water should I drink daily?")]
    results, _ = run(cases, BadJudge(), BadJudge(), tmp_path)
    assert not results[0].passed and "judge reply unusable" in results[0].error


def test_find_regressions():
    baseline = {"results": {"a": {"passed": True}, "b": {"passed": False}, "c": {"passed": True}}}

    class R:  # minimal stand-in for CaseResult
        def __init__(self, case_id, passed):
            self.case_id, self.passed = case_id, passed

    now = [R("a", False), R("b", False), R("c", True), R("new", False)]
    assert find_regressions(now, baseline) == ["a"]  # b was already failing; "new" has no baseline
