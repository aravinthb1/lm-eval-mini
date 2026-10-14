"""Run every case concurrently, apply checks + judge, and compare against a baseline."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .checks import CheckResult, run_checks
from .judge import JudgeParseError, judge_answer
from .llm import ANSWER_SYSTEM
from .schema import Case
from .tracing import Tracer


@dataclass
class CaseResult:
    case_id: str
    answer: str = ""
    checks: list[CheckResult] = field(default_factory=list)
    judge_score: int | None = None
    judge_reason: str = ""
    error: str | None = None
    passed: bool = False

    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]


async def evaluate_case(case: Case, llm, judge_llm, tracer: Tracer, min_score: int) -> CaseResult:
    result = CaseResult(case_id=case.id)

    # Step 1: get the answer (errors are recorded, never allowed to crash the whole run)
    start = time.perf_counter()
    try:
        resp = await llm.complete(case.question, system=ANSWER_SYSTEM)
    except Exception as e:  # noqa: BLE001 - we want to record any provider failure
        tracer.record(case.id, "answer", (time.perf_counter() - start) * 1000, error=repr(e))
        result.error = f"answer call failed: {e!r}"
        return result
    tracer.record(case.id, "answer", (time.perf_counter() - start) * 1000,
                  resp.input_tokens, resp.output_tokens, resp.tool_calls)
    result.answer = resp.text

    # Step 2: deterministic checks
    result.checks = run_checks(case, resp.text)
    ok = all(c.passed for c in result.checks)

    # Step 3: LLM-as-a-judge (optional)
    if judge_llm is not None:
        start = time.perf_counter()
        try:
            judged, jresp = await judge_answer(judge_llm, case, resp.text)
            tracer.record(case.id, "judge", (time.perf_counter() - start) * 1000,
                          jresp.input_tokens, jresp.output_tokens, jresp.tool_calls)
            result.judge_score, result.judge_reason = judged.score, judged.reason
            ok = ok and judged.score >= min_score
        except JudgeParseError as e:
            tracer.record(case.id, "judge", (time.perf_counter() - start) * 1000, error=str(e))
            result.error = f"judge reply unusable: {e}"
            ok = False
        except Exception as e:  # noqa: BLE001
            tracer.record(case.id, "judge", (time.perf_counter() - start) * 1000, error=repr(e))
            result.error = f"judge call failed: {e!r}"
            ok = False

    result.passed = ok
    return result


async def run_all(cases: list[Case], llm, judge_llm, tracer: Tracer,
                  concurrency: int = 3, min_score: int = 4) -> list[CaseResult]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(case: Case) -> CaseResult:
        async with sem:
            return await evaluate_case(case, llm, judge_llm, tracer, min_score)

    return list(await asyncio.gather(*(guarded(c) for c in cases)))  # gather keeps input order


def summarize(results: list[CaseResult]) -> dict:
    scores = [r.judge_score for r in results if r.judge_score is not None]
    return {
        "total": len(results),
        "passed": sum(r.passed for r in results),
        "failed_ids": [r.case_id for r in results if not r.passed],
        "avg_judge_score": round(sum(scores) / len(scores), 2) if scores else None,
    }


def save_results(results: list[CaseResult], dataset_version: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset_version": dataset_version,
               "results": {r.case_id: asdict(r) for r in results}}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_regressions(results: list[CaseResult], baseline: dict) -> list[str]:
    """Case ids that passed in the baseline run but fail now."""
    old = baseline.get("results", {})
    return [r.case_id for r in results
            if old.get(r.case_id, {}).get("passed") is True and not r.passed]
