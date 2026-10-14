"""CLI: python run_eval.py --provider mock   |   python run_eval.py --provider anthropic"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from evalkit.llm import AnthropicLLM, MockLLM
from evalkit.runner import find_regressions, run_all, save_results, summarize
from evalkit.schema import load_dataset
from evalkit.tracing import Tracer


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate an LLM on a versioned dataset.")
    p.add_argument("--provider", choices=["mock", "anthropic"], default="mock")
    p.add_argument("--dataset", default="data/cases_v1.json")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--min-score", type=int, default=4, help="judge score needed to pass (1-5)")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM-as-a-judge step")
    p.add_argument("--save", default="results/latest.json")
    p.add_argument("--baseline", help="earlier results file; exit non-zero on regressions")
    args = p.parse_args()

    version, cases = load_dataset(args.dataset)
    llm = MockLLM() if args.provider == "mock" else AnthropicLLM()
    judge_llm = None if args.no_judge else llm

    run_id = uuid.uuid4().hex[:8]
    tracer = Tracer(Path("traces") / f"run-{run_id}.jsonl", run_id)
    results = asyncio.run(run_all(cases, llm, judge_llm, tracer, args.concurrency, args.min_score))

    print(f"dataset v{version} | provider={args.provider} | run_id={run_id}\n")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        score = f"judge={r.judge_score}" if r.judge_score is not None else "judge=-"
        extra = f"  <- {r.error or r.failed_checks}" if not r.passed else ""
        print(f"[{status}] {r.case_id:<24} {score}{extra}")
    s = summarize(results)
    print(f"\n{s['passed']}/{s['total']} passed | avg judge score: {s['avg_judge_score']}")
    print(f"trace: traces/run-{run_id}.jsonl")

    regressions: list[str] = []
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        regressions = find_regressions(results, baseline)
        print(f"regressions vs baseline: {regressions or 'none'}")

    save_results(results, version, args.save)
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
