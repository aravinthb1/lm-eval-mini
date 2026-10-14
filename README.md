# llm-eval-mini

A small, readable framework for evaluating LLM answers: **deterministic checks + LLM-as-a-judge**, **tracing** of every call (latency, tokens, tool calls), a **versioned dataset with a schema**, and **regression tests** built from real failures.

The sample dataset is general health Q&A (sleep, medication safety, misinformation), because reliability and safety matter most in that domain.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                      # 35 tests, fully offline
python run_eval.py --provider mock                       # offline demo, no API key
export ANTHROPIC_API_KEY=sk-ant-...                      # Windows: set ANTHROPIC_API_KEY=...
python run_eval.py --provider anthropic                  # real model (default model: claude-haiku-5-5; override with EVAL_MODEL)
```

Compare against an earlier run to catch regressions (exit code 1 if a previously-passing case now fails):

```bash
python run_eval.py --provider anthropic --save results/baseline.json
# ...change the prompt or model...
python run_eval.py --provider anthropic --baseline results/baseline.json
```

## How it works

| File | Job |
|---|---|
| `data/cases_v1.json` | Versioned dataset: question, expected phrases, forbidden phrases, word limit, disclaimer rule, tags |
| `evalkit/schema.py` | Validates the dataset (unknown keys, duplicate ids, bad types) so typos fail loudly |
| `evalkit/checks.py` | Deterministic checks: non-empty, max words, must-include-any, must-not-include, disclaimer |
| `evalkit/judge.py` | LLM-as-a-judge: prompt, plus a strict parser for the judge's JSON reply |
| `evalkit/tracing.py` | One JSONL event per LLM call: run id, step, latency, tokens, tool calls, error |
| `evalkit/runner.py` | Async concurrent runner (semaphore-limited), error isolation, summary, baseline regression detection |
| `evalkit/llm.py` | `AnthropicLLM` (real) and `MockLLM` (deterministic, used by tests) |

A case passes only if **all deterministic checks pass AND the judge score is at least `--min-score`** (default 4/5).

## Turning a failure into a test

1. Run `python run_eval.py --provider anthropic` and open `traces/run-<id>.jsonl` and `results/latest.json`.
2. Find the failing case and decide the root cause: is the model wrong, is the check too strict or too loose, or is the parser breaking?
3. Fix the cause (prompt, check, or code).
4. Add a test to `tests/test_regressions.py` that reproduces the failure, and confirm it fails without the fix.

## Known limitations (be honest about these in interviews)

- Substring checks are crude: they miss paraphrases and can false-positive. They are a fast first filter, not the whole evaluation.
- The judge can be wrong or biased, and using the same model to answer and judge is weak. Use a different, stronger judge model for real work.
- 10 cases is a demo, not a benchmark. Real datasets need far more cases and labelled edge cases.
- There is no cost tracking, retry or backoff yet.
