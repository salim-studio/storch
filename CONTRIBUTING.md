# Contributing to storch

Thanks for stopping by! storch aims to be the friendliest NumPy-powered bridge from
databases to deep learning. Small, focused PRs are the fastest way to get merged.

## Workflow

1. Fork the repo and create a feature branch (`git checkout -b feat/my-thing`).
2. Install dev extras: `pip install -e ".[dev]"`.
3. Add a test in `tests/` for every feature or fix.
4. Run the suite: `pytest tests/ -v` (must be green).
5. Open a PR using the template — describe the *why*, not just the *what*.

## Style

- NumPy-only in the core package (optional deps live behind extras).
- Public APIs get English docstrings + one example in `examples/` when it helps.
- Keep new ops autograd-correct: add a numeric-gradient check in the test.

## Reporting bugs

Use the bug-report template: minimal repro, expected vs actual, `storch.info()`.
