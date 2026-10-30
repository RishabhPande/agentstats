# Contributing

Contributions are welcome. agentstats keeps its dependency list short
(numpy, scipy, pandas, tabulate) and its claims honest.

## What fits here

* Loaders for other agent benchmarks
* Additional variance, significance or minimum-runs methods
* Worked examples on public run data
* Bug fixes, clearer docs, faster code

## Ground rules

1. **No new dependencies without an issue first.** The current set is
   declared in pyproject.toml.
2. **Tests for behavior changes.** `python3 -m pytest tests/ -q`
   must stay green.
3. **Deterministic.** Seed everything.
4. **Honest numbers.** If a method does not change the conclusion on
   the bundled example, say so in the PR.
5. **Real data needs a license.** New bundled examples must come from
   public datasets with a stated license (the current one is
   ASSERT-KTH, CC-BY-4.0).

## How to contribute

1. Fork the repo and create a branch from `main`.
2. Make your change with tests.
3. Run the full suite: `python3 -m pytest tests/ -q`.
4. Open a pull request describing what changed and the before/after
   numbers.
