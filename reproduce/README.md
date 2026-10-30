# Reproducing the paper "Within the Noise"

Install first, from the repo root: `pip install -e ".[swebench]" matplotlib`

## Section 5: SWE-bench Verified leaderboard audit

```bash
git clone --depth 1 https://github.com/SWE-bench/experiments
agentstats swebench-audit experiments --pass1 --top 20      # headline numbers, ~30 s

# full analysis, figure and sensitivity table
cd swebench_audit
python build_swe.py ../experiments     # -> swe_meta.csv, swe_runs.csv
python audit_swe.py                    # -> swe_audit.json
python sensitivity.py
python fig_swe.py                      # -> fig_swe_audit.pdf
```

The leaderboard changes over time. The paper's numbers are from the snapshot taken on 2026-10-19.

## Section 7: repeated-run validation

`kth/kth_runs.csv` holds per-task outcomes for 12 configurations x 10 runs, extracted from the
[ASSERT-KTH/agentic-evals-artifacts](https://huggingface.co/datasets/ASSERT-KTH/agentic-evals-artifacts)
dataset (Bjarnason, Silva and Monperrus, "On Randomness in Agentic Evals", 2026; CC-BY-4.0).
Please cite their work if you use it.

```bash
cd kth && python kth_run.py && python fig_kth.py
```

## Section 6: simulation study

```bash
cd simulation && python run_sim.py     # ~5 min; synthetic data only
```
