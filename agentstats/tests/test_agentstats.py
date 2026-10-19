import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import agentstats as ast
from agentstats.simulate import SimConfig, simulate


def small(**kw):
    base = dict(n_tasks=200, true_scores=(0.5, 0.5), sd_interaction=0.0, seed=1)
    base.update(kw)
    return SimConfig(**base)


def test_load_validates():
    with pytest.raises(ValueError):
        ast.load(pd.DataFrame({"agent": ["a"], "task": ["t"], "run": [0]}))
    with pytest.raises(ValueError):
        ast.load(pd.DataFrame({"agent": ["a"], "task": ["t"], "run": [0], "success": [2.0]}))
    with pytest.raises(ValueError):
        ast.load(pd.DataFrame({"agent": ["a", "a"], "task": ["t", "t"], "run": [0, 0], "success": [1, 0]}))


def test_holm():
    adj = ast.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_exact_run_floor():  # utility for permutation reference
    assert ast.min_runs_for_exact_run_test(0.05) == 4
    assert ast.min_runs_for_exact_run_test(0.01) == 5


def _rejection_rate(cfg_kw, n_sims, k, **cmp_kw):
    rej = 0
    for s in range(n_sims):
        df = simulate(small(seed=100 + s, **cfg_kw), k, run_seed=10_000 + s)
        rej += ast.compare(df, "A01", "A02", n_perm=400, seed=s, **cmp_kw).p_value < 0.05
    return rej / n_sims


def test_type_one_error_sharp_null():
    rate = _rejection_rate({}, 150, 3)
    assert rate <= 0.10, rate


def test_type_one_error_weak_null_studentized():
    # same benchmark score, different per-task profiles
    rate = _rejection_rate({"sd_interaction": 1.0}, 150, 3)
    assert rate <= 0.10, rate


def test_correlated_noise_breaks_task_level_but_not_run_level():
    kw = {"sd_run_shock": 0.5}
    task_rate = _rejection_rate(kw, 80, 6)
    run_rate = _rejection_rate(kw, 80, 6, level="run")
    assert run_rate <= 0.125, run_rate
    assert task_rate > run_rate + 0.15, (task_rate, run_rate)
    df = simulate(small(seed=3, sd_run_shock=0.5), 8)
    assert ast.design_effect(df)["deff"].median() > 2


def test_power_formula_roughly_calibrated():
    # pick a true gap, compute k from a big pilot, check empirical power ~0.8
    cfg = SimConfig(n_tasks=300, true_scores=(0.52, 0.50), sd_interaction=0.8, scale=1.0, seed=7)
    pilot = simulate(cfg, 20, run_seed=1)
    pl = ast.plan_pair(pilot, "A01", "A02", 0.02, method="task")
    k = int(pl.k_min)
    assert 3 <= k <= 200
    hits = 0
    n = 60
    for s in range(n):
        df = simulate(cfg, k, run_seed=500 + s)
        hits += ast.compare(df, "A01", "A02", n_perm=300, seed=s).p_value < 0.05
    assert 0.55 <= hits / n <= 0.97, (k, hits / n)


def test_generic_planner_monotone():
    k1 = ast.k_min_generic(500, 0.01, 0.07)
    k2 = ast.k_min_generic(500, 0.02, 0.07)
    assert k1 > k2 >= 3
    mdg = ast.min_detectable_gap(500, int(k2), 0.07)
    assert mdg <= 0.02 + 1e-9


def test_population_unattainable():
    k = ast.k_min_from_components(1e-4, 0.01, sigma2_delta=0.5, n_tasks=100, inference="population")
    assert math.isinf(k)


def test_scores_and_ranks():
    cfg = SimConfig(n_tasks=300, true_scores=(0.8, 0.6, 0.4), seed=2)
    df = simulate(cfg, 4)
    sc = ast.agent_scores(df)
    assert list(sc.agent) == ["A01", "A02", "A03"]
    rd = ast.rank_distribution(df, n_boot=500)
    assert rd.loc[0, "p_top1"] > 0.99
    tr = ast.tiers(df, n_perm=500)
    assert tr.tier.tolist() == [1, 2, 3]
    fs = ast.flip_statistics(df, n_draws=200)
    assert fs["kendall_tau_mean"] > 0.95


def test_decompose_detects_api_factor():
    cfg = SimConfig(n_tasks=400, true_scores=(0.5, 0.5, 0.5, 0.5), api_dates=("d0", "d1", "d2", "d3"),
                    sd_api=0.6, seed=4)
    df = simulate(cfg, 12)
    dec = ast.decompose(df, ["api_date"])
    assert (dec["sd_api_date"] > dec["sd_seed"]).mean() >= 0.5


def test_audit():
    out = ast.audit_leaderboard({"x": 70.0, "y": 69.5, "z": 60.0}, 1.5)
    assert out.within_noise.tolist() == [True, False]


def test_strata_and_continuous_scores():
    cfg = SimConfig(n_tasks=100, true_scores=(0.6, 0.6), api_dates=("d0", "d1"), seed=5)
    df = simulate(cfg, 4)
    rng = np.random.default_rng(0)
    df["success"] = np.clip(df["success"] * 0.8 + rng.random(len(df)) * 0.2, 0, 1)
    res = ast.compare(df, "A01", "A02", strata=["api_date"], n_perm=300)
    assert 0 <= res.p_value <= 1 and "strata" in res.test


def test_report_and_cli(tmp_path):
    cfg = SimConfig(n_tasks=100, true_scores=(0.6, 0.55, 0.5), api_dates=("d0", "d1"), seed=6)
    df = simulate(cfg, 6)
    md = ast.report(df, n_perm=300)
    assert "Statistical tiers" in md and "Protocol compliance" in md
    path = tmp_path / "runs.csv"
    df.to_csv(path, index=False)
    out = subprocess.run([sys.executable, "-m", "agentstats.cli", "plan", str(path), "--delta", "0.05"],
                         capture_output=True, text=True, check=True).stdout
    assert "Run every agent at least" in out
    sc = tmp_path / "lb.json"
    sc.write_text(json.dumps({"a": 50, "b": 49}))
    out = subprocess.run([sys.executable, "-m", "agentstats.cli", "audit", str(sc), "--sd", "1.0"],
                         capture_output=True, text=True, check=True).stdout
    assert "within_noise" in out


def test_many_pairs_resolve_large_gaps():
    # 10 agents, 9pp apart: Holm over 45 pairs must still separate them all
    cfg = SimConfig(n_tasks=300, true_scores=tuple(0.9 - 0.09 * i for i in range(10)), seed=8)
    df = simulate(cfg, 5)
    for level in ("task", "run"):
        tr = ast.tiers(df, n_perm=500, level=level)
        assert tr.tier.nunique() == 10, (level, tr)


def test_swebench_loader_and_rerun(tmp_path):
    root = tmp_path / "evaluation" / "verified"
    rng = np.random.default_rng(0)
    ids = [f"repo__x-{i}" for i in range(50)]
    for j, name in enumerate(["20260101_a", "20260102_b", "20260103_broken"]):
        d = root / name; (d / "results").mkdir(parents=True)
        res = [] if "broken" in name else [i for i in ids if rng.random() < 0.6 - 0.1 * j]
        (d / "results" / "results.json").write_text(json.dumps({"resolved": res}))
        (d / "per_instance_details.json").write_text(json.dumps({i: {"resolved": i in res} for i in ids}))
        (d / "metadata.yaml").write_text(f"info:\n  name: {name}\ntags:\n  system:\n    attempts: '1'\n  agent: x\n")
    runs, meta = ast.load_swebench_submissions(str(tmp_path))
    assert len(meta) == 2 and runs.task.nunique() == 50
    st = ast.rerun_stability({"a": 70, "b": 70, "c": 50}, 1.2)
    assert 0.3 < st["p_top1_changes"] < 0.7
    assert ast.rerun_stability({"a": 90, "b": 50}, 1.0, method="eb")["p_top1_changes"] < 0.01
    out = subprocess.run([sys.executable, "-m", "agentstats.cli", "swebench-audit", str(tmp_path), "--top", "2"],
                         capture_output=True, text=True, check=True).stdout
    assert "Re-run (plug-in)" in out
