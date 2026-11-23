"""Invalid scoring, rank and audit inputs must be rejected, not silently computed.

A leaderboard audit that silently treats a missing noise estimate as
"not within noise", or a confidence interval computed with alpha outside
(0, 1), reports the opposite of the evidence. These entry points now fail
fast, the way ``compare`` and ``plan`` already do.
"""
import numpy as np
import pandas as pd
import pytest

import agentstats as ast
from agentstats.simulate import SimConfig, simulate

SCORES = {"a": 55.0, "b": 53.5, "c": 50.0}
BAD_ALPHA = [0, 1, -0.1, 1.5, float("nan"), float("inf"), True, "0.05", None]
BAD_COUNT = [0, -5, 1.5, True, float("nan"), "500", None]
BAD_SD = [0, -1.2, float("nan"), float("inf"), True, "1.2", None]


@pytest.fixture
def runs():
    return simulate(SimConfig(n_tasks=12, true_scores=(.6, .5), seed=4), 3)


@pytest.mark.parametrize("alpha", BAD_ALPHA)
def test_agent_scores_alpha_rejected(runs, alpha):
    with pytest.raises(ValueError, match="alpha must be"):
        ast.agent_scores(runs, alpha=alpha)


@pytest.mark.parametrize("alpha", BAD_ALPHA)
def test_rank_distribution_alpha_rejected(runs, alpha):
    with pytest.raises(ValueError, match="alpha must be"):
        ast.rank_distribution(runs, n_boot=32, alpha=alpha)


@pytest.mark.parametrize("alpha", BAD_ALPHA)
def test_audit_leaderboard_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha must be"):
        ast.audit_leaderboard(SCORES, 1.2, alpha=alpha)


@pytest.mark.parametrize("n_boot", BAD_COUNT)
def test_rank_distribution_n_boot_rejected(runs, n_boot):
    with pytest.raises(ValueError, match="n_boot must be"):
        ast.rank_distribution(runs, n_boot=n_boot)


@pytest.mark.parametrize("n_draws", BAD_COUNT)
def test_single_run_leaderboards_n_draws_rejected(runs, n_draws):
    with pytest.raises(ValueError, match="n_draws must be"):
        ast.single_run_leaderboards(runs, n_draws=n_draws)


@pytest.mark.parametrize("n_draws", BAD_COUNT)
def test_flip_statistics_n_draws_rejected(runs, n_draws):
    with pytest.raises(ValueError, match="n_draws must be"):
        ast.flip_statistics(runs, n_draws=n_draws)


@pytest.mark.parametrize("n_draws", BAD_COUNT)
def test_rerun_stability_n_draws_rejected(n_draws):
    with pytest.raises(ValueError, match="n_draws must be"):
        ast.rerun_stability(SCORES, 1.0, n_draws=n_draws)


@pytest.mark.parametrize("sd", BAD_SD)
def test_audit_leaderboard_sd_rejected(sd):
    with pytest.raises(ValueError, match="single_run_sd must be"):
        ast.audit_leaderboard(SCORES, sd)


@pytest.mark.parametrize("sd", BAD_SD + [{"a": 1.2, "b": 1.1, "c": 1.0}])
def test_rerun_stability_sd_rejected(sd):
    with pytest.raises(ValueError, match="single_run_sd must be"):
        ast.rerun_stability(SCORES, sd)


@pytest.mark.parametrize("sd", [
    {"a": 1.2, "b": 1.1},                       # missing an agent: must not read as "not within noise"
    {"a": 1.2, "b": 0.0, "c": 1.0},             # zero noise is not a real estimate
    {"a": 1.2, "b": float("nan"), "c": 1.0},    # NaN must not propagate into p-values
    {"a": -1.2, "b": 1.1, "c": 1.0},
])
def test_audit_leaderboard_sd_dict_needs_valid_value_per_agent(sd):
    with pytest.raises(ValueError, match="single_run_sd"):
        ast.audit_leaderboard(SCORES, sd)


@pytest.mark.parametrize("scores", [{"a": 55.0}, {}, pd.Series([55.0], index=["a"])])
def test_audit_needs_two_entries(scores):
    with pytest.raises(ValueError, match="at least two"):
        ast.audit_leaderboard(scores, 1.2)
    with pytest.raises(ValueError, match="at least two"):
        ast.rerun_stability(scores, 1.0)


def test_audit_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="finite"):
        ast.audit_leaderboard({"a": 55.0, "b": float("nan")}, 1.2)
    with pytest.raises(ValueError, match="finite"):
        ast.rerun_stability({"a": 55.0, "b": float("inf")}, 1.0)


def test_audit_leaderboard_with_complete_dict_is_finite():
    out = ast.audit_leaderboard(SCORES, {"a": 1.2, "b": 1.1, "c": 1.0})
    assert np.isfinite(out["noise_floor_pp"]).all()
    assert np.isfinite(out["p_value"]).all()
    assert out["within_noise"].tolist() == [True, False]


def test_valid_numpy_scalar_inputs(runs):
    expected = ast.agent_scores(runs, alpha=.05)
    actual = ast.agent_scores(runs, alpha=np.float64(.05))
    pd.testing.assert_frame_equal(expected, actual)

    expected = ast.rank_distribution(runs, n_boot=200, alpha=.05, seed=0)
    actual = ast.rank_distribution(runs, n_boot=np.int64(200), alpha=np.float64(.05), seed=0)
    pd.testing.assert_frame_equal(expected, actual)

    assert (ast.flip_statistics(runs, n_draws=np.int64(100), seed=0)
            == ast.flip_statistics(runs, n_draws=100, seed=0))

    pd.testing.assert_frame_equal(ast.audit_leaderboard(SCORES, np.float64(1.2)),
                                  ast.audit_leaderboard(SCORES, 1.2))
    assert (ast.rerun_stability(SCORES, np.float64(1.0), n_draws=np.int64(200))
            == ast.rerun_stability(SCORES, 1.0, n_draws=200))
