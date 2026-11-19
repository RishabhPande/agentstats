"""Flip-rate output equivalence and temporary-allocation regression guards."""
import gc
import tracemalloc
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from agentstats.rank import flip_statistics


def original_statistics(a, b):
    """Small-input oracle retaining the original full pairwise broadcasts."""
    taus = np.array([stats.kendalltau(x, y).statistic for x, y in zip(a, b)])
    iu = np.triu_indices(a.shape[1], 1)
    sa = np.sign(a[:, :, None] - a[:, None, :])[:, iu[0], iu[1]]
    sb = np.sign(b[:, :, None] - b[:, None, :])[:, iu[0], iu[1]]
    return {'kendall_tau_mean': float(np.nanmean(taus)),
            'kendall_tau_q05': float(np.nanquantile(taus, .05)),
            'p_top1_changes': float(np.mean(a.argmax(1) != b.argmax(1))),
            'pair_flip_rate': float(np.mean(sa * sb < 0))}


@pytest.mark.parametrize('seed', range(20))
@pytest.mark.parametrize('discrete', [False, True])
def test_flip_statistics_matches_broadcast_oracle(seed, discrete):
    rng = np.random.default_rng(seed)
    shape = (1 + seed * 3, 2 + seed % 9)
    if discrete:
        a, b = [rng.integers(0, 4, size=shape) / 3 for _ in range(2)]
        # Ensure at least one defined tau, while keeping ties in other rows.
        a[0] = np.arange(shape[1])
        b[0] = a[0][::-1]
    else:
        a, b = [rng.uniform(size=shape) for _ in range(2)]
    with patch('agentstats.rank.single_run_leaderboards', side_effect=[a, b]):
        actual = flip_statistics(None, n_draws=shape[0], seed=seed)
    assert actual == original_statistics(a, b)


def test_ties_are_not_counted_as_pair_flips():
    a = np.array([[1., 1., 0.], [0., 1., 1.]])
    b = np.array([[0., 1., 1.], [1., 1., 0.]])
    with patch('agentstats.rank.single_run_leaderboards', side_effect=[a, b]):
        assert flip_statistics(None, n_draws=2)['pair_flip_rate'] == 1 / 3


def test_pair_work_does_not_allocate_draw_by_agent_by_agent_arrays():
    rng = np.random.default_rng(107)
    a, b = [rng.uniform(size=(96, 180)) for _ in range(2)]
    # Generate draws before tracking: this guard targets temporary pair arrays,
    # not the intentionally retained draw-by-agent score tables.
    with patch('agentstats.rank.single_run_leaderboards', side_effect=[a, b]):
        gc.collect()
        tracemalloc.start()
        try:
            result = flip_statistics(None, n_draws=96)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert 0 <= result['pair_flip_rate'] <= 1
    # A structural allocation guard, not a runtime benchmark. The previous
    # full broadcasts retain tens of MB for this synthetic example.
    assert peak < 5_000_000
