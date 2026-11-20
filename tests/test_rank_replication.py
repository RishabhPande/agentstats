"""Unknown run noise must not look like a certain leaderboard."""
import pandas as pd
import pytest

from agentstats import rank_distribution


def table(runs=1):
    return pd.DataFrame([
        (agent, task, run, float((task + run + offset) % 3 != 0))
        for agent, offset in [('A', 0), ('B', 1)]
        for task in range(4) for run in range(runs)
    ], columns=['agent', 'task', 'run', 'success'])


@pytest.mark.parametrize('level', ['task', 'run'])
def test_one_run_is_not_enough_for_benchmark_ranks(level):
    with pytest.raises(ValueError, match='at least 2'):
        rank_distribution(table(), level=level, n_boot=30)


def test_one_underreplicated_cell_is_rejected():
    df = table(2).drop(index=[0])
    with pytest.raises(ValueError, match='at least 2 runs per'):
        rank_distribution(df, n_boot=30)


def test_population_task_ranks_still_allow_one_run():
    result = rank_distribution(table(), inference='population', n_boot=30)
    assert len(result) == 2
    assert result['p_top1'].sum() == pytest.approx(1.0)


def test_run_inference_requires_replication_for_every_agent():
    df = table(2)
    df = df[~((df.agent == 'B') & (df.run == 1))]
    with pytest.raises(ValueError, match='at least 2 runs per agent'):
        rank_distribution(df, level='run', n_boot=30)


def test_unshared_task_does_not_trigger_replication_guard():
    df = pd.concat([table(2), pd.DataFrame([('A', 99, 0, 1.0)], columns=table().columns)])
    actual = rank_distribution(df, n_boot=30)
    expected = rank_distribution(table(2), n_boot=30)
    pd.testing.assert_frame_equal(actual, expected)


def test_replicated_deterministic_runs_remain_valid():
    df = table(2)
    df['success'] = (df.agent == 'A').astype(float)
    for level in ('task', 'run'):
        result = rank_distribution(df, level=level, n_boot=30).set_index('agent')
        assert result.loc['A', 'p_top1'] == 1.0
        assert result.loc['A', 'rank_lo'] == result.loc['A', 'rank_hi'] == 1
