"""Invalid comparison options must not change methods or emit suspect evidence."""
import numpy as np
import pytest

from agentstats import compare
from agentstats.simulate import SimConfig, simulate


@pytest.fixture
def runs():
    return simulate(SimConfig(n_tasks=12, true_scores=(.6, .5), seed=4), 3)


@pytest.mark.parametrize('field,value', [
    ('run_test', 'welhc'), ('run_test', ''), ('run_test', None),
    ('alpha', 0), ('alpha', 1), ('alpha', -.1), ('alpha', float('nan')),
    ('alpha', float('inf')), ('alpha', True), ('alpha', '0.05'),
    ('n_perm', 0), ('n_perm', -1), ('n_perm', 1.5), ('n_perm', True),
    ('n_perm', float('nan')), ('n_perm', '100'),
    ('studentize', 'False'), ('studentize', 0), ('studentize', None),
])
@pytest.mark.parametrize('mode', ['task', 'run', 'population'])
def test_invalid_options_rejected(runs, field, value, mode):
    options = dict(n_perm=64)
    if mode == 'run':
        options['level'] = 'run'
    elif mode == 'population':
        options['inference'] = 'population'
    options[field] = value
    with pytest.raises(ValueError, match=field):
        compare(runs, 'A01', 'A02', **options)


def test_valid_numpy_scalar_options(runs):
    expected = compare(runs, 'A01', 'A02', n_perm=64, alpha=.05)
    actual = compare(runs, 'A01', 'A02', n_perm=np.int64(64),
                     alpha=np.float64(.05), studentize=np.bool_(True))
    assert actual.to_dict() == expected.to_dict()
