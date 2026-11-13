"""Invalid planner options must not produce plausible minimum-run counts."""
import math
import numpy as np
import pytest
from scipy import stats
from agentstats.power import k_min_from_components

@pytest.mark.parametrize('field,value', [
    ('inference', 'populaton'), ('inference', ''),
    ('w', -0.1), ('w', float('nan')), ('w', float('inf')), ('w', True),
    ('delta', 0), ('delta', -0.1), ('delta', float('nan')), ('delta', float('inf')), ('delta', True),
    ('sigma2_delta', -1), ('sigma2_delta', float('nan')), ('sigma2_delta', float('inf')),
    ('alpha', 0), ('alpha', 1), ('alpha', -0.1), ('alpha', float('nan')), ('alpha', True),
    ('power', 0), ('power', 1), ('power', float('inf')), ('power', float('nan')),
    ('k_floor', 0), ('k_floor', -1), ('k_floor', 1.5), ('k_floor', True),
])
def test_invalid_components_rejected(field, value):
    options = dict(w=.001, delta=.05)
    options[field] = value
    with pytest.raises(ValueError, match=field):
        k_min_from_components(**options)

@pytest.mark.parametrize('n_tasks', [None, 0, -1, True, 1.5, float('nan')])
def test_population_requires_positive_integer_tasks(n_tasks):
    with pytest.raises(ValueError, match='n_tasks'):
        k_min_from_components(.001, .05, inference='population', n_tasks=n_tasks)

@pytest.mark.parametrize('inference', ['benchmark', 'population'])
def test_valid_formula_and_zero_variance_preserved(inference):
    options = dict(sigma2_delta=.0001, n_tasks=np.int64(100), inference=inference)
    z2 = (stats.norm.ppf(1-.05/2) + stats.norm.ppf(.8))**2
    denominator = .05**2 - (z2*.0001/100 if inference == 'population' else 0)
    assert k_min_from_components(.001, .05, **options) == float(max(3, math.ceil(z2*.001/denominator)))
    assert k_min_from_components(0, .05, k_floor=np.int64(4), **options) == 4.


def test_population_unattainable_stays_infinite():
    assert math.isinf(k_min_from_components(.001, .01, sigma2_delta=.5, n_tasks=100, inference='population'))
