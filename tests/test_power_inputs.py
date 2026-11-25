"""Invalid planner options must not produce plausible minimum-run counts."""
import math
import numpy as np
import pytest
from scipy import stats
from agentstats.power import k_min_from_components, k_min_generic, min_detectable_gap

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


@pytest.mark.parametrize('n_tasks', [0, -1, True, 1.5, float('nan')])
def test_generic_requires_positive_integer_tasks(n_tasks):
    with pytest.raises(ValueError, match='n_tasks'):
        k_min_generic(n_tasks, .05)


@pytest.mark.parametrize('flakiness', [-0.1, float('nan'), float('inf'), True])
def test_generic_requires_valid_flakiness(flakiness):
    with pytest.raises(ValueError, match='flakiness'):
        k_min_generic(500, .05, flakiness)


@pytest.mark.parametrize('field,value', [
    ('n_tasks', 0), ('n_tasks', True), ('n_tasks', 1.5),
    ('k', 0), ('k', -2), ('k', True), ('k', 2.5),
    ('flakiness', -0.1), ('flakiness', float('nan')), ('flakiness', float('inf')),
    ('alpha', 0), ('alpha', 1), ('alpha', float('nan')),
    ('power', 0), ('power', 1), ('power', float('inf')),
])
def test_min_detectable_gap_rejects_invalid_inputs(field, value):
    options = dict(n_tasks=500, k=5, flakiness=.07)
    options[field] = value
    with pytest.raises(ValueError, match=field):
        min_detectable_gap(**options)


def test_valid_generic_and_gap_answers_preserved():
    z = stats.norm.ppf(1-.05/2) + stats.norm.ppf(.8)
    assert k_min_generic(500, .05, .07) == float(max(3, math.ceil(z**2 * 2*.07/500 / .05**2)))
    assert k_min_generic(np.int64(500), np.float64(.05), np.float64(.07)) == k_min_generic(500, .05, .07)
    assert min_detectable_gap(500, 5, .07) == float(z * math.sqrt(2*.07/2500))
    assert min_detectable_gap(np.int64(500), np.int64(5), .07) == min_detectable_gap(500, 5, .07)
