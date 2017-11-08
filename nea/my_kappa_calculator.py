import numpy as np
from quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
from quadratic_weighted_kappa import linear_weighted_kappa as lwk

def assert_inputs(rater_a, rater_b):
	assert np.issubdtype(rater_a.dtype, np.integer), 'Integer array expected, got ' + str(rater_a.dtype)
	assert np.issubdtype(rater_b.dtype, np.integer), 'Integer array expected, got ' + str(rater_b.dtype)

def quadratic_weighted_kappa(rater_a, rater_b, min_rating, max_rating):
	assert_inputs(rater_a, rater_b)
	return qwk(rater_a, rater_b, min_rating, max_rating)

def linear_weighted_kappa(rater_a, rater_b, min_rating, max_rating):
	assert_inputs(rater_a, rater_b)
	return lwk(rater_a, rater_b, min_rating, max_rating)
