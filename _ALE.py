"""accumulated local effect for regression and classification models"""

import numpy as np

from sklearn.base import is_regressor
from sklearn.utils import _safe_indexing, _get_column_indices
from sklearn.utils.extmath import cartesian
from scipy.stats.mstats import mquantiles
from torch import quantile

def _quantiles_from_x(x, n_quantiles):
    """Generate a grid of points based on the quantiles of x.
    Parameters
    ----------
    x : ndarray, shape (n_samples, n_target_features)
        The data

    n_quantiles : int
        The number of quantiles on the grid for each
        feature.

    Returns
    -------
    grid : ndarray, shape (n_points, n_target_features)
        A value for each feature at each point in the grid. ``n_points`` is
        always ``<= n_quantiles ** x.shape[1]``.

    values : list of 1d ndarrays
        The values with which the grid has been created. The size of each
        array ``values[j]`` is either ``n_quantiles``, or the number of
        unique values in ``x[:, j]``, whichever is smaller.
    """
    if n_quantiles <= 1:
        raise ValueError("'n_quantiles' must be strictly greater than 1.")

    for feature in range(x.shape[1]):
        uniques = np.unique(_safe_indexing(x, feature, axis=1))
    if uniques.shape[0] < n_quantiles:
        # feature has low resolution use unique vals
        emp_percentiles = uniques
    else:
        # create axis based on percentiles and grid resolution
        emp_percentiles = mquantiles(
            _safe_indexing(x, feature, axis=1), prob=np.linspace(0, 1, n_quantiles)
        )

    return emp_percentiles

def _ale_for_numeric(est, grid, x, feature, response_method='auto'):
    """computes first order accumulated local efffects for a numeric feature"""

    # define the prediction_method (predict, predict_proba, decision_function).
    
    if is_regressor(est):
        prediction_method = est.predict
    else:
        predict_proba = getattr(est, 'predict_proba', None)
        decision_function = getattr(est, 'decision_function', None)
        if response_method == 'auto':
            # try predict_proba, then decision_function if it doesn't exist
            prediction_method = predict_proba or decision_function
        else:
            prediction_method = (predict_proba if response_method ==
                                 'predict_proba' else decision_function)
        if prediction_method is None:
            if response_method == 'auto':
                raise ValueError(
                    'The estimator has no predict_proba and no '
                    'decision_function method.'
                )
            if response_method == 'predict_proba':
                raise ValueError('The estimator has no predict_proba method.')
            raise ValueError('The estimator has no decision_function method.')

    x_eval = x.copy()
    x_feat = _safe_indexing(x, feature, axis=1)
    quantiles = np.quantile(np.nan_to_num(x_feat.iloc[:,0]), q=grid).astype(int)
    x_eval_2 = x.copy()
    x_feat_incremented = _safe_indexing(x_eval_2, feature, axis=1)
    x_feat_incremented = grid[quantiles + 1]
    x_eval_2.iloc[:,feature[0]] = x_feat_incremented
    y_hat = prediction_method(x_eval)
    y_hat_2 = prediction_method(x_eval_2)
    delta = y_hat_2 - y_hat
    quantile_delta = np.array([quantiles, delta]).T
    effects = np.split(quantile_delta[:,1], np.unique(quantile_delta[:, 0], return_index=True)[1][1:])
    avg_effects = effects.mean().flatten()
    ale_edges = np.array([0, * np.cumsum(avg_effects)])
    ale_centers = (ale_edges[1:] + ale_edges[:-1]) / 2
    ale = ale_centers - np.sum(ale_centers * effects.size / x.shape[0])
    return ale

def _ale_for_categorical(est, grid, x, feature, response_method='auto'):
    """computes first order accumulated local efffects for a categorical feature"""
    if is_regressor(est):
        prediction_method = est.predict
    else:
        predict_proba = getattr(est, 'predict_proba', None)
        decision_function = getattr(est, 'decision_function', None)
        if response_method == 'auto':
            # try predict_proba, then decision_function if it doesn't exist
            prediction_method = predict_proba or decision_function
        else:
            prediction_method = (predict_proba if response_method ==
                                 'predict_proba' else decision_function)
        if prediction_method is None:
            if response_method == 'auto':
                raise ValueError(
                    'The estimator has no predict_proba and no '
                    'decision_function method.'
                )
            if response_method == 'predict_proba':
                raise ValueError('The estimator has no predict_proba method.')
            raise ValueError('The estimator has no decision_function method.')

    categories = feature.value_counts().sort_index().values
    n_categories = len(feature.unique())
    effects = np.zeros((n_categories, n_categories))
    for i in range(n_categories):
        x_eval = x[x.iloc[:, 0] == categories[i]]
        x_min = x.copy()
        x_plus  = x.copy()
        x_min.iloc[:, 0] = grid[i - 1]
        x_plus.iloc[:, 0] = grid[i + 1]
        effects[i] += (prediction_method(x_plus) - prediction_method(x_min)).sum() / x_eval.shape[0]
    distance_matrix = effects.cumsum()
    ale = distance_matrix.mean()
    return ale

def accumulated_local_effects(est, x, feature, n_quantiles):
    """calculates ale for a feature"""
    ale = np.array(n_quantiles,)
    features_indices = np.asarray(
        _get_column_indices(x, feature), dtype=np.int32, order='C'
    ).ravel()
    quantiles = _quantiles_from_x(_safe_indexing(x, features_indices, axis=1), n_quantiles)
    x_feat = _safe_indexing(x, feature, axis=1)
    if x_feat.to_numpy().dtype.name == "category" or x_feat.to_numpy().dtype == "object":
        ale = _ale_for_categorical(est, quantiles, x, x_feat)
    else:
        ale = _ale_for_numeric(est, quantiles, x, x_feat)
    return ale
