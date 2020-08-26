"""accumulated local effect for regression and classification models."""

import numpy as np
import pandas as pd

from sklearn.utils import _safe_indexing
from sklearn.utils.extmath import cartesian

def _quantiles_from_X(X, n_quantiles):
    """Generate a grid of points based on the quantiles of X.
    Parameters
    ----------
    X : ndarray, shape (n_samples, n_target_features)
        The data

    n_quantiles : int
        The number of quantiles on the grid for each
        feature.

    Returns
    -------
    grid : ndarray, shape (n_points, n_target_features)
        A value for each feature at each point in the grid. ``n_points`` is
        always ``<= n_quantiles ** X.shape[1]``.

    values : list of 1d ndarrays
        The values with which the grid has been created. The size of each
        array ``values[j]`` is either ``n_quantiles``, or the number of
        unique values in ``X[:, j]``, whichever is smaller.
    """
     if n_quantiles <= 1:
        raise ValueError("'n_quantiles' must be strictly greater than 1.")
     values = []
     for feature in range(X.shape[1]):
         uniques = np.unique(_safe_indexing(X, feature, axis=1))
        if uniques.shape[0] < n_quantiles:
            # feature has low resolution use unique vals
            axis = uniques
        else:
            # create axis based on percentiles and grid resolution
            emp_percentiles = mquantiles(
                _safe_indexing(X, feature, axis=1), prob=np.linspace(0, 1, n_quantiles)
            )

    return cartesian(values), values

def _calculate_accumulated_local_effects(est, grid, features, X, response_method):
    predictions = []
    averaged_predictions = []

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
            elif response_method == 'predict_proba':
                raise ValueError('The estimator has no predict_proba method.')
            else:
                raise ValueError(
                    'The estimator has no decision_function method.')
    X_eval = X.copy()
    a1 = pd.cut(X_eval.iloc[:, 0], bins=grid[0], labels=False).fillna(0.0).astype(int)
    X_eval_2 = X.copy()
    X_eval_2.iloc[:, 0] = grid[0][a1 + 1]
    y_hat = prediction_method(X_eval)
    y_hat_2 = prediction_method(X_eval_2)
    delta = y_hat_2 - y_hat
    effects  = pd.DataFrame({"a1": a1, "delta": delta}).groupby(
        "a1"
    ).mean()
    b1 = np.array(a1.value_counts(sort=False))
    centers = (effects[:-1] + effects[1:]) / 2
    predictions = []
    for i, value in enumerate(effects.to_numpy()):
        predictions.append(value * b1[i])
    return averaged_predictions, predictions
