'''Plot accumulated local effect for regression and classification models'''

import numbers
from itertools import chain
from itertools import count
from math import ceil
import numpy as np
from sklearn.utils import _safe_indexing
from sklearn.utils import check_array
from sklearn.base import is_regressor
from joblib import Parallel, delayed
from scipy.stats.mstats import mquantiles
from scipy import sparse
from matplotlib import transforms

from _ale import accumulated_local_effects

def plot_ale(estimator, X, features, *, feature_names=None,
                            target=None, n_cols=3,
                            n_quantiles=40,
                            n_jobs=None, verbose=0,
                            line_kw=None, contour_kw=None, ax=None,
                            subsample=1000, kind="Average"):
    """Accumulated Local Effect(ALE) plots.

    The ``len(features)`` plots are arranged in a grid with ``n_cols``
    columns.  The deciles of the feature values will be shown with tick
    marks on the x-axes for one-way plots"""

    import matplotlib.pyplot as plt  # noqa


    # set target_idx for multi-class estimators
    if hasattr(estimator, 'classes_') and np.size(estimator.classes_) > 2:
        if target is None:
            raise ValueError('target must be specified for multi-class')
        target_idx = np.searchsorted(estimator.classes_, target)
        if (not (0 <= target_idx < len(estimator.classes_)) or
                estimator.classes_[target_idx] != target):
            raise ValueError('target not in est.classes_, got {}'.format(
                target))
    else:
        # regression and binary classification
        target_idx = 0

    # Use check_array only on lists and other non-array-likes / sparse. Do not
    # convert DataFrame into a NumPy array.
    if not(hasattr(X, '__array__') or sparse.issparse(X)):
        X = check_array(X, force_all_finite='allow-nan', dtype=object)
    n_features = X.shape[1]

    # convert feature_names to list
    if feature_names is None:
        if hasattr(X, "loc"):
            # get the column names for a pandas dataframe
            feature_names = X.columns.tolist()
        else:
            # define a list of numbered indices for a numpy array
            feature_names = [str(i) for i in range(n_features)]
    elif hasattr(feature_names, "tolist"):
        # convert numpy array or pandas index to a list
        feature_names = feature_names.tolist()
    if len(set(feature_names)) != len(feature_names):
        raise ValueError('feature_names should not contain duplicates.')

    def convert_feature(fx):
        if isinstance(fx, str):
            try:
                fx = feature_names.index(fx)
            except ValueError:
                raise ValueError('Feature %s not in feature_names' % fx)
        return int(fx)

    # convert features into a seq of int tuples
    tmp_features = []
    for fxs in features:
        if isinstance(fxs, (numbers.Integral, str)):
            fxs = (fxs,)
        try:
            fxs = tuple(convert_feature(fx) for fx in fxs)
        except TypeError:
            raise ValueError('Each entry in features must be either an int, '
                             'a string, or an iterable of size at most 2.')
        if not 1 <= np.size(fxs) <= 2:
            raise ValueError('Each entry in features must be either an int, '
                             'a string, or an iterable of size at most 2.')
        if kind != 'average' and np.size(fxs) > 1:
            raise ValueError(
                f"It is not possible to display individual effects for more "
                f"than one feature at a time. Got: features={features}.")
        tmp_features.append(fxs)

    features = tmp_features

    # Early exit if the axes does not have the correct number of axes
    if ax is not None and not isinstance(ax, plt.Axes):
        axes = np.asarray(ax, dtype=object)
        if axes.size != len(features):
            raise ValueError("Expected ax to have {} axes, got {}".format(
                             len(features), axes.size))

    for i in chain.from_iterable(features):
        if i >= len(feature_names):
            raise ValueError('All entries of features must be less than '
                             'len(feature_names) = {0}, got {1}.'
                             .format(len(feature_names), i))

    if isinstance(subsample, numbers.Integral):
        if subsample <= 0:
            raise ValueError(
                f"When an integer, subsample={subsample} should be positive."
            )
    elif isinstance(subsample, numbers.Real):
        if subsample <= 0 or subsample >= 1:
            raise ValueError(
                f"When a floating-point, subsample={subsample} should be in "
                f"the (0, 1) range."
            )

    # compute predictions and/or averaged predictions
    ale_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(accumulated_local_effects)(estimator, X, fxs,
                                    n_quantiles=n_quantiles,)
        for fxs in features)

    # For multioutput regression, we can only check the validity of target
    # now that we have the predictions.
    # Also note: as multiclass-multioutput classifiers are not supported,
    # multiclass and multioutput scenario are mutually exclusive. So there is
    # no risk of overwriting target_idx here.
    ale_result = ale_results[0]  # checking the first result is enough
    n_tasks = (ale_result.average.shape[0] if kind == 'average'
               else ale_result.individual.shape[0])
    if is_regressor(estimator) and n_tasks > 1:
        if target is None:
            raise ValueError(
                'target must be specified for multi-output regressors')
        if not 0 <= target <= n_tasks:
            raise ValueError(
                'target must be in [0, n_tasks], got {}.'.format(target))
        target_idx = target

    # get global min and max average predictions of ale grouped by plot type
    ale_lim = {}
    for ale in ale_results:
        values = ale["values"]
        preds = (ale.average if kind == 'average' else ale.individual)
        min_ale = preds[target_idx].min()
        max_ale = preds[target_idx].max()
        n_fx = len(values)
        old_min_ale, old_max_ale = ale_lim.get(n_fx, (min_ale, max_ale))
        min_ale = min(min_ale, old_min_ale)
        max_ale = max(max_ale, old_max_ale)
        ale_lim[n_fx] = (min_ale, max_ale)

    deciles = {}
    for fx in chain.from_iterable(features):
        if fx not in deciles:
            X_col = _safe_indexing(X, fx, axis=1)
            deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))

    display = ALEDisplay(ale_results=ale_results,
                         features=features,
                         feature_names=feature_names,
                         target_idx=target_idx,
                         ale_lim=ale_lim,
                         deciles=deciles,
                         kind=kind,
                         subsample=subsample)
    return display.plot(ax=ax, n_cols=n_cols, line_kw=line_kw,
                        contour_kw=contour_kw)

class ALEDisplay:
    """Accumulated Local Effects (ALE) Plot"""

    def __init__(self, ale_results, *, features, feature_names, target_idx,
                 ale_lim, deciles, kind='average', subsample=1000):
        self.ale_results = ale_results
        self.features = features
        self.feature_names = feature_names
        self.target_idx = target_idx
        self.ale_lim = ale_lim
        self.deciles = deciles
        self.kind = kind
        self.subsample = subsample

    def _get_sample_count(self, n_samples):
        if isinstance(self.subsample, numbers.Integral):
            if self.subsample < n_samples:
                return self.subsample
            return n_samples
        elif isinstance(self.subsample, numbers.Real):
            return ceil(n_samples * self.subsample)
        return n_samples

    def plot(self, ax=None, n_cols=3, line_kw=None, contour_kw=None):
        import matplotlib.pyplot as plt  # noqa
        from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa

        if line_kw is None:
            line_kw = {}
        if contour_kw is None:
            contour_kw = {}

        if ax is None:
            _, ax = plt.subplots()

        default_contour_kws = {"alpha": 0.75}
        contour_kw = {**default_contour_kws, **contour_kw}

        default_line_kws = {'color': 'C0'}
        line_kw = {**default_line_kws, **line_kw}
        individual_line_kw = line_kw.copy()

        if self.kind == 'individual' or self.kind == 'both':
            individual_line_kw['alpha'] = 0.3
            individual_line_kw['linewidth'] = 0.5

        n_features = len(self.features)
        n_sampled = 1
        if self.kind == 'individual':
            n_instances = len(self.ale_results[0].individual[0])
            n_sampled = self._get_sample_count(n_instances)
        elif self.kind == 'both':
            n_instances = len(self.ale_results[0].individual[0])
            n_sampled = self._get_sample_count(n_instances) + 1

        if isinstance(ax, plt.Axes):
            # If ax was set off, it has most likely been set to off
            # by a previous call to plot.
            if not ax.axison:
                raise ValueError("The ax was already used in another plot "
                                 "function, please set ax=display.axes_ "
                                 "instead")

            ax.set_axis_off()
            self.bounding_ax_ = ax
            self.figure_ = ax.figure

            n_cols = min(n_cols, n_features)
            n_rows = int(np.ceil(n_features / float(n_cols)))

            self.axes_ = np.empty((n_rows, n_cols), dtype=object)
            if self.kind == 'average':
                self.lines_ = np.empty((n_rows, n_cols), dtype=object)
            else:
                self.lines_ = np.empty((n_rows, n_cols, n_sampled),
                                       dtype=object)
            self.contours_ = np.empty((n_rows, n_cols), dtype=object)

            axes_ravel = self.axes_.ravel()

            gs = GridSpecFromSubplotSpec(n_rows, n_cols,
                                         subplot_spec=ax.get_subplotspec())
            for i, spec in zip(range(n_features), gs):
                axes_ravel[i] = self.figure_.add_subplot(spec)

        else:  # array-like
            ax = np.asarray(ax, dtype=object)
            if ax.size != n_features:
                raise ValueError("Expected ax to have {} axes, got {}"
                                 .format(n_features, ax.size))

            if ax.ndim == 2:
                n_cols = ax.shape[1]
            else:
                n_cols = None

            self.bounding_ax_ = None
            self.figure_ = ax.ravel()[0].figure
            self.axes_ = ax
            if self.kind == 'average':
                self.lines_ = np.empty_like(ax, dtype=object)
            else:
                self.lines_ = np.empty(ax.shape + (n_sampled,),
                                       dtype=object)
            self.contours_ = np.empty_like(ax, dtype=object)

        # create contour levels for two-way plots
        if 2 in self.ale_lim:
            Z_level = np.linspace(*self.ale_lim[2], num=8)

        self.deciles_vlines_ = np.empty_like(self.axes_, dtype=object)
        self.deciles_hlines_ = np.empty_like(self.axes_, dtype=object)

        # Create 1d views of these 2d arrays for easy indexing
        lines_ravel = self.lines_.ravel(order='C')
        contours_ravel = self.contours_.ravel(order='C')
        vlines_ravel = self.deciles_vlines_.ravel(order='C')
        hlines_ravel = self.deciles_hlines_.ravel(order='C')

        for i, axi, fx, ale_result in zip(count(), self.axes_.ravel(),
                                         self.features, self.ale_results):

            avg_preds = None
            preds = None
            values = ale_result["values"]
            if self.kind == 'individual':
                preds = ale_result.individual
            elif self.kind == 'average':
                avg_preds = ale_result.average
            else:  # kind='both'
                avg_preds = ale_result.average
                preds = ale_result.individual

            if len(values) == 1:
                if self.kind == 'individual' or self.kind == 'both':
                    n_samples = self._get_sample_count(
                        len(preds[self.target_idx])
                    )
                    ice_lines = preds[self.target_idx]
                    sampled = ice_lines[np.random.choice(
                        ice_lines.shape[0], n_samples, replace=False
                    ), :]
                    for j, ins in enumerate(sampled):
                        lines_ravel[i * j + j] = axi.plot(
                            values[0], ins.ravel(), **individual_line_kw
                        )[0]
                if self.kind == 'average':
                    lines_ravel[i] = axi.plot(
                        values[0], avg_preds[self.target_idx].ravel(),
                        **line_kw
                    )[0]
                elif self.kind == 'both':
                    lines_ravel[i] = axi.plot(
                        values[0], avg_preds[self.target_idx].ravel(),
                        label='average', **line_kw
                    )[0]
                    axi.legend()
            else:
                # contour plot
                XX, YY = np.meshgrid(values[0], values[1])
                Z = avg_preds[self.target_idx].T
                CS = axi.contour(XX, YY, Z, levels=Z_level, linewidths=0.5,
                                 colors='k')
                contours_ravel[i] = axi.contourf(XX, YY, Z, levels=Z_level,
                                                 vmax=Z_level[-1],
                                                 vmin=Z_level[0],
                                                 **contour_kw)
                axi.clabel(CS, fmt='%2.2f', colors='k', fontsize=10,
                           inline=True)

            trans = transforms.blended_transform_factory(axi.transData,
                                                         axi.transAxes)
            ylim = axi.get_ylim()
            vlines_ravel[i] = axi.vlines(self.deciles[fx[0]], 0, 0.05,
                                         transform=trans, color='k')
            axi.set_ylim(ylim)

            # Set xlabel if it is not already set
            if not axi.get_xlabel():
                axi.set_xlabel(self.feature_names[fx[0]])

            if len(values) == 1:
                if n_cols is None or i % n_cols == 0:
                    if not axi.get_ylabel():
                        axi.set_ylabel('Partial dependence')
                else:
                    axi.set_yticklabels([])
                axi.set_ylim(self.ale_lim[1])
            else:
                # contour plot
                trans = transforms.blended_transform_factory(axi.transAxes,
                                                             axi.transData)
                xlim = axi.get_xlim()
                hlines_ravel[i] = axi.hlines(self.deciles[fx[1]], 0, 0.05,
                                             transform=trans, color='k')
                # hline erases xlim
                axi.set_ylabel(self.feature_names[fx[1]])
                axi.set_xlim(xlim)
        return self
