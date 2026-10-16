"""Jackknife errors.  Every error bar in the Helicity pipeline uses this convention.

The Exp rows of the rpitd files are already jackknife samples: row i is the
average over every configuration except i (so their spread is ~1/(N-1) of the
per-configuration spread).  A quantity theta is therefore computed on each row
directly, theta_i = f(row_i), and its error is

    sigma_JK^2 = (N - 1) / N * sum_i (theta_i - theta_bar)^2 = (N - 1) * var(theta_i),

i.e. sigma_JK = sqrt(N - 1) * std(theta_i, ddof=0).  The synthetic Exp replicas
mimic the real rows, so the same rule applies to every model projection.
Independent ensembles (different FM) are jackknifed one at a time, holding the
others at their mean, and their variances add.

Evidence: with std/sqrt(N) instead, the model_selection fits had chi^2/dof
~ (N-1)^2 ~ 1e6 in every FM; with sigma_JK they are O(1).
"""

import numpy as np


def jk_error(theta, axis=0):
    """Jackknife error from per-sample estimates ``theta`` along ``axis``; NaNs are skipped."""
    theta = np.asarray(theta, dtype=float)
    n = np.isfinite(theta).sum(axis=axis)
    dev2 = np.nansum((theta - np.nanmean(theta, axis=axis, keepdims=True)) ** 2, axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 1, np.sqrt((n - 1) / n * dev2), 0.0)


def jk_series_error(s):
    """pandas-friendly jackknife error of a column of jackknife samples, for ``groupby(...).agg``."""
    return float(jk_error(np.asarray(s, dtype=float)))
