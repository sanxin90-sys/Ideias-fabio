"""
07 · common.py · toolbox for the synthetic test
===============================================

IN ONE LINE
    Helper functions used only by exp_synthetic.py. Computes nothing by itself.

WHAT IS INSIDE
    poly_library   all products of the variables up to a given degree (the candidate terms)
    fourier_cols   sine/cosine of the hour of day (tested and rejected in the paper)
    derivative     five ways to differentiate a noisy series (fd, Savitzky-Golay, TV, spline, Kalman)
    weak_system    the weak form (same idea as rebio.weak_system)
    stlsq          sparse regression from the pysindy library (sets small coefficients to zero)
    simulate       RK4 integration of a model, stops if T leaves 200-400 K

MODEL    dT/dt = a R - g T^4 + s(x); time in hours, T in K, R in W m^-2.
NEEDS    numpy, scipy, pysindy
"""
import numpy as np
import pysindy as ps
from itertools import combinations_with_replacement
from scipy.signal import savgol_filter

H = 0.5  # sampling step (h)


# ---------------------------------------------------------------- libraries
def poly_library(cols, names, degree):
    """All monomials up to `degree` of the given columns (incl. constant)."""
    feats, fnames = [np.ones(len(cols[0]))], ['1']
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(len(cols)), d):
            feats.append(np.prod([cols[i] for i in combo], axis=0))
            fnames.append('*'.join(names[i] for i in combo))
    return np.column_stack(feats), fnames


def fourier_cols(hour, n_harm=2):
    cols, names = [], []
    for k in range(1, n_harm + 1):
        w = 2 * np.pi * k * hour / 24
        cols += [np.sin(w), np.cos(w)]
        names += [f'sin{k}h', f'cos{k}h']
    return cols, names


# ---------------------------------------------------------------- derivatives
def derivative(T, method):
    t = np.arange(len(T)) * H
    if method == 'fd':
        return np.gradient(T, H)
    if method == 'savgol':          # window 3.5 h, quadratic
        return savgol_filter(T, 7, 2, deriv=1, delta=H)
    if method == 'tv':              # total-variation regularised
        return ps.SINDyDerivative(kind='trend_filtered', order=0, alpha=1e-4)._differentiate(T[:, None], t)[:, 0]
    if method == 'spline':
        return ps.SINDyDerivative(kind='spline', s=len(T) * 0.15 ** 2)._differentiate(T[:, None], t)[:, 0]
    if method == 'kalman':
        return ps.SINDyDerivative(kind='kalman', alpha=0.05)._differentiate(T[:, None], t)[:, 0]
    raise ValueError(method)


# ---------------------------------------------------------------- weak form
def weak_system(T, R, Theta, a, g, width_h=4.0, stride=2, p=4):
    """Weak SINDy: integrate the residual equation against compact bumps.

    For phi vanishing at the window ends,  int phi dT/dt = -int phi' T,
    so no derivative of the noisy data is ever taken.
    Returns (G, b) with  G xi ~= b.
    """
    m = int(round(width_h / H))          # half-width in samples
    u = np.linspace(-1, 1, 2 * m + 1)
    phi = (1 - u ** 2) ** p
    dphi = -2 * p * u * (1 - u ** 2) ** (p - 1) / (m * H)
    w = np.full(2 * m + 1, H); w[[0, -1]] = H / 2   # trapezoid weights
    rows_G, rows_b = [], []
    for c in range(m, len(T) - m, stride):
        sl = slice(c - m, c + m + 1)
        lhs = -np.sum(w * dphi * T[sl])                  # int phi dT/dt
        known = np.sum(w * phi * (a * R[sl] - g * T[sl] ** 4))
        rows_b.append(lhs - known)
        rows_G.append((w * phi) @ Theta[sl])
    return np.array(rows_G), np.array(rows_b)


# ---------------------------------------------------------------- regression
def stlsq(Theta, y, threshold, alpha=1e-3):
    opt = ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=50, normalize_columns=False)
    opt.fit(Theta, y[:, None])
    return opt.coef_.ravel()


# ---------------------------------------------------------------- simulation
def simulate(T0, R, rhs, n_sub=6):
    """RK4 with R linearly interpolated inside each 30-min step.
    rhs(T, R) -> dT/dt (K/h). Returns T at the sampling instants (nan if blow-up)."""
    T = np.empty(len(R)); T[0] = T0
    h = H / n_sub
    for i in range(len(R) - 1):
        x = T[i]
        for j in range(n_sub):
            f0 = j / n_sub; f1 = (j + 0.5) / n_sub; f2 = (j + 1) / n_sub
            r0 = R[i] + f0 * (R[i + 1] - R[i]); r1 = R[i] + f1 * (R[i + 1] - R[i]); r2 = R[i] + f2 * (R[i + 1] - R[i])
            k1 = rhs(x, r0); k2 = rhs(x + h * k1 / 2, r1); k3 = rhs(x + h * k2 / 2, r1); k4 = rhs(x + h * k3, r2)
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not np.isfinite(x) or x < 200 or x > 400:   # physical sanity bound
            T[i + 1:] = np.nan
            return T
        T[i + 1] = x
    return T
