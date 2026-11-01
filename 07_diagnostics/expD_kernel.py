"""
07 · expD_kernel.py · How long does the air "remember" radiation?
================================================================

IN ONE LINE
    Fits a linear memory kernel: how much radiation from 0-24 h ago still affects s now.

WHAT HAPPENS
        s(t) = c + b (T(t) - Tbar) + sum_k K(k dt) R(t - k dt) dt,   dt = 0.5 h
    1. Fit K by least squares with a smoothness penalty (penalty chosen by 5-fold CV over days).
    2. Fit K(tau) for tau >= 1 h with A exp(-a tau); the memory time is 1/a.
    3. Repeat on the synthetic twin, where the true memory time is 3 h (sanity check).

    Finding: about 0.9 h in the wet season; no clear exponential in the dry season.

INPUT    ../work/{dry,wet}_res.npz, ../work/y2008.pkl, ../work/dry_params.npy (step 01)
OUTPUT   expD_results.json, expD_kernels.npz
TIME     a few seconds
RUN      cd 07_diagnostics ; python expD_kernel.py
PAPER    Sec. III E
"""
import numpy as np, pandas as pd, json
from scipy.optimize import curve_fit

NL, DT = 49, 0.5


def design(T, R):
    n = len(T)
    X = np.full((n, 2 + NL), np.nan)
    X[:, 0] = 1; X[:, 1] = T - T.mean()
    for k in range(NL):
        X[k:, 2 + k] = R[:n - k] * DT
    return X


def fit(X, y, lam):
    D = np.diff(np.eye(NL), 2, axis=0)
    P = np.zeros((X.shape[1], X.shape[1])); P[2:, 2:] = lam * D.T @ D + 1e-6 * np.eye(NL)
    return np.linalg.solve(X.T @ X + P, X.T @ y)


def cv(X, y, day, lam, cols=None):
    cols = np.arange(X.shape[1]) if cols is None else cols
    u = np.unique(day); folds = np.array_split(u, 5); pred = np.empty_like(y)
    for f in folds:
        te = np.isin(day, f)
        Xs = X[:, cols]
        if len(cols) == X.shape[1]:
            beta = fit(Xs[~te], y[~te], lam)
        else:
            beta = np.linalg.lstsq(Xs[~te], y[~te], rcond=None)[0]
        pred[te] = Xs[te] @ beta
    return 1 - np.mean((y - pred) ** 2) / np.var(y)


def analyse(T, R, s, label):
    X = design(T, R); ok = np.isfinite(X).all(1) & np.isfinite(s)
    X, y = X[ok], s[ok]; day = np.arange(len(T))[ok] // 48
    lams = 10.0 ** np.arange(-4, 8)
    scores = [cv(X, y, day, l) for l in lams]
    lam = lams[int(np.argmax(scores))]
    beta = fit(X, y, lam); K = beta[2:]; tau = np.arange(NL) * DT
    m = tau >= 1.0
    try:
        (A, a), _ = curve_fit(lambda t, A, a: A * np.exp(-a * t), tau[m], K[m], p0=[K[m][0], 0.3], maxfev=20000)
        resid = K[m] - A * np.exp(-a * tau[m]); fitq = 1 - resid.var() / K[m].var()
    except Exception:
        A = a = fitq = np.nan
    r = {'CV_R2_pointwise_linear(T,R)': round(cv(X, y, day, 0, cols=np.array([0, 1, 2])), 3),
         'CV_R2_kernel': round(max(scores), 3), 'lambda': float(lam),
         'K0_instant': float(K[0]), 'K_1h': float(K[2]), 'K_3h': float(K[6]), 'K_6h': float(K[12]),
         'K_12h': float(K[24]), 'integral_K_over_1h_plus': float(K[m].sum() * DT),
         'exp_fit_A': float(A), 'exp_fit_a_per_h': float(a),
         'memory_time_h': float(1 / a) if np.isfinite(a) and a > 0 else None,
         'exp_fit_R2_on_kernel': float(fitq)}
    print(label, r, flush=True)
    return r, K


out, kernels = {}, {}
for season in ['dry', 'wet']:
    d = np.load(f'../work/{season}_res.npz')
    out[season], kernels[season] = analyse(d['T'], d['R'], d['s'], season)

# synthetic twin of Exp B (true kernel: -kappa/tau e^{-tau'/3h})
import expB_synth_data as syn
out['synthetic'], kernels['synthetic'] = analyse(syn.Tobs, syn.R, syn.s_exp, 'synthetic (true memory time 3 h)')
out['synthetic']['true_kernel_a_per_h'] = 1 / syn.tau
json.dump(out, open('expD_results.json', 'w'), indent=1)
np.savez('expD_kernels.npz', tau=np.arange(NL) * DT, **kernels)
