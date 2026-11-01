"""
07 · exp_synthetic.py · Does the method find a correction we already know?
=========================================================================

IN ONE LINE
    Builds fake data with a KNOWN correction and checks which methods recover it.

WHY
    With real data the true answer is unknown. Here it is known, so wrong terms can be counted.

WHAT HAPPENS
    The fake temperature is driven by the real 2008 dry-season radiation, with
        s_true = c1*(T-T0)/dT + c2*(T-T0)/dT * R/Rs     (a relaxation + a T-dependent response to R)
    A. Raw T, R library vs standardized library (conditioning).
    B. Five ways to get dT/dt vs the weak form, with noise 0.05-0.3 K, 5 seeds each.
       Counts true terms found and false (spurious) terms. Weak form: 0 spurious.
    C. Identifiability: scale alpha and gamma together and see which terms change.

INPUT    ../work/dry_res.npz (step 01)
OUTPUT   printed text; must equal ../reference/synthetic_results.txt
TIME     about 4 min
RUN      cd 07_diagnostics ; python exp_synthetic.py > synthetic_new.txt
PAPER    Sec. IV C (identifiability), choice of the weak form (Sec. II C)
NEEDS    numpy, scipy, pysindy
"""
import numpy as np, warnings, sys
sys.path.insert(0, '.')
from common import *
warnings.filterwarnings('ignore')

d = np.load('../work/dry_res.npz')
R = d['R']; a, g = float(d['a_h']), float(d['g_h'])
T0, dT, Rs, c1, c2 = 298.0, 3.0, 300.0, -0.25, -0.12
s_true = lambda T, R: c1 * (T - T0) / dT + c2 * (T - T0) / dT * R / Rs
rhs_true = lambda T, R: a * R - g * T ** 4 + s_true(T, R)
T_clean = simulate(d['T'][0], R, rhs_true)
assert np.all(np.isfinite(T_clean)) and 285 < T_clean.min() and T_clean.max() < 315
print(f'a={a:.3e} g={g:.3e}  T range {T_clean.min():.1f}-{T_clean.max():.1f} K, '
      f'std(s_true)={s_true(T_clean, R).std():.3f} K/h, std(dT/dt)={np.gradient(T_clean, H).std():.3f} K/h')

mT, sT, mR, sR = T_clean.mean(), T_clean.std(), R.mean(), R.std()


def lib(T, R, degree=3):
    return poly_library([(T - mT) / sT, (R - mR) / sR], ['th', 'r'], degree)


Th_c, names = lib(T_clean, R)
# truth expressed in the standardised basis (exact, since it is affine in th and r)
xi_true = np.linalg.lstsq(Th_c, s_true(T_clean, R), rcond=None)[0]
xi_true[np.abs(xi_true) < 1e-10] = 0
true_support = np.abs(xi_true) > 0.02   # terms detectable at the STLSQ threshold
print('true support (|coef| > threshold):', [n for n, k in zip(names, true_support) if k])


EXPS = [(0, 0) if n == '1' else (n.split('*').count('th'), n.split('*').count('r')) for n in names]


def fast_rhs(xi, ak, gk):
    """Scalar right-hand side using only the active monomials (fast inside RK4)."""
    act = [(c, i, j) for c, (i, j) in zip(xi, EXPS) if c != 0]
    def rhs(T, R_):
        th, rr = (T - mT) / sT, (R_ - mR) / sR
        return ak * R_ - gk * T ** 4 + sum(c * th ** i * rr ** j for c, i, j in act)
    return rhs


def evaluate(xi, T_obs):
    Tsim = simulate(T_obs[0], R, fast_rhs(xi, a, g))
    rmse = np.sqrt(np.mean((Tsim - T_clean) ** 2)) if np.all(np.isfinite(Tsim)) else np.inf
    sup = np.abs(xi) > 0
    return rmse, int((sup & true_support).sum()), int((sup & ~true_support).sum()), \
        np.linalg.norm(xi - xi_true) / np.linalg.norm(xi_true)


# ---------------------------------------------------- A. RAW vs STANDARDIZED library (conditioning)
print('\n[A] Library conditioning (noise 0.1 K, Savitzky-Golay derivative)')
rng = np.random.default_rng(0)
T_obs = T_clean + rng.normal(0, 0.1, len(T_clean))
y = derivative(T_obs, 'savgol') - a * R + g * T_obs ** 4
Th_raw, names_raw = poly_library([T_obs, R], ['T', 'R'], 4)
print(f'  cond(raw T,R degree 4) = {np.linalg.cond(Th_raw):.1e}; cond(standardised degree 3) = {np.linalg.cond(lib(T_obs, R)[0]):.1e}')
xi_raw = stlsq(Th_raw, y, threshold=1e-3)
print(f'  raw library: {np.sum(xi_raw != 0)} active terms of {len(xi_raw)}:',
      ', '.join(f'{n}={c:.2e}' for n, c in zip(names_raw, xi_raw) if c != 0))
xi_std = stlsq(lib(T_obs, R)[0], y, threshold=0.02)
r = evaluate(xi_std, T_obs)
print('  standardised library:', ', '.join(f'{n}={c:+.3f}' for n, c in zip(names, xi_std) if c != 0),
      f'| true: ' + ', '.join(f'{n}={c:+.3f}' for n, c in zip(names, xi_true) if c != 0))
print(f'  standardised: sim RMSE={r[0]:.3f} K, true+={r[1]}/{true_support.sum()}, false+={r[2]}, coef err={r[3]:.2f}')

# ---------------------------------------------------- B. HOW TO GET dT/dt vs NOISE (counts true and false terms)
print('\n[B] Derivative strategy vs measurement noise (5 seeds; mean over seeds)')
print(f'  noise  method   simRMSE(K)  true+/{true_support.sum()}  false+  coefErr')
methods = ['fd', 'savgol', 'savgol_both', 'spline', 'kalman', 'weak']   # tv run separately (slow)
for noise in [0.05, 0.15, 0.3]:
    for m in methods:
        out = []
        for seed in range(5):
            T_obs = T_clean + np.random.default_rng(seed).normal(0, noise, len(T_clean))
            Th = lib(T_obs, R)[0]
            if m == 'savgol_both':     # same Savitzky-Golay filter applied to both sides of the ODE
                yy = derivative(T_obs, 'savgol') - savgol_filter(a * R - g * T_obs ** 4, 7, 2)
                xi = stlsq(savgol_filter(Th, 7, 2, axis=0), yy, threshold=0.02)
            elif m == 'weak':
                G, b = weak_system(T_obs, R, Th, a, g)
                xi = stlsq(G / 1.0, b, threshold=0.02)
            else:
                yy = derivative(T_obs, m) - a * R + g * T_obs ** 4
                xi = stlsq(Th, yy, threshold=0.02)
            out.append(evaluate(xi, T_obs))
        o = np.array(out, dtype=float)
        print(f'  {noise:4.2f}  {m:7s}  {np.mean(o[:,0]):8.3f}   {o[:,1].mean():4.1f}   {o[:,2].mean():4.1f}   {o[:,3].mean():.2f}')

print('  (total-variation derivative, 1 seed, noise 0.15 K)')
T_obs = T_clean + np.random.default_rng(0).normal(0, 0.15, len(T_clean))
for alpha_tv in [1e-4, 1e-3]:
    yy = ps.SINDyDerivative(kind='trend_filtered', order=0, alpha=alpha_tv)._differentiate(T_obs[:, None], np.arange(len(T_obs)) * H)[:, 0] - a * R + g * T_obs ** 4
    r = evaluate(stlsq(lib(T_obs, R)[0], yy, threshold=0.02), T_obs)
    print(f'  0.15  tv(a={alpha_tv:g}) {r[0]:8.3f}   {r[1]:4d}   {r[2]:4d}   {r[3]:.2f}')

# ---------------------------------------------------- C. IDENTIFIABILITY: scale a and g together
print('\n[C] Identifiability: refit the correction with (a, g) scaled by k along the valley')
T_obs = T_clean + np.random.default_rng(1).normal(0, 0.1, len(T_clean))
for k in [0.5, 1.0, 1.5]:
    ak, gk = a * k, g * k
    G, b = weak_system(T_obs, R, lib(T_obs, R)[0], ak, gk)
    xi = stlsq(G, b, threshold=0.02)
    Tsim = simulate(T_obs[0], R, fast_rhs(xi, ak, gk))
    print(f'  k={k}: correction = ' + ', '.join(f'{n}={c:+.3f}' for n, c in zip(names, xi) if c != 0)
          + f' | sim RMSE={np.sqrt(np.nanmean((Tsim - T_clean) ** 2)):.3f} K')
