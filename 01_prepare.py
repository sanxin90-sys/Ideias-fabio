"""
STEP 01 · Baseline model and residual (2008)
============================================

IN ONE LINE
    Fits the simple physical model and computes what it gets wrong (the residual s).

WHAT HAPPENS
    1. Read the 2008 columns of the spreadsheet; split dry (Jul-Sep) and wet (Jan-Mar).
    2. Fit alpha*, gamma* of  dT/dt = alpha R - gamma T^4  by running the model (RK4)
       and making the simulated temperature as close as possible to the measured one.
    3. Residual:  s = measured dT/dt - alpha R + gamma T^4.
    4. Table V: how much of s a flexible regressor can predict (5 folds grouped by day).
    5. Condition numbers: how much T, T^2, T^3, T^4 get confused, raw vs standardized.
    6. Compare with the paper (check lines).

INPUT    data/Dados Rebio Jaru 2006 a 2008.xlsx
OUTPUT   work/dry_res.npz, work/wet_res.npz   <- every later step reads these
         results/prepare.json
TIME     1-5 min
RUN      python 01_prepare.py
PAPER    Sec. II B, Sec. IV C, Table V
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import rebio as rb

# ------------------------------------------------------------------ 1. READ 2008
COLS = {'R_liq.4': 'R', 'T_ar_62m.1': 'Tc', 'UR_62m.4': 'RH', 'VV_ane_62m.4': 'U', 'PRP.3': 'P'}
raw = rb.sheet()
y2008 = raw[['Ano.4', 'Mês', 'Dia', 'Hora'] + list(COLS)].iloc[:366 * rb.NPD].reset_index(drop=True)
y2008.to_pickle(f'{rb.WORK}/y2008.pkl')     # by position: the year column has a few blanks
y2008 = y2008.rename(columns=COLS)

out = {}
for season, months in rb.SEASONS.items():
    s = y2008[y2008['Mês'].isin(months)].reset_index(drop=True)
    s[list(COLS.values())] = s[list(COLS.values())].interpolate(limit=4)    # gaps up to 2 h
    s = s.dropna(subset=['R', 'Tc']).reset_index(drop=True)
    R, T = s.R.values, s.Tc.values + 273.15

    # -------------------------------------------------------------- 2. FIT alpha*, gamma*
    # Run the simple model (RK4) and let the optimizer pick the two numbers that make the
    # simulated T closest to the measured T over the whole season.
    # Fitted in the units of the original notebook (time unit 1/(60e-6*60) h), then converted to hours.
    dt = 30 * 60e-6

    def sim(a, g):
        Tm = np.empty_like(T); Tm[0] = T[0]
        f = lambda x, r: a * r - g * x ** 4
        for i in range(1, len(T)):
            x, r = Tm[i - 1], R[i - 1]
            k1 = dt * f(x, r); k2 = dt * f(x + k1 / 2, r); k3 = dt * f(x + k2 / 2, r); k4 = dt * f(x + k3, r)
            Tm[i] = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return Tm

    fit = minimize(lambda p: np.mean((T - sim(p[0], p[1] * 1e-8)) ** 2), [1.2, 1.0], bounds=[(0.03, 48), (0.025, 40)])
    a, g = fit.x[0], fit.x[1] * 1e-8
    np.save(f'{rb.WORK}/{season}_params.npy', [a, g])
    unit = 1 / (60e-6 * 60)
    alpha, gamma = a / unit, g / unit                       # K/h per W m^-2, K^-3 h^-1

    # -------------------------------------------------------------- 3. RESIDUAL s (what the simple model misses)
    dTdt = np.gradient(T, rb.DT)
    res = dTdt - alpha * R + gamma * T ** 4
    hour = s['Hora'].values // 100 + (s['Hora'].values % 100) / 60
    np.savez(f'{rb.WORK}/{season}_res.npz', T=T, R=R, s=res, dTdt=dTdt, hour=hour,
             RH=s.RH.values, U=s.U.values, a_h=alpha, g_h=gamma)

    # -------------------------------------------------------------- 4. TABLE V: how predictable is s?
    # A flexible regressor, 5 folds grouped by day, one input set at a time.
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict
    X = pd.DataFrame({'T': T, 'R': R, 'RH': s.RH.values, 'U': s.U.values, 'hour': hour, 'dRdt': np.gradient(R, rb.DT)})
    for k in (1, 2, 4):
        X[f'R_lag{k}h'] = pd.Series(R).shift(2 * k).values
    X['RH_lag2h'] = pd.Series(s.RH.values).shift(4).values
    ok = X.notna().all(axis=1).values & np.isfinite(res)
    day = np.arange(len(T)) // rb.NPD
    lags = ['R_lag1h', 'R_lag2h', 'R_lag4h']
    sets = {'T only': ['T'], 'T, R': ['T', 'R'], 'T, R, dR/dt': ['T', 'R', 'dRdt'],
            'T, R, lagged R': ['T', 'R'] + lags,
            'T, R, RH, wind, lags': ['T', 'R', 'RH', 'U'] + lags + ['RH_lag2h'], 'hour only': ['hour']}
    y = res[ok]
    table5 = {}
    for name, c in sets.items():
        p = cross_val_predict(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05), X.loc[ok, c].values,
                              y, groups=day[ok], cv=GroupKFold(n_splits=5))
        table5[name] = float(1 - np.mean((y - p) ** 2) / np.var(y))

    # -------------------------------------------------------------- 5. CONDITION NUMBERS
    # Large = the polynomial terms are nearly the same variable (raw T); small = standardized.
    from sklearn.preprocessing import PolynomialFeatures
    Z = np.c_[(T - T.mean()) / T.std(), (R - R.mean()) / R.std()]
    out[season] = dict(alpha=alpha, gamma=gamma, ratio=alpha / gamma, fit_rmse_K=float(np.sqrt(fit.fun)), n=len(T),
                       std_alphaR=float((alpha * R).std()), std_gammaT4=float((gamma * T ** 4).std()),
                       corr_T_T4=float(np.corrcoef(T, T ** 4)[0, 1]),
                       cond_raw=float(np.linalg.cond(PolynomialFeatures(4).fit_transform(np.c_[T, R]))),
                       cond_std=float(np.linalg.cond(PolynomialFeatures(4).fit_transform(Z))), table5=table5)
    print(f"{season}: alpha*={alpha:.4e} gamma*={gamma:.4e} fit RMSE {np.sqrt(fit.fun):.3f} K, n={len(T)}")
    print('   Table V:', {k: round(v, 2) for k, v in table5.items()})

rb.save(out, 'prepare.json')

# ------------------------------------------------------------------ 6. CHECKS against the paper
print('\nChecks')
TABLE5 = {'dry': [0.10, 0.43, 0.47, 0.51, 0.54, 0.50], 'wet': [-0.05, 0.28, 0.46, 0.45, 0.56, 0.25]}
ref = rb.load('final_equations.json', rb.REF)
for season in out:
    o = out[season]
    rb.check(f'{season} alpha*, gamma* (final_equations.json)', [o['alpha'], o['gamma'] * 1e8],
             [ref[season]['a_star'], ref[season]['g_star'] * 1e8], tol=1e-6)
    # tol 0.006: the paper rounds to 2 decimals
    rb.check(f'{season} Table V (as printed in the paper)', list(o['table5'].values()), TABLE5[season], tol=0.006)
print('   condition numbers (paper: 2.8e18/4.5e19 raw, 94/327 standardized):',
      {s: (f"{o['cond_raw']:.1e}", round(o['cond_std'])) for s, o in out.items()})
print('   std of alpha R and gamma T^4 (paper: 0.74/0.023 dry, 0.063/0.001 wet):',
      {s: (round(o['std_alphaR'], 3), round(o['std_gammaT4'], 3)) for s, o in out.items()})
