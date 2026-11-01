"""
STEP 05 · Other years, coefficients frozen (Table IV)
=====================================================

IN ONE LINE
    Applies the 2008 equations, unchanged, to 2006-2010 and compares them with the polynomial.

WHAT HAPPENS
    For each season and each year (2006, 2007, 2008 at 22 m, 2009, 2010, 2008 test block):
    1. Daily rollouts: restart every complete day from its first measured T, run 24 h.
    2. Free runs over gap-free stretches of at least 3 days.
    3. Day-block bootstrap (5000) of the difference, three terms minus polynomial.
    4. Extra: shift the reference temperature to the mean of the previous 7 days ("re-centred").
    5. Compare with the original results (check lines).

INPUT    spreadsheet, results/final_closure.json (step 03)
OUTPUT   results/validation.json   (Fig. 2 reads it)
TIME     about 1 min
RUN      python 05_validate_years.py
PAPER    Table IV, daily rollouts of Table III, Fig. 2
"""
import numpy as np
import pandas as pd

import rebio as rb

YEARS = (2006, 2007, '2008_T22', 2009, 2010, 2008)     # 2008: only the held-out test block


def summarize(day_rmse, day_bias, free):
    v = np.asarray(day_rmse, float); ok = np.isfinite(v)
    return dict(n_days=int(len(v)), n_days_diverged=int((~ok).sum()), day_rmse_median=float(np.median(v)),
                day_rmse_rms=float(np.sqrt(np.mean(v[ok] ** 2))) if ok.any() else None,
                day_bias_mean=float(np.nanmean(day_bias)), free_rmse=free['rmse'],
                free_segments=free['segments'], free_diverged=free['diverged'])


def recentred(df, models, mu_th):
    """Daily rollouts with the model state measured from the mean T of the previous 7 days."""
    days = [d for _, d in df.groupby('day')]
    res = {m: [] for m in models}
    for k, d in enumerate(days):
        if k < 7 or len(d) != rb.NPD or d[['R', 'T']].isna().any().any():
            continue
        prev = pd.concat(days[k - 7:k])['T']
        if prev.notna().sum() < 3 * rb.NPD:
            continue
        shift = prev.mean() - mu_th
        R, T = d.R.values, d['T'].values
        for m, f in models.items():
            Ts = rb.rk4(T[0], {'R': R, 'dR': np.gradient(R, rb.DT)}, lambda x, F, f=f: f(x - shift, F))
            res[m].append(rb.rmse(Ts, T) if np.all(np.isfinite(Ts)) else np.inf)
    out = {'n_days': len(res['SINDy 3 terms'])}
    for m, v in res.items():
        p = rb.pooled(v); out[m] = dict(rmse=p['rmse'], diverged=p['diverged'])
    return out


results = {}
for season in ('dry', 'wet'):
    e = rb.load('final_closure.json')[season]
    a, g, mu, sd = e['alpha'], e['gamma'], e['mu'], e['sd']
    d8 = rb.season_frame(2008, season)
    n_tr = rb.n_train_steps(len(d8))
    assert abs(d8['T'].iloc[:n_tr].mean() - mu['th']) < 0.05, 'training block does not match step 03'
    tr = d8.iloc[:n_tr][['R', 'T']].interpolate()
    xi = rb.fit_polynomial(tr['T'].values, tr.R.values, a, g, mu, sd)
    models = {'dry model': rb.dry_model(a, g), 'SINDy 3 terms': rb.three_terms(a, g, e['coef_std'], mu, sd),
              'polynomial 15 terms': rb.polynomial(a, g, xi, mu, sd)}
    results[season] = {'poly_coefs_std': xi.tolist(), 'years': {}}
    for year in YEARS:
        df = rb.season_frame(year, season)
        if year == 2008:
            df = df.iloc[n_tr:]
        day_rmse, day_bias = rb.daily_rollouts(df, {m: (f, None) for m, f in models.items()}, with_bias=True)
        res = {m: summarize(day_rmse[m], day_bias[m], rb.free_runs(df, f)) for m, f in models.items()}
        res.update(coverage_RT=float(df[['R', 'T']].notna().all(axis=1).mean()), T_height_m=rb.COLS[year][4],
                   mean_T_K=float(df['T'].mean()), mean_R=float(df.R.mean()),
                   sindy_minus_poly_daily=rb.boot_diff(day_rmse['SINDy 3 terms'], day_rmse['polynomial 15 terms']),
                   sindy_minus_dry_daily=rb.boot_diff(day_rmse['SINDy 3 terms'], day_rmse['dry model']))
        if year != 2008:
            res['recentred_7day'] = recentred(df, {k: models[k] for k in ('SINDy 3 terms', 'polynomial 15 terms')}, mu['th'])
        results[season]['years'][str(year)] = res
        bd = res['sindy_minus_poly_daily']     # wet 2009: the polynomial survives one day only
        print(f"{season} {year}: daily RMSE dry {res['dry model']['day_rmse_rms']:.2f}, "
              f"3 terms {res['SINDy 3 terms']['day_rmse_rms']:.2f}, polynomial {res['polynomial 15 terms']['day_rmse_rms']:.2f} K "
              f"(polynomial diverged {res['polynomial 15 terms']['n_days_diverged']} of {res['polynomial 15 terms']['n_days']} days); "
              f"3 terms - polynomial {bd['diff']:+.2f} [{bd['ci95'][0]:+.2f}, {bd['ci95'][1]:+.2f}] K", flush=True)

rb.save(results, 'validation.json')

print('\nChecks against results.json and checks_results.json')
ref, ref_c = rb.load('results.json', rb.REF), rb.load('checks_results.json', rb.REF)
KEYS = ('n_days', 'n_days_diverged', 'day_rmse_median', 'day_rmse_rms', 'day_bias_mean', 'free_rmse',
        'free_segments', 'free_diverged')
num = lambda x: np.nan if x is None else x
for s in results:
    rb.check(f'{s} polynomial coefficients', results[s]['poly_coefs_std'], ref[s]['poly_coefs_std'], tol=1e-9)
    for y in map(str, YEARS):
        o, r = results[s]['years'][y], ref[s]['years'][y]
        rb.check(f'{s} {y} all model scores',
                 [num(o[m][k]) for m in ('dry model', 'SINDy 3 terms', 'polynomial 15 terms') for k in KEYS],
                 [num(r[m][k]) for m in ('dry model', 'SINDy 3 terms', 'polynomial 15 terms') for k in KEYS], tol=1e-8)
        rb.check(f'{s} {y} bootstrap differences',
                 [o[k]['diff'] for k in ('sindy_minus_poly_daily', 'sindy_minus_dry_daily')]
                 + o['sindy_minus_poly_daily']['ci95'] + o['sindy_minus_dry_daily']['ci95'],
                 [r[k]['diff'] for k in ('sindy_minus_poly_daily', 'sindy_minus_dry_daily')]
                 + r['sindy_minus_poly_daily']['ci95'] + r['sindy_minus_dry_daily']['ci95'], tol=1e-8)
        if y != '2008':
            o2, r2 = o['recentred_7day'], ref_c[f'{s}_{y}_recentred']
            rb.check(f'{s} {y} re-centred', [o2['n_days'], num(o2['SINDy 3 terms']['rmse']), o2['SINDy 3 terms']['diverged'],
                                             num(o2['polynomial 15 terms']['rmse']), o2['polynomial 15 terms']['diverged']],
                     [r2['n_days'], num(r2['sindy_rms']), r2['sindy_div'], num(r2['poly_rms']), r2['poly_div']], tol=1e-8)
