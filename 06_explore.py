"""
STEP 06 · Two alternatives tested out of year
=============================================

IN ONE LINE
    Tests whether two alternatives to the three-term equation hold up in other years.

WHAT HAPPENS
    1. "Cycle + anomaly": s = mean diurnal cycle of s + a linear model of what is left.
       It is a time-of-day forcing, not a mechanism. Does it transfer to other years?
    2. The 11-term wet-season equation from step 02 (wind, humidity, VPD).
       Does its 2008 gain hold elsewhere?
    Everything is frozen on the 2008 training block; daily rollouts as in step 05;
    differences against the three terms with the day-block bootstrap.

INPUT    spreadsheet, results/final_closure.json (step 03), results/selection.json (step 02)
OUTPUT   results/explore.json
TIME     about 1 min
RUN      python 06_explore.py
PAPER    Sec. III C (cycle + anomaly), Sec. III D (11 terms)
"""
import re

import numpy as np

import rebio as rb

EXTRA = ['R1', 'R2', 'Hr', 'RH', 'U']                 # inputs the 11-term equation needs
COLS = ('R', 'dR', 'R1', 'R2', 'Hr', 'RH', 'U', 'hour')


def cycle_models(tr, a, g, mu, sd):
    """Fit on the 2008 training block, from the central-difference residual."""
    d = tr.copy(); d[['R', 'T']] = d[['R', 'T']].interpolate()
    T, R = d['T'].values, d['R'].values; dR = np.gradient(R, rb.DT)
    s = np.gradient(T, rb.DT) - a * R + g * T ** 4
    hr = d['hour'].values
    cyc = {k: np.array([v[hr == h].mean() for h in range(rb.NPD)]) for k, v in dict(T=T, R=R, dR=dR, s=s).items()}
    Ta, Ra, dRa, sa = T - cyc['T'][hr], R - cyc['R'][hr], dR - cyc['dR'][hr], s - cyc['s'][hr]
    one = np.ones(len(T))
    b_lin = rb.lstsq(np.c_[one, Ta, Ra], sa)
    b_3 = rb.lstsq(np.c_[one, Ta, Ra, dRa], sa)
    b_s = rb.lstsq(np.c_[one, (T - mu['th']) / sd['th'], (R - mu['r']) / sd['r'], (dR - mu['dr']) / sd['dr']], s)
    base = lambda T_, x: a * x['R'] - g * T_ ** 4
    anom = lambda T_, x, b: b[0] + b[1] * (T_ - cyc['T'][x['hour']]) + b[2] * (x['R'] - cyc['R'][x['hour']])
    models = {
        'cycle only': lambda T_, x: base(T_, x) + cyc['s'][x['hour']],
        "cycle + linear (T', R')": lambda T_, x: base(T_, x) + cyc['s'][x['hour']] + anom(T_, x, b_lin),
        'cycle + 3 terms on anomalies': lambda T_, x: (base(T_, x) + cyc['s'][x['hour']] + anom(T_, x, b_3)
                                                       + b_3[3] * (x['dR'] - cyc['dR'][x['hour']])),
        '3 terms + const, FD fit (control)': lambda T_, x: (base(T_, x) + b_s[0] + b_s[1] * (T_ - mu['th']) / sd['th']
                                                            + b_s[2] * (x['R'] - mu['r']) / sd['r']
                                                            + b_s[3] * (x['dR'] - mu['dr']) / sd['dr'])}
    info = dict(b_lin=b_lin.tolist(), b_3=b_3.tolist(), b_s=b_s.tolist(),
                relax_h_lin=float(-1 / b_lin[1]) if b_lin[1] < 0 else None,
                relax_h_3=float(-1 / b_3[1]) if b_3[1] < 0 else None)
    return models, info


def eq11(season):
    """The variant-A equation from step 02, as rhs(T, x). Term names look like 'r1·v', 'u^2', '1'."""
    e = rb.load('selection.json')[season]
    sel, mu, sd, a, g = e['A weak-SSE, free'], e['mu'], e['sd'], e['a_h'], e['g_h']
    factors = []
    for name in sel['terms']:
        f = []
        for part in ([] if name == '1' else name.split('·')):
            m = re.match(r'([a-z0-9]+)(?:\^(\d+))?$', part)
            f += [m.group(1)] * int(m.group(2) or 1)
        factors.append(f)
    src = {'r': 'R', 'dr': 'dR', 'r1': 'R1', 'r2': 'R2', 'h': 'Hr', 'q': 'RH', 'u': 'U'}

    def rhs(T, x):
        z = {k: (x[v] - mu[k]) / sd[k] for k, v in src.items()}
        z['th'] = (T - mu['th']) / sd['th']; z['v'] = (rb.vpd(T, x['RH']) - mu['v']) / sd['v']
        return a * x['R'] - g * T ** 4 + sum(c * np.prod([z[b] for b in f]) for c, f in zip(sel['coef'], factors))
    return rhs, len(sel['coef'])


results = {}
for season in ('dry', 'wet'):
    e = rb.load('final_closure.json')[season]
    a, g, mu, sd = e['alpha'], e['gamma'], e['mu'], e['sd']
    d8 = rb.season_frame(2008, season, extra=True)
    n_tr = rb.n_train_steps(len(d8))
    cyc, info = cycle_models(d8.iloc[:n_tr], a, g, mu, sd)
    models = {'dry model': (rb.dry_model(a, g), None), 'SINDy 3 terms': (rb.three_terms(a, g, e['coef_std'], mu, sd), None)}
    models |= {k: (f, None) for k, f in cyc.items()}
    if season == 'wet':
        f11, k11 = eq11(season)
        models[f'SINDy {k11} terms (wind, RH, VPD)'] = (f11, EXTRA)
    results[season] = {'anomaly_fit_info': info, 'years': {}}
    for year in (2008, 2006, '2008_T22', 2010, 2009):
        df = rb.season_frame(year, season, extra=True)
        if year == 2008:
            df = df.iloc[n_tr:]
        days = rb.daily_rollouts(df, models, cols=COLS)
        res = {}
        for m, v in days.items():
            res[m] = rb.pooled(v)
            if m != 'SINDy 3 terms':
                res[m]['minus_sindy3'] = rb.boot_diff(v, days['SINDy 3 terms'], min_n=2)
        res['_n_days'] = len(days['dry model']); res['_T_height_m'] = rb.COLS[year][4]
        res['_RH_over_100_in_column'] = int((rb.sheet()[rb.COLS[year][2]] > 100).sum())
        results[season]['years'][str(year)] = res
        print(f'{season} {year}:', {m: (round(r['rmse'], 2) if r['rmse'] else None, r['diverged'])
                                    for m, r in res.items() if not m.startswith('_')}, flush=True)

rb.save(results, 'explore.json')

print('\nChecks against explore_out_of_year.json')
ref = rb.load('explore_out_of_year.json', rb.REF)
num = lambda x: np.nan if x is None else x
for s in results:
    rb.check(f'{s} anomaly-model coefficients', results[s]['anomaly_fit_info']['b_3'] + results[s]['anomaly_fit_info']['b_s'],
             ref[s]['anomaly_fit_info']['b_3'] + ref[s]['anomaly_fit_info']['b_s'], tol=1e-9)
    for y, o in results[s]['years'].items():
        r = ref[s]['years'][y]
        ms = [m for m in o if not m.startswith('_')]
        assert ms == [m for m in r if not m.startswith('_')]
        vals = lambda R, key: [num(R[m][key]) for m in ms]
        diffs = lambda R: [num((R[m].get('minus_sindy3') or {}).get('diff')) for m in ms]
        rb.check(f'{s} {y} RMSE, n, diverged, difference vs 3 terms',
                 vals(o, 'rmse') + vals(o, 'n') + vals(o, 'diverged') + diffs(o) + [o['_RH_over_100_in_column']],
                 vals(r, 'rmse') + vals(r, 'n') + vals(r, 'diverged') + diffs(r) + [r['_RH_over_100_in_column']], tol=1e-8)
