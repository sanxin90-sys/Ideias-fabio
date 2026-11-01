"""
07 · expC_real.py · Neural networks on the real 2008 data
=========================================================

IN ONE LINE
    How well do MLP, DeepONet and FNO predict s, and does that improve the temperature?

WHAT HAPPENS
    1. Split the days into 5 folds (blocks of whole days).
    2. For each fold: train each network with 3 seeds on the other days.
    3. Score: R^2 of s on the held-out days, and RMSE of T after a 1-day rollout.

    Main finding: the FNO explains more of s, but T does not improve.

INPUT    ../work/{dry,wet}_res.npz (step 01)
OUTPUT   expC_results.json (compare.py checks it against ../reference/)
TIME     about 6 min
RUN      cd 07_diagnostics ; python expC_real.py
PAPER    Sec. III B (pointwise MLP), Sec. III E
NEEDS    jax, optax
"""
import numpy as np, json, sys
from ops import *

SEEDS = [0, 1, 2]
out = {}
for season in ['dry', 'wet']:
    d = np.load(f'../work/{season}_res.npz')
    T, R, s, a_h, g_h = d['T'], d['R'], d['s'], float(d['a_h']), float(d['g_h'])
    nd = len(T) // NPD
    days = np.arange(1, nd)
    folds = np.array_split(days, 5)
    res = {m: {'s_true': [], 's_hat': [], 'rmse': []} for m in list(MODELS) + ['physics']}
    for k, te in enumerate(folds):
        trd = np.setdiff1d(days, te)
        rng = np.random.default_rng(k)
        vad = rng.choice(trd, size=max(2, len(trd) // 7), replace=False)
        trd = np.setdiff1d(trd, vad)
        rtr, rva, rte = make_days(T, R, s, trd), make_days(T, R, s, vad), make_days(T, R, s, te)
        st = stats_of(rtr)
        btr, bva, bte = (standardize(x, st) for x in (rtr, rva, rte))
        Tp = rollout_physics(rte, a_h, g_h, s_const=st[4])
        res['physics']['rmse'].append(np.sqrt(np.mean((Tp - rte['T_obs']) ** 2)))
        res['physics']['s_true'].append(rte['s']); res['physics']['s_hat'].append(np.full_like(rte['s'], st[4]))
        for m in MODELS:
            preds, tms = [], []
            for sd in SEEDS:
                p, ap = fit(m, btr, bva, seed=sd)
                preds.append(np.asarray(ap(p, bte)) * st[5] + st[4])
                tms.append(rollout(m, p, ap, rte, st, a_h, g_h))
            res[m]['s_true'].append(rte['s']); res[m]['s_hat'].append(np.mean(preds, 0))
            Tm = np.mean(tms, 0)
            res[m]['rmse'].append(np.sqrt(np.mean((Tm - rte['T_obs']) ** 2)))
            if k == 0:
                print(season, m, 'params:', n_params(p), flush=True)
        print(season, 'fold', k, {m: round(float(res[m]['rmse'][-1]), 3) for m in res}, flush=True)
    out[season] = {}
    for m in res:
        y = np.concatenate(res[m]['s_true']).ravel(); yh = np.concatenate(res[m]['s_hat']).ravel()
        out[season][m] = {'R2_s': float(r2(y, yh)), 'rollout_RMSE_K': float(np.mean(res[m]['rmse'])),
                          'rollout_RMSE_sd': float(np.std(res[m]['rmse']))}
    print(json.dumps(out[season], indent=1), flush=True)
json.dump(out, open('expC_results.json', 'w'), indent=1)
