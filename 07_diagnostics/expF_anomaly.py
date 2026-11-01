"""
07 · expF_anomaly.py · Is anything predictable beyond the daily cycle?
=====================================================================

IN ONE LINE
    Removes the mean daily cycle from s and asks how much of what is left any model can predict.

WHAT HAPPENS
    For each of 5 folds of whole days (same folds as expC):
    1. Mean daily cycle of T, R and s from the training days only.
    2. Anomaly s' = s - cycle(hour of day).
    3. Predict s' with the 4 networks and 3 linear models (plain, memory kernel, radiation reservoirs).
    4. Score: R^2 of s', R^2 of total s, RMSE of T after a 1-day rollout.

    Finding: in the dry season no model explains more than 7% of the anomaly.

INPUT    ../work/{dry,wet}_res.npz (step 01)
OUTPUT   expF_results.json (compare.py checks it)
TIME     about 6 min
RUN      cd 07_diagnostics ; python expF_anomaly.py
PAPER    Sec. III E
NEEDS    jax, optax
"""
import numpy as np, json
from ops import *

SEEDS = [0, 1, 2]
NEURAL = list(MODELS)
LINEAR = ['linear (T\', R\')', 'kernel R\' 0-24 h', 'storage H 0.9 h + 6 h']
NL, DT = 49, 0.5


def storage(R, tau, dt=DT):
    H = np.empty_like(R); H[0] = 0.0; w = np.exp(-dt / tau)
    for i in range(1, len(R)):
        H[i] = w * H[i - 1] + (1 - w) * R[i - 1]
    return H


def lin_features(name, Ta, Ra):
    """Columns: [1, T', exogenous...]. T' is column 1 so rollouts can feed back simulated T."""
    n = len(Ta)
    if name.startswith('linear'):
        ex = Ra[:, None]
    elif name.startswith('kernel'):
        ex = np.zeros((n, NL))
        for k in range(NL):
            ex[k:, k] = Ra[:n - k] * DT
    else:
        ex = np.c_[Ra, storage(Ra, 0.9), storage(Ra, 6.0)]
    return np.c_[np.ones(n), Ta, ex]


def lin_fit(name, X, y, lam):
    P = np.zeros((X.shape[1], X.shape[1]))
    if name.startswith('kernel'):
        D = np.diff(np.eye(NL), 2, axis=0)
        P[2:, 2:] = lam * D.T @ D + 1e-6 * np.eye(NL)
    return np.linalg.solve(X.T @ X + P + 1e-9 * np.eye(X.shape[1]), X.T @ y)


def rollout_anom(Rday, T0, cycT, cycs, sfun, a_h, g_h, dt=DT):
    """dT/dt = a R - g T^4 + cyc_s(hour) + s'(i, T); R held over each step."""
    B = Rday.shape[0]
    Tm = np.zeros((B, NPD)); Tm[:, 0] = T0
    for i in range(1, NPD):
        r = Rday[:, i - 1]
        f = lambda T: a_h * r - g_h * T ** 4 + cycs[i - 1] + sfun(i - 1, T)
        x = Tm[:, i - 1]
        k1 = f(x); k2 = f(x + dt * k1 / 2); k3 = f(x + dt * k2 / 2); k4 = f(x + dt * k3)
        Tm[:, i] = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return Tm


out = {}
for season in ['dry', 'wet']:
    d = np.load(f'../work/{season}_res.npz')
    T, R, s, a_h, g_h = d['T'], d['R'], d['s'], float(d['a_h']), float(d['g_h'])
    nd = len(T) // NPD
    hour = np.arange(len(T)) % NPD
    days = np.arange(1, nd)
    folds = np.array_split(days, 5)
    names = ['cycle only'] + NEURAL + LINEAR
    res = {m: {'sa': [], 'sa_hat': [], 's': [], 's_hat': [], 'rmse': []} for m in names}
    for k, te in enumerate(folds):
        trd = np.setdiff1d(days, te)
        rng = np.random.default_rng(k)
        vad = rng.choice(trd, size=max(2, len(trd) // 7), replace=False)
        fitd = np.setdiff1d(trd, vad)
        # mean diurnal cycle from training + validation days only (never test days)
        M = lambda x: x[:nd * NPD].reshape(nd, NPD)[trd].mean(0)
        cycT, cycR, cycs = M(T), M(R), M(s)
        Ta, Ra, sa = T - cycT[hour], R - cycR[hour], s - cycs[hour]
        rte_raw = make_days(T, R, s, te)
        Rday, T0, s_te = rte_raw['R_day'], rte_raw['T_obs'][:, 0], rte_raw['s']
        sa_te = s_te - cycs[None, :]
        add = lambda m, sa_hat, Tm: (res[m]['sa'].append(sa_te), res[m]['sa_hat'].append(sa_hat),
                                     res[m]['s'].append(s_te), res[m]['s_hat'].append(sa_hat + cycs[None, :]),
                                     res[m]['rmse'].append(np.sqrt(np.mean((Tm - rte_raw['T_obs']) ** 2))))
        # cycle only
        Tm = rollout_anom(Rday, T0, cycT, cycs, lambda i, Tx: 0.0, a_h, g_h)
        add('cycle only', np.zeros_like(sa_te), Tm)
        # neural models on anomalies
        rtr, rva, rte = make_days(Ta, Ra, sa, fitd), make_days(Ta, Ra, sa, vad), make_days(Ta, Ra, sa, te)
        st = stats_of(rtr); mT, sT, mR, sR, ms, ss = st
        btr, bva, bte = (standardize(x, st) for x in (rtr, rva, rte))
        for m in NEURAL:
            preds, tms = [], []
            for sd in SEEDS:
                p, ap = fit(m, btr, bva, seed=sd)
                if m in ('DeepONet', 'FNO'):
                    S = np.asarray(ap(p, bte)) * ss + ms
                    sfun = lambda i, Tx, S=S: S[:, i]
                else:
                    def sfun(i, Tx, p=p, ap=ap):
                        b = {'T_day': ((Tx - cycT[i] - mT) / sT)[:, None],
                             'R_day': ((rte['R_day'][:, i] - mR) / sR)[:, None],
                             'R_lags': ((rte['R_lags'][:, i] - mR) / sR)[:, None, :]}
                        return np.asarray(ap(p, b))[:, 0] * ss + ms
                preds.append(np.asarray(ap(p, bte)) * ss + ms)
                tms.append(rollout_anom(Rday, T0, cycT, cycs, sfun, a_h, g_h))
            add(m, np.mean(preds, 0), np.mean(tms, 0))
        # linear models on anomalies (penalty for the kernel chosen on validation days)
        idx = lambda dd: (dd[:, None] * NPD + np.arange(NPD)).ravel()
        for m in LINEAR:
            X = lin_features(m, Ta, Ra)
            best = None
            for lam in (10.0 ** np.arange(-2, 8) if m.startswith('kernel') else [0.0]):
                b = lin_fit(m, X[idx(fitd)], sa[idx(fitd)], lam)
                v = np.mean((X[idx(vad)] @ b - sa[idx(vad)]) ** 2)
                if best is None or v < best[0]:
                    best = (v, lam)
            b = lin_fit(m, X[idx(trd)], sa[idx(trd)], best[1])
            Xte = X[idx(te)].reshape(len(te), NPD, -1)
            exo = Xte @ b - b[1] * Xte[:, :, 1]          # everything except the T' term
            sfun = lambda i, Tx, exo=exo, b=b: exo[:, i] + b[1] * (Tx - cycT[i])
            add(m, Xte @ b, rollout_anom(Rday, T0, cycT, cycs, sfun, a_h, g_h))
        print(season, 'fold', k, {m: round(float(res[m]['rmse'][-1]), 3) for m in names}, flush=True)
    out[season] = {}
    for m in names:
        cat = lambda key: np.concatenate(res[m][key]).ravel()
        out[season][m] = {'R2_anomaly': round(float(r2(cat('sa'), cat('sa_hat'))), 3),
                          'R2_total_s': round(float(r2(cat('s'), cat('s_hat'))), 3),
                          'rollout_RMSE_K': round(float(np.mean(res[m]['rmse'])), 3),
                          'rollout_RMSE_sd': round(float(np.std(res[m]['rmse'])), 3)}
    print(json.dumps(out[season], indent=1), flush=True)
json.dump(out, open('expF_results.json', 'w'), indent=1)
