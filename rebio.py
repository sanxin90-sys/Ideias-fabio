"""
REBIO · the toolbox
===================

IN ONE LINE
    Every function the steps share. This file computes nothing by itself: the steps import it.

WHAT IS INSIDE (8 sections, top to bottom)
    1. Paths and constants   where files live; 30-min step; season months; data splits
    2. Tower data            read one season of one year from the spreadsheet
    3. 2008 data             load the files made by step 01
    4. Weak form             the core of the method: an equation for T without differentiating T
    5. Models                the equations: no correction, three terms, 15-term polynomial
    6. Integration, metrics  RK4 (run a model forward in time), daily rollouts, RMSE
    7. Bootstrap             uncertainty by resampling whole days
    8. 58-term library       the candidate terms searched in step 02

UNITS    time in hours, T in K, R in W m^-2, dT/dt in K/h.
READ IT  sections 4 (weak form) and 5 (models) are the heart of the paper.
"""
import json
import os
from itertools import combinations_with_replacement

import numpy as np

np.seterr(over='ignore', invalid='ignore')   # diverging runs overflow; rk4 flags them via T_BOUNDS

# ================================================================== 1. PATHS AND CONSTANTS
# Where files live and the fixed settings every step uses.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')          # spreadsheet goes here (not distributed)
WORK = os.path.join(HERE, 'work')          # intermediate files made by the scripts
OUT = os.path.join(HERE, 'results')        # JSON results and figures
REF = os.path.join(HERE, 'reference')      # original outputs, used only by check()
XLSX = os.path.join(DATA, 'Dados Rebio Jaru 2006 a 2008.xlsx')
for _d in (WORK, OUT):
    os.makedirs(_d, exist_ok=True)

DT = 0.5          # sampling step (h)
NPD = 48          # samples per day
SEASONS = {'dry': (7, 8, 9), 'wet': (1, 2, 3)}
TRAIN = 0.70      # first 70% of whole days = fit + validation; last 30% = test
FIT = 0.56        # first 56% = fit; 56-70% = validation (term selection only)
T_BOUNDS = (270.0, 330.0)   # a simulation leaving this range counts as diverged


def save(obj, name):
    json.dump(obj, open(os.path.join(OUT, name), 'w'), indent=1, default=float)


def load(name, where=OUT):
    return json.load(open(os.path.join(where, name)))


def check(label, new, ref, tol=1e-6):
    """Print whether a new number equals the original one (absolute tolerance tol)."""
    new, ref = np.asarray(new, float), np.asarray(ref, float)
    ok = new.shape == ref.shape and np.allclose(new, ref, atol=tol, rtol=0, equal_nan=True)
    print(f'  check {label:55s} {"same" if ok else "DIFFERENT"}')
    if not ok:
        print('     new:', np.round(new, 6).tolist(), '\n     ref:', np.round(ref, 6).tolist())
    return ok


# ================================================================== 2. TOWER DATA
# Read one season of one year from the spreadsheet, with the calendar fixed and gaps handled.
# Columns of sheet "DADOS OK" for each year: net radiation, air temperature, humidity, wind, height of T.
COLS = {2006: ('R_liq.2', 'T_ar_22m.1', 'UR_62m.2', 'VV_ane_62m.2', 22),
        2007: ('R_liq.3', 'T_ar_22m.2', 'UR_62m.3', 'VV_ane_62m.3', 22),
        2008: ('R_liq.4', 'T_ar_62m.1', 'UR_62m.4', 'VV_ane_62m.4', 62),
        '2008_T22': ('R_liq.4', 'T_ar_22m.3', 'UR_62m.4', 'VV_ane_62m.4', 22),
        2009: ('R_liq filtro2', 'T_ar_62m.2', 'UR_62m.5', 'VV_ane_62m.5', 62),
        2010: ('R_liq filtro2.1', 'T_ar_62m.3', 'UR_62m.6', 'VV_ane_62m.6', 62)}
YEARCOL = {2006: 'Ano.2', 2007: 'Ano.3', 2008: 'Ano.4', '2008_T22': 'Ano.4', 2009: 'Ano.5', 2010: 'Ano.6'}
TAU_H = 0.9       # time constant of the radiation reservoir dH/dt = (R - H)/tau (h)
_SHEET = None


def sheet():
    global _SHEET
    if _SHEET is None:
        import pandas as pd
        _SHEET = pd.read_excel(XLSX, 'DADOS OK')
    return _SHEET


def vpd(Tk, RH):
    """Vapour-pressure deficit (kPa) from temperature (K) and relative humidity (%)."""
    Tc = Tk - 273.15
    return 0.6108 * np.exp(17.27 * Tc / (Tc + 237.3)) * (1 - RH / 100)


def season_frame(year, season, extra=False):
    """One season of one year on the true calendar (the sheet's month/day columns follow a leap year).

    R and T: gaps up to 1 h interpolated, longer gaps left as NaN.
    extra=True adds humidity and wind (all internal gaps filled, RH capped at 100%), dR/dt,
    R lagged 1 h and 2 h, the radiation reservoir H, and the half-hour index of the day.
    """
    import pandas as pd
    raw = sheet()
    y = 2008 if year == '2008_T22' else year
    rc, tc, qc, uc, _ = COLS[year]
    n = 366 * NPD if y % 4 == 0 else 365 * NPD
    t = pd.date_range(f'{y}-01-01 00:30', periods=n, freq='30min')
    assert (raw[YEARCOL[year]].iloc[:n].dropna() == y).mean() > 0.999, 'year column mismatch'
    df = pd.DataFrame({'R': raw[rc].iloc[:n].values, 'T': raw[tc].iloc[:n].values + 273.15}, index=t)
    if extra:
        df['RH'], df['U'] = raw[qc].iloc[:n].values, raw[uc].iloc[:n].values
    start = df.index - pd.Timedelta('30min')            # row k closes the k-th half hour
    df['day'] = start.normalize()
    if extra:
        df['hour'] = start.hour * 2 + start.minute // 30
    df = df[df.day.dt.month.isin(SEASONS[season])].copy()
    for c in ('R', 'T'):
        df[c] = df[c].interpolate(limit=2, limit_area='inside')
    assert df['T'].dropna().between(278.15, 318.15).all(), 'T outside 5-45 degC'
    assert df['R'].dropna().between(-200, 1400).all(), 'R outside -200..1400 W m-2'
    if extra:
        for c in ('RH', 'U'):
            df[c] = df[c].interpolate(limit_area='inside')
        df['RH'] = df['RH'].clip(upper=100)
        R = df['R'].values
        df['dR'] = np.gradient(R, DT)
        df['R1'] = np.r_[np.full(2, np.nan), R[:-2]]
        df['R2'] = np.r_[np.full(4, np.nan), R[:-4]]
        Hr = np.empty_like(R); Hr[0] = R[0]; w = 1 - np.exp(-DT / TAU_H)
        for i in range(1, len(R)):
            ok = np.isfinite(Hr[i - 1]) and np.isfinite(R[i - 1])
            Hr[i] = Hr[i - 1] + w * (R[i - 1] - Hr[i - 1]) if ok else (R[i] if np.isfinite(R[i]) else np.nan)
        df['Hr'] = Hr
    return df


def n_train_steps(n):
    """Number of steps in the training block (first 70% of whole days)."""
    return int(TRAIN * (n // NPD)) * NPD


# ================================================================== 3. 2008 DATA FOR FITTING
# Load the 2008 series and residual saved by step 01; standardize variables.
def ident(season, trim=True):
    """2008 series used to identify the closures (made by step 01: work/{season}_res.npz).

    trim=True drops the first 2 h, as the term-selection code did (it needs 2 h of lagged R);
    the final three-term fit and every 2008 number in the paper use the trimmed series.
    Returns a dict with T, R, dR (central difference), RH, U, s (residual), hour, alpha, gamma.
    """
    d = np.load(os.path.join(WORK, f'{season}_res.npz'))
    k = 4 if trim else 0
    out = {key: d[key][k:] for key in ('T', 'R', 's', 'hour', 'RH', 'U')}
    out['dR'] = np.gradient(d['R'], DT)[k:]
    out['alpha'], out['gamma'] = float(d['a_h']), float(d['g_h'])
    return out


def standardizer(x, tr):
    """(mean, sd) of x over the slice tr, and the standardized series."""
    mu, sd = float(x[tr].mean()), float(x[tr].std())
    return mu, sd, (x - mu) / sd


# ================================================================== 4. WEAK FORM  (the heart of the method)
# Turns "dT/dt = model" into a linear system G xi = b WITHOUT differentiating the measured T.
def weak_system(T, R, Theta, alpha, gamma, half_window=4.0, stride=1, p=4):
    """Weak SINDy linear system G xi = b.

    For each window centre c, multiply dT/dt = alpha R - gamma T^4 + Theta xi by a bump phi that vanishes
    at the window ends and integrate. Integration by parts gives  int phi dT/dt = -int phi' T,
    so the measured T is never differentiated. phi(u) = (1 - u^2)^p on an 8 h window, trapezoid rule.
    """
    m = int(round(half_window / DT))
    u = np.linspace(-1, 1, 2 * m + 1)
    phi = (1 - u ** 2) ** p
    dphi = -2 * p * u * (1 - u ** 2) ** (p - 1) / (m * DT)
    w = np.full(2 * m + 1, DT); w[[0, -1]] = DT / 2
    G, b = [], []
    for c in range(m, len(T) - m, stride):
        s = slice(c - m, c + m + 1)
        b.append(-np.sum(w * dphi * T[s]) - np.sum(w * phi * (alpha * R[s] - gamma * T[s] ** 4)))
        G.append((w * phi) @ Theta[s])
    return np.array(G), np.array(b)


def lstsq(A, y):
    return np.linalg.lstsq(A, y, rcond=None)[0]


# ================================================================== 5. MODELS
# Each model is a function rhs(T, forcing) that returns dT/dt.
# Every model is rhs(T, f) -> dT/dt, where f is a dict of forcing values at that instant ('R', 'dR', ...).
PIJ = [(i, j) for i in range(5) for j in range(5 - i)]   # the 15 monomials theta^i r^j of degree <= 4


def dry_model(alpha, gamma):
    return lambda T, f: alpha * f['R'] - gamma * T ** 4


def three_terms(alpha, gamma, c, mu, sd):
    """dT/dt = alpha R - gamma T^4 + c_T theta + c_Rdot rdot + c_R r  (c, mu, sd dicts keyed th, dr, r)."""
    def rhs(T, f):
        return (alpha * f['R'] - gamma * T ** 4 + c['th'] * (T - mu['th']) / sd['th']
                + c['dr'] * (f['dR'] - mu['dr']) / sd['dr'] + c['r'] * (f['R'] - mu['r']) / sd['r'])
    return rhs


def poly_features(th, r):
    return np.column_stack([th ** i * r ** j for i, j in PIJ])


def fit_polynomial(T, R, alpha, gamma, mu, sd, rows=slice(None)):
    """15-term comparator: OLS of the central-difference residual on theta^i r^j (form of Eq. 8 of the 2026 paper).

    The derivative is taken on the arrays given, then only `rows` enter the fit.
    """
    y = np.gradient(T, DT) - alpha * R + gamma * T ** 4
    X = poly_features((T - mu['th']) / sd['th'], (R - mu['r']) / sd['r'])
    return lstsq(X[rows], y[rows])


def polynomial(alpha, gamma, xi, mu, sd):
    def rhs(T, f):
        th, r = (T - mu['th']) / sd['th'], (f['R'] - mu['r']) / sd['r']
        return alpha * f['R'] - gamma * T ** 4 + sum(c * th ** i * r ** j for c, (i, j) in zip(xi, PIJ))
    return rhs


def from_final(season, kind, poly_xi=None):
    """Model built from results/final_closure.json (step 03)."""
    e = load('final_closure.json')[season]
    a, g, mu, sd = e['alpha'], e['gamma'], e['mu'], e['sd']
    if kind == 'dry':
        return dry_model(a, g)
    if kind == 'three':
        return three_terms(a, g, e['coef_std'], mu, sd)
    return polynomial(a, g, poly_xi, mu, sd)


def stable_fixed_points(rhs, T_mean, forcing_mean):
    """Roots of rhs(T, mean forcing) within T_mean +- 20 K where d(rhs)/dT < 0."""
    grid = np.linspace(T_mean - 20, T_mean + 20, 801)
    f = np.array([rhs(t, forcing_mean) for t in grid])
    roots = [grid[i] for i in range(len(f) - 1) if f[i] * f[i + 1] < 0]
    return [float(t) for t in roots if (rhs(t + 1e-3, forcing_mean) - rhs(t - 1e-3, forcing_mean)) / 2e-3 < 0]


# ================================================================== 6. RUN A MODEL IN TIME, AND MEASURE ERROR
# RK4 integration, daily rollouts, free runs, RMSE.
def rk4(T0, forcing, rhs, n_sub=6):
    """Integrate dT/dt = rhs(T, f) with classical RK4, n_sub sub-steps per half hour.

    forcing: dict of equal-length arrays; values are linearly interpolated inside each step
    (non-float entries such as 'hour' are held constant). Returns T at the sampling instants,
    NaN from the first step that leaves T_BOUNDS.
    """
    keys = list(forcing)
    n = len(forcing[keys[0]])
    T = np.full(n, np.nan); T[0] = x = T0; h = DT / n_sub
    for i in range(n - 1):
        lo = {k: forcing[k][i] for k in keys}; hi = {k: forcing[k][i + 1] for k in keys}
        for j in range(n_sub):
            fs = [{k: lo[k] + q * (hi[k] - lo[k]) if k != 'hour' else lo[k] for k in keys}
                  for q in (j / n_sub, (j + .5) / n_sub, (j + 1) / n_sub)]
            k1 = rhs(x, fs[0]); k2 = rhs(x + h * k1 / 2, fs[1]); k3 = rhs(x + h * k2 / 2, fs[1]); k4 = rhs(x + h * k3, fs[2])
            x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not (np.isfinite(x) and T_BOUNDS[0] < x < T_BOUNDS[1]):
            return T
        T[i + 1] = x
    return T


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def daily_rollouts(df, models, cols=('R',), need=('R', 'T'), with_bias=False):
    """Restart every complete day from its first observed T; return {model: [day RMSE]}.

    models: {name: (rhs, req)}; req lists extra columns the model needs (days missing them are NaN).
    A diverged day is inf. `cols`: forcing columns passed to the models; dR/dt is recomputed per day
    by central difference unless 'dR' is listed in cols. with_bias=True also returns {model: [day mean error]}.
    """
    out = {m: [] for m in models}; bias = {m: [] for m in models}
    for _, d in df.groupby('day'):
        if len(d) != NPD or d[list(need)].isna().any().any():
            continue
        T = d['T'].values
        forcing = {c: d[c].values for c in cols}
        if 'dR' not in cols:
            forcing['dR'] = np.gradient(d['R'].values, DT)
        for m, (rhs, req) in models.items():
            if req and d[list(req)].isna().any().any():
                out[m].append(np.nan); bias[m].append(np.nan); continue
            Ts = rk4(T[0], forcing, rhs)
            ok = np.all(np.isfinite(Ts))
            out[m].append(rmse(Ts, T) if ok else np.inf)
            bias[m].append(float(np.mean(Ts - T)) if ok else np.nan)
    return (out, bias) if with_bias else out


def pooled(day_rmse):
    """Pooled daily RMSE over finite days, and counts of diverged (inf) and skipped (NaN) days."""
    v = np.asarray(day_rmse, float); ok = np.isfinite(v)
    return dict(rmse=float(np.sqrt(np.mean(v[ok] ** 2))) if ok.any() else None,
                n=int(ok.sum()), diverged=int(np.isinf(v).sum()), skipped=int(np.isnan(v).sum()))


def free_runs(df, rhs, min_days=3):
    """Continuous runs over gap-free segments of at least min_days; pooled RMSE over all segments."""
    ok = df[["R", "T"]].notna().all(axis=1).values
    segs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        if (not v or i == len(ok) - 1) and start is not None:
            end = i + 1 if v else i
            if end - start >= min_days * NPD:
                segs.append((start, end))
            start = None
    se = n = div = 0
    for a, b in segs:
        R, T = df.R.values[a:b], df['T'].values[a:b]
        Ts = rk4(T[0], {'R': R, 'dR': np.gradient(R, DT)}, rhs)
        if not np.all(np.isfinite(Ts)):
            div += 1; continue
        se += float(np.sum((Ts - T) ** 2)); n += len(T)
    return dict(rmse=float(np.sqrt(se / n)) if n else None, segments=len(segs), diverged=div)


# ================================================================== 7. UNCERTAINTY (DAY-BLOCK BOOTSTRAP)
# Resample whole days many times; the spread of the results gives the 95% interval.
def boot_diff(a, b, n_boot=5000, seed=1, min_n=1):
    """Difference in pooled daily RMSE (a - b) and its 95% interval, resampling days (both finite).

    Returns None when fewer than min_n days have both models finite.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < min_n:
        return None
    idx = np.random.default_rng(seed).integers(0, len(a), (n_boot, len(a)))
    d = np.sqrt(np.mean(a[idx] ** 2, 1)) - np.sqrt(np.mean(b[idx] ** 2, 1))
    return dict(diff=float(np.sqrt(np.mean(a ** 2)) - np.sqrt(np.mean(b ** 2))),
                ci95=np.percentile(d, [2.5, 97.5]).tolist(), n=int(len(a)), frac_a_better=float(np.mean(a < b)))


def day_groups(n_rows, half_window=4.0):
    """Indices of weak-form rows (one per window centre, stride 1) grouped by day."""
    m = int(round(half_window / DT)); day = (np.arange(n_rows) + m) // NPD
    return [np.flatnonzero(day == k) for k in np.unique(day)]


def boot_coefs(G, b, rng, n_boot=1000):
    """Day-block bootstrap of the weak-form least-squares coefficients."""
    groups = day_groups(len(b))
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        out.append(lstsq(G[idx], b[idx]))
    return np.array(out)


# ================================================================== 8. THE 58 CANDIDATE TERMS (step 02)
# 1 constant + 9 variables + 45 products of two + 3 cubic terms = 58.
BASE = ['th', 'r', 'dr', 'r1', 'r2', 'h', 'q', 'u', 'v']   # T, R, dR/dt, R(t-1h), R(t-2h), reservoir, RH, wind, VPD
TERMS = ([()] + [(v,) for v in BASE] + list(combinations_with_replacement(BASE, 2))
         + [('th', 'th', 'th'), ('th', 'th', 'r'), ('th', 'r', 'r')])      # 1 + 9 + 45 + 3 = 58


def term_name(t):
    if not t:
        return '1'
    return '·'.join(v if t.count(v) == 1 else f'{v}^{t.count(v)}' for v in dict.fromkeys(t))


class Library:
    """2008 identification data for one season with the 58 standardized candidate functions.

    VPD is recomputed from the simulated T during integration, so it is a state-dependent term.
    """
    def __init__(self, season):
        d = ident(season)
        full = np.load(os.path.join(WORK, f'{season}_res.npz'))
        R_full = full['R']
        RH, U = _fill(full['RH'])[4:], _fill(full['U'])[4:]      # gaps filled on the full series, then trimmed
        R1 = np.r_[np.full(2, np.nan), R_full[:-2]][4:]; R2 = np.r_[np.full(4, np.nan), R_full[:-4]][4:]
        Hr = np.empty_like(R_full); Hr[0] = R_full[0]; w = 1 - np.exp(-DT / TAU_H)
        for i in range(1, len(R_full)):
            Hr[i] = Hr[i - 1] + w * (R_full[i - 1] - Hr[i - 1])
        self.T, self.R, self.a, self.g = d['T'], d['R'], d['alpha'], d['gamma']
        self.n = len(self.T); nd = self.n // NPD
        self.i_fit, self.i_tr = int(FIT * nd) * NPD, int(TRAIN * nd) * NPD
        src = dict(th=self.T, r=self.R, dr=d['dR'], r1=R1, r2=R2, h=Hr[4:], q=RH, u=U, v=vpd(self.T, RH))
        tr = slice(0, self.i_tr)
        self.mu = {k: float(src[k][tr].mean()) for k in BASE}
        self.sd = {k: float(src[k][tr].std()) for k in BASE}
        self.Z = {k: (src[k] - self.mu[k]) / self.sd[k] for k in BASE}
        self.Theta = np.column_stack([np.prod([self.Z[v] for v in t], axis=0) if t else np.ones(self.n) for t in TERMS])
        self.forcing = dict(R=self.R, RH=RH, r=self.Z['r'], dr=self.Z['dr'], r1=self.Z['r1'], r2=self.Z['r2'],
                            q=self.Z['q'], u=self.Z['u'], h=self.Z['h'])

    def weak(self, lo, hi, stride=2):
        return weak_system(self.T[lo:hi], self.R[lo:hi], self.Theta[lo:hi], self.a, self.g, stride=stride)

    def rhs(self, support, xi):
        mu, sd, a, g = self.mu, self.sd, self.a, self.g
        terms = [TERMS[j] for j in support]
        def f(T, x):
            z = dict(th=(T - mu['th']) / sd['th'], r=x['r'], dr=x['dr'], r1=x['r1'], r2=x['r2'], q=x['q'],
                     u=x['u'], h=x['h'], v=(vpd(T, x['RH']) - mu['v']) / sd['v'])
            s = 0.0
            for c, t in zip(xi, terms):
                for v in t:
                    c = c * z[v]
                s += c
            return a * x['R'] - g * T ** 4 + s
        return f

    def run(self, lo, hi, rhs):
        Ts = rk4(self.T[lo], {k: v[lo:hi] for k, v in self.forcing.items()}, rhs)
        return rmse(Ts, self.T[lo:hi]) if np.all(np.isfinite(Ts)) else np.inf

    def fixed_points(self, rhs):
        x = dict(R=self.mu['r'], RH=self.mu['q'], r=0, dr=0, r1=0, r2=0, q=0, u=0, h=0)
        return stable_fixed_points(rhs, self.mu['th'], x)


def _fill(x):
    ok = np.isfinite(x)
    return np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
