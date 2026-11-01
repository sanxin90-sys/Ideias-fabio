"""
08 · run_pysr.py · Symbolic regression: let the computer invent formulas
=======================================================================

IN ONE LINE
    PySR searches for any formula s(theta, r, rdot), with no fixed list of terms,
    and checks whether something beats the three-term equation.

    dT/dt = a* R - g* T^4 + s(theta, r, rdot)

THE PROTOCOL (fixed BEFORE running)
    1. Target s on the 2008 fit block. dT/dt from a Savitzky-Golay derivative (3.5 h window);
       DERIV=rts switches to an RTS smoother whose setting is chosen automatically.
    2. PySR with NSEED seeds, operators + - * /, formulas of at most 20 nodes.
       Every formula of every seed is kept.
    3. Rank formulas by daily-rollout error on the VALIDATION block; keep only those that
       never blow up and have a stable equilibrium.
    4. Refit the constants of the best 5 in the weak form; freeze; score on the 2008 TEST
       block and on 2006 against the three-term equation, with day-block bootstrap intervals.
    Decision rule: adopt a new formula only if it beats the three terms in BOTH tests,
    with intervals excluding zero. (No formula did.)

INPUT    pysr_data.npz (made by prepare_data.py)
OUTPUT   results_pysr.json, candidates_pysr.csv, run.log
TIME     hours. Run on your own computer: the first run downloads Julia.
INSTALL  pip install "pysr>=0.19" "pynumdiff>=0.3" sympy scipy pandas
RUN      cd 08_pysr ; python run_pysr.py      (settings: env NSEED=10 NITER=60)
NOTE     The search is random: formulas change between runs; the ingredients did not.
PAPER    Sec. II F, Sec. III F
"""
import os, sys, json, time, platform
import numpy as np, pandas as pd, sympy as sp
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
H = 0.5
NSEED = int(os.environ.get('NSEED', 10))
NITER = int(os.environ.get('NITER', 60))
BANDLIMIT = 1 / 24.0
# DERIV=sg (default since 2026-10-27 21:3x): the RTS q/r criterion hit the grid boundary (log q/r = -8) on the
# real 2008 fit blocks and flattened dT/dt (wet target std 0.06 K/h), so the target became ~ -a*R. Savitzky-Golay
# (window 3.5 h, quadratic; best worst-case in ruido/resultado.md) replaces it. DERIV=rts keeps the original choice.
DERIV = os.environ.get('DERIV', 'sg')
TOPK = 5
SYM = sp.symbols('th r dr')
LOG = open(os.path.join(HERE, 'run.log'), 'a')


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True); LOG.write(msg + '\n'); LOG.flush()


# ------------------------------------------------------------------ 1. DERIVATIVE (RTS smoother, setting chosen automatically)
# Only used with DERIV=rts. Tries settings and keeps the best balance of fit vs roughness.
def rts_derivative(x):
    import pynumdiff
    tvgamma = BANDLIMIT ** -1.6 * H ** -0.71 * np.exp(-5.1)   # same heuristic as pynumdiff.optimize
    best = None
    for v in np.arange(-8.0, 14.0, 1.0):
        d = pynumdiff.rtsdiff(x, H, order=2, log_qr_ratio=v)[1]
        xr = np.concatenate([[0.0], np.cumsum((d[1:] + d[:-1]) / 2 * H)])
        xr += np.mean(x - xr)
        cost = np.sqrt(np.mean((xr - x) ** 2)) + tvgamma * np.sum(np.abs(np.diff(d)))
        if best is None or cost < best[0]:
            best = (cost, v, d)
    return best[2], float(best[1])


# ------------------------------------------------------------------ 2. WEAK FORM (same as rebio.weak_system, as sparse matrices)
def weak_ops(n, width_h=4.0, stride=1, p=4):
    import scipy.sparse as sps
    m = int(round(width_h / H)); u = np.linspace(-1, 1, 2 * m + 1)
    phi = (1 - u ** 2) ** p; dphi = -2 * p * u * (1 - u ** 2) ** (p - 1) / (m * H)
    w = np.full(2 * m + 1, H); w[[0, -1]] = H / 2
    c = np.arange(m, n - m, stride)
    rows = np.repeat(np.arange(len(c)), 2 * m + 1); cols = (c[:, None] + np.arange(-m, m + 1)).ravel()
    W = sps.csr_matrix((np.tile(w * phi, len(c)), (rows, cols)), shape=(len(c), n))
    D = sps.csr_matrix((np.tile(w * dphi, len(c)), (rows, cols)), shape=(len(c), n))
    return W, D


# ------------------------------------------------------------------ 3. FORMULAS, ROLLOUTS, EQUILIBRIUM, BOOTSTRAP
# Closure turns a PySR formula into a function whose constants can be refitted.
class Closure:
    """s(th, r, dr) from a sympy expression; numeric constants exposed as parameters."""
    def __init__(self, expr):
        expr = sp.sympify(str(expr), locals=dict(zip(['th', 'r', 'dr'], SYM)))
        floats = sorted(expr.atoms(sp.Float), key=lambda f: float(f))
        self.p0 = np.array([float(f) for f in floats])
        ps = sp.symbols(f'p0:{len(floats)}') if floats else ()
        self.template = expr.xreplace(dict(zip(floats, ps)))
        self.f = sp.lambdify(SYM + tuple(ps), self.template, 'numpy')
        self.p = self.p0.copy(); self.expr = expr

    def __call__(self, th, r, dr, p=None):
        with np.errstate(all='ignore'):
            v = self.f(th, r, dr, *(self.p if p is None else p))
        return np.broadcast_to(np.asarray(v, float), np.broadcast(th, r, dr).shape)

    def text(self, p=None):
        pp = self.p if p is None else p
        e = self.template.xreplace({s: sp.Float(v, 4) for s, v in zip(sp.symbols(f'p0:{len(pp)}'), pp)})
        return str(e)


def rollout(Tday, Rday, M, s):
    """Daily rollouts (days x 48), RK4 with 6 sub-steps, R and dR/dt linear inside each step,
    reset to the observed T at the first step of each day (as validate.py). Returns per-day RMSE (inf if blow-up)."""
    a, g, mu, sd = M['a'], M['g'], M['mu'], M['sd']
    dR = np.gradient(Rday, H, axis=1)
    nd = len(Tday); x = Tday[:, 0].copy(); Ts = np.empty_like(Tday); Ts[:, 0] = x
    alive = np.ones(nd, bool); nsub = 6; h = H / nsub

    def rhs(T, R, D):
        th = (T - mu['th']) / sd['th']; r = (R - mu['r']) / sd['r']; dr = (D - mu['dr']) / sd['dr']
        return a * R - g * T ** 4 + s(th, r, dr)
    for i in range(47):
        for j in range(nsub):
            fr = [(j + q) / nsub for q in (0, .5, 1)]
            R3 = [Rday[:, i] + f * (Rday[:, i + 1] - Rday[:, i]) for f in fr]
            D3 = [dR[:, i] + f * (dR[:, i + 1] - dR[:, i]) for f in fr]
            with np.errstate(all='ignore'):
                k1 = rhs(x, R3[0], D3[0]); k2 = rhs(x + h * k1 / 2, R3[1], D3[1])
                k3 = rhs(x + h * k2 / 2, R3[1], D3[1]); k4 = rhs(x + h * k3, R3[2], D3[2])
                x = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        bad = ~np.isfinite(x) | (x < 270) | (x > 330)
        alive &= ~bad; x = np.where(alive, x, Tday[:, i + 1]); Ts[:, i + 1] = x
    err = np.sqrt(np.mean((Ts - Tday) ** 2, axis=1))
    return np.where(alive, err, np.inf)


def pooled(e):
    return float(np.sqrt(np.mean(e ** 2))) if np.all(np.isfinite(e)) else float('inf')


def stable_fixed_point(M, s):
    mu, sd = M['mu'], M['sd']
    grid = np.linspace(mu['th'] - 20, mu['th'] + 20, 801)
    with np.errstate(all='ignore'):
        f = M['a'] * mu['r'] - M['g'] * grid ** 4 + s((grid - mu['th']) / sd['th'], 0.0 * grid, 0.0 * grid - mu['dr'] / sd['dr'])
    return [float(grid[i]) for i in range(len(f) - 1) if np.isfinite(f[i]) and np.isfinite(f[i + 1]) and f[i] > 0 > f[i + 1]]


def boot_diff(a, b, B=5000, seed=1):
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    idx = np.random.default_rng(seed).integers(0, len(a), (B, len(a)))
    d = np.sqrt(np.mean(a[idx] ** 2, 1)) - np.sqrt(np.mean(b[idx] ** 2, 1))
    return dict(diff=float(np.sqrt(np.mean(a ** 2)) - np.sqrt(np.mean(b ** 2))),
                ci95=np.percentile(d, [2.5, 97.5]).tolist(), n_days=int(len(a)))


# ------------------------------------------------------------------ 4. MAIN: search, rank, refit, test
def main():
    from pysr import PySRRegressor, __version__ as pysr_version
    Dz = np.load(os.path.join(HERE, 'pysr_data.npz'))
    results = dict(config=dict(DERIV=DERIV, NSEED=NSEED, NITER=NITER, bandlimit_per_h=BANDLIMIT, maxsize=20,
                               operators=['+', '-', '*', '/'], pysr=pysr_version, python=platform.python_version(),
                               numpy=np.__version__, platform=platform.platform()), seasons={})
    cand_rows = []
    for season in ('dry', 'wet'):
        M = json.loads(str(Dz[f'{season}_meta']))
        T, R = Dz[f'{season}_train_T'], Dz[f'{season}_train_R']; n_fit = int(Dz[f'{season}_n_fit'])
        sets = {k: (Dz[f'{season}_{k}_T'], Dz[f'{season}_{k}_R']) for k in ('val', 'test', 'y2006')}
        mu, sd = M['mu'], M['sd']
        dRtr = np.gradient(R, H)
        Z = np.column_stack([(T - mu['th']) / sd['th'], (R - mu['r']) / sd['r'], (dRtr - mu['dr']) / sd['dr']])
        if DERIV == 'rts':
            dT, qr = rts_derivative(T[:n_fit])
        else:
            import pynumdiff
            dT, qr = pynumdiff.savgoldiff(T[:n_fit], H, degree=2, window_size=7, smoothing_win=3)[1], 'n/a (SG 3.5 h)'
        y = dT - M['a'] * R[:n_fit] + M['g'] * T[:n_fit] ** 4
        log(f'== {season}: fit {n_fit // 48} d, RTS log(q/r) = {qr}, target std {y.std():.3f} K/h')
        c3 = M['sindy3']
        sindy = Closure(f"{c3['th']}*th + {c3['r']}*r + {c3['dr']}*dr")
        ref = {k: rollout(*v, M, sindy) for k, v in sets.items()}
        log(f'   SINDy-3 frozen: val {pooled(ref["val"]):.3f}  test {pooled(ref["test"]):.3f}  2006 {pooled(ref["y2006"]):.3f} K')
        cands = []
        for seed in range(NSEED):
            t0 = time.time()
            model = PySRRegressor(niterations=NITER, populations=15, maxsize=20,
                                  binary_operators=['+', '-', '*', '/'], unary_operators=[],
                                  model_selection='best', random_state=seed, verbosity=0, progress=False)
            model.fit(Z[:n_fit], y, variable_names=['th', 'r', 'dr'])
            for _, row in model.equations_.iterrows():
                try:
                    cl = Closure(row['sympy_format'])
                except Exception as ex:
                    log('   skip', row['equation'], ex); continue
                ev = rollout(*sets['val'], M, cl); fp = stable_fixed_point(M, cl)
                rec = dict(season=season, seed=seed, complexity=int(row['complexity']), loss=float(row['loss']),
                           expr=str(cl.expr), val_rmse=pooled(ev), stable_fp=len(fp) > 0)
                cand_rows.append(rec); cands.append((rec, cl))
            log(f'   seed {seed}: {len(model.equations_)} eqs, {time.time() - t0:.0f} s, best val '
                f'{min([r["val_rmse"] for r, _ in cands if r["seed"] == seed and r["stable_fp"]] + [np.inf]):.3f} K')
        ok = sorted([c for c in cands if c[0]['stable_fp'] and np.isfinite(c[0]['val_rmse'])], key=lambda c: c[0]['val_rmse'])
        top, seen = [], set()
        for rec, cl in ok:
            if rec['expr'] in seen: continue
            seen.add(rec['expr']); top.append((rec, cl))
            if len(top) == TOPK: break
        # weak refit of the constants on fit + validation (whole training block)
        W, Dm = weak_ops(len(T)); b = -(Dm @ T) - W @ (M['a'] * R - M['g'] * T ** 4)
        season_out = dict(rts_log_qr=qr, sindy3=dict(val=pooled(ref['val']), test=pooled(ref['test']), y2006=pooled(ref['y2006'])),
                          n_candidates=len(cands), n_stable=len(ok), top=[])
        for rec, cl in top:
            entry = dict(expr_pysr=rec['expr'], seed=rec['seed'], complexity=rec['complexity'], val_rmse_pysr=rec['val_rmse'])
            for tag, p in (('pysr', cl.p0), ('weak_refit', None)):
                if tag == 'weak_refit':
                    if len(cl.p0) == 0: continue
                    fit = least_squares(lambda pp: W @ cl(Z[:, 0], Z[:, 1], Z[:, 2], pp) - b, cl.p0, method='lm')
                    p = fit.x
                cl.p = p
                fp = stable_fixed_point(M, cl)
                ev = {k: rollout(*v, M, cl) for k, v in sets.items()}
                entry[tag] = dict(expr=cl.text(p), stable_fp=fp, val=pooled(ev['val']), test=pooled(ev['test']),
                                  y2006=pooled(ev['y2006']),
                                  test_minus_sindy3=boot_diff(ev['test'], ref['test']),
                                  y2006_minus_sindy3=boot_diff(ev['y2006'], ref['y2006']),
                                  days_diverged={k: int(np.sum(~np.isfinite(v))) for k, v in ev.items()})
            cl.p = cl.p0
            season_out['top'].append(entry)
            log('   TOP', json.dumps(entry)[:600])
        results['seasons'][season] = season_out
        pd.DataFrame(cand_rows).to_csv(os.path.join(HERE, 'candidates_pysr.csv'), index=False)
        json.dump(results, open(os.path.join(HERE, 'results_pysr.json'), 'w'), indent=1)
    log('DONE')


if __name__ == '__main__':
    main()
