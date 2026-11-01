"""
STEP 02 · Which terms matter? (58 candidates)
=============================================

IN ONE LINE
    Picks the few terms, out of 58 candidates, that best explain the residual s.

WHAT HAPPENS
    1. Build the 58 candidate terms and the weak-form system on the 2008 fit block (first 56% of days).
    2. Add terms one at a time, in three ways:
         A  free: the term that most reduces the weak-form error
         B  same, but the temperature term theta is forced in first
         C  greedy: the term that gives the lowest error of the SIMULATED temperature
    3. Keep only models with a stable equilibrium (the temperature returns to it).
    4. 5% rule: the smallest model whose validation error is within 5% of the best.
    5. Refit, bootstrap intervals, score on the test block, compare with the original results.

    The paths point to theta, rdot, r (used in step 03). Variant A in the wet season
    also gives the 11-term equation tested in step 06.

INPUT    work/{dry,wet}_res.npz   (step 01)
OUTPUT   results/selection.json   (step 06 reads it)
TIME     2-5 min (variant C simulates every candidate)
RUN      python 02_select_terms.py
PAPER    Sec. II C-D, III A, III D
"""
import numpy as np

import rebio as rb

KMAX = 12          # longest path for variants A and B
KMAX_C = 8         # longest path for variant C


def forward_stepwise(G, b, kmax, forced=()):
    """Add, one at a time, the column that most reduces the least-squares error of G xi = b."""
    sel, path = list(forced), []
    sst = np.sum((b - b.mean()) ** 2)
    if sel:
        xi = rb.lstsq(G[:, sel], b)
        path.append((list(sel), xi))
    for _ in range(kmax - len(sel)):
        sse = {j: np.sum((b - G[:, sel + [j]] @ rb.lstsq(G[:, sel + [j]], b)) ** 2)
               for j in range(G.shape[1]) if j not in sel}
        sel = sel + [min(sse, key=sse.get)]
        path.append((list(sel), rb.lstsq(G[:, sel], b)))
    return path


def forward_by_simulation(L, G, b, kmax):
    """Add the term giving the lowest validation RMSE of the integrated T (stable models only)."""
    sel, path = [], []
    for _ in range(kmax):
        best = None
        for j in range(G.shape[1]):
            if j in sel:
                continue
            xi = rb.lstsq(G[:, sel + [j]], b)
            f = L.rhs(sel + [j], xi)
            if not L.fixed_points(f):
                continue
            v = L.run(L.i_fit, L.i_tr, f)
            if best is None or v < best[0]:
                best = (v, j, xi)
        if best is None:
            break
        sel = sel + [best[1]]
        path.append((list(sel), best[2]))
    return path


def boot_ci(L, support, n_boot=500, seed=0):
    """95% intervals of the refitted coefficients, resampling whole days of the weak system."""
    G, b = L.weak(0, L.i_tr, stride=1)
    B = rb.boot_coefs(G[:, support], b, np.random.default_rng(seed), n_boot)
    return np.percentile(B, [2.5, 97.5], axis=0)


out = {}
for season in ('dry', 'wet'):
    L = rb.Library(season)
    print(f'\n{season}: fit/validation/test days = {L.i_fit // 48}/{(L.i_tr - L.i_fit) // 48}/{(L.n - L.i_tr) // 48}, '
          f'{len(rb.TERMS)} candidate terms')
    G, b = L.weak(0, L.i_fit)                     # selection: fit block, stride 2
    Gtr, btr = L.weak(0, L.i_tr, stride=1)        # refit: fit + validation, stride 1
    out[season] = {}
    for label, path in [('A weak-SSE, free', forward_stepwise(G, b, KMAX)),
                        ('B weak-SSE, th forced', forward_stepwise(G, b, KMAX, forced=[rb.TERMS.index(('th',))])),
                        ('C validation-RMSE, greedy', forward_by_simulation(L, G, b, KMAX_C))]:
        rows = []
        for sel, xi in path:
            f = L.rhs(sel, xi)
            rows.append((len(sel), L.run(L.i_fit, L.i_tr, f) if L.fixed_points(f) else np.inf, sel))
        best = min(r[1] for r in rows if np.isfinite(r[1]))
        k, _, sel = min((r for r in rows if r[1] <= 1.05 * best), key=lambda r: r[0])     # 5% rule
        xi = rb.lstsq(Gtr[:, sel], btr)
        f = L.rhs(sel, xi)
        ci = boot_ci(L, sel)
        out[season][label] = dict(k=k, terms=[rb.term_name(rb.TERMS[j]) for j in sel], coef=xi.tolist(),
                                  ci=np.array(ci).T.tolist(), test_rmse=L.run(L.i_tr, L.n, f),
                                  full_rmse=L.run(0, L.n, f), fixed_point=L.fixed_points(f),
                                  validation_path=[r[1] for r in rows])
        o = out[season][label]
        print(f"  [{label}] k={k}: {' + '.join(o['terms'])}; test RMSE {o['test_rmse']:.3f} K")
    out[season].update(mu=L.mu, sd=L.sd, a_h=L.a, g_h=L.g)

rb.save(out, 'selection.json')

print('\nChecks against the original equations.json')
ref = rb.load('equations.json', rb.REF)
for season in out:
    for label in ('A weak-SSE, free', 'B weak-SSE, th forced', 'C validation-RMSE, greedy'):
        o, r = out[season][label], ref[season][label]
        same_terms = o['terms'] == r['terms']
        print(f'  check {season} {label:28s} terms {"same" if same_terms else "DIFFERENT"}')
        if same_terms:
            rb.check(f'{season} {label} coefficients', o['coef'], r['coef'], tol=1e-8)
            rb.check(f'{season} {label} test RMSE', o['test_rmse'], r['test_rmse'], tol=1e-6)
            rb.check(f'{season} {label} bootstrap CI', o['ci'], r['ci'], tol=1e-8)
