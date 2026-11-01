"""
STEP 04 · Try to break the 2008 result
======================================

IN ONE LINE
    Five "what if" tests showing the three-term result does not hinge on a detail.

WHAT HAPPENS
    A. What if we differentiate T instead of using the weak form? (estimator check)
    B. What if dR/dt uses only the past? (causal derivative)
    C. How good is a "climatology" with no physics (mean T of each half hour)?
    D. Day-block bootstrap of the test-block difference, polynomial minus three terms.
    E. What if the test block is the FIRST 30% of the season instead of the last?

INPUT    work/{dry,wet}_res.npz (step 01), results/final_closure.json (step 03)
OUTPUT   results/checks_2008.json
TIME     about 10 s
RUN      python 04_checks_2008.py
PAPER    Sec. III B
"""
import numpy as np
import pandas as pd

import rebio as rb

out = {}
rng_ci = np.random.default_rng(1)            # D: one generator for both seasons, dry first
for season in ('dry', 'wet'):
    e = rb.load('final_closure.json')[season]
    d = rb.ident(season)
    T, R, dR, a, g = d['T'], d['R'], d['dR'], d['alpha'], d['gamma']
    mu, sd = e['mu'], e['sd']
    n = len(T); n_tr = rb.n_train_steps(n); tr, te = slice(0, n_tr), slice(n_tr, n)
    th, r, dr = (T - mu['th']) / sd['th'], (R - mu['r']) / sd['r'], (dR - mu['dr']) / sd['dr']
    X3, XP = np.column_stack([th, dr, r]), rb.poly_features(th, r)
    y_fd = np.gradient(T, rb.DT) - a * R + g * T ** 4

    def score(rhs, dRx=dR, mu_dr=mu['dr']):
        Ts = rb.rk4(T[te][0], {'R': R[te], 'dR': dRx[te]}, rhs)
        return dict(test=rb.rmse(Ts, T[te]) if np.all(np.isfinite(Ts)) else np.inf,
                    stable_fp=rb.stable_fixed_points(rhs, mu['th'], {'R': mu['r'], 'dR': mu_dr}))

    three = lambda c: rb.three_terms(a, g, dict(zip(['th', 'dr', 'r'], c)), mu, sd)
    poly = lambda c: rb.polynomial(a, g, c, mu, sd)
    weak = lambda X: rb.lstsq(*rb.weak_system(T[tr], R[tr], X[tr], a, g))

    # A. ESTIMATOR: weak form vs finite differences, for both the polynomial and the three terms
    o = dict(poly15_fd=score(poly(rb.lstsq(XP[tr], y_fd[tr]))), poly15_weak=score(poly(weak(XP))),
             sindy3_fd=score(three(rb.lstsq(X3[tr], y_fd[tr]))), sindy3_weak=score(three(weak(X3))),
             dry_model=score(rb.dry_model(a, g)))

    # C. NO-PHYSICS BASELINE: mean T of each half hour, from the training days
    nd = n_tr // rb.NPD
    clim = T[:nd * rb.NPD].reshape(nd, rb.NPD).mean(0)
    o['T_climatology'] = rb.rmse(T[te], np.resize(clim, n - n_tr))

    # B. CAUSAL dR/dt: backward difference (uses only the past)
    dRb = np.r_[0.0, np.diff(R) / rb.DT]
    mu_b, sd_b = dict(mu, dr=float(dRb[tr].mean())), dict(sd, dr=float(dRb[tr].std()))
    cb = rb.lstsq(*rb.weak_system(T[tr], R[tr], np.column_stack([th, (dRb - mu_b['dr']) / sd_b['dr'], r])[tr], a, g))
    o['sindy3_weak_causal_dRdt'] = score(rb.three_terms(a, g, dict(zip(['th', 'dr', 'r'], cb)), mu_b, sd_b),
                                         dRx=dRb, mu_dr=mu_b['dr']) | dict(coef=cb.tolist())

    # D. UNCERTAINTY of the free-run difference on the test block
    forcing = {'R': R[te], 'dR': dR[te]}
    sims = {'sindy3': rb.rk4(T[te][0], forcing, three(weak(X3))),
            'poly15': rb.rk4(T[te][0], forcing, poly(rb.lstsq(XP[tr], y_fd[tr]))),
            'dry': rb.rk4(T[te][0], forcing, rb.dry_model(a, g))}
    ndt = (n - n_tr) // rb.NPD; k = ndt * rb.NPD
    se = {m: ((v[:k] - T[te][:k]) ** 2).reshape(ndt, rb.NPD).mean(1) for m, v in sims.items()}
    boots = [np.sqrt(se['poly15'][i].mean()) - np.sqrt(se['sindy3'][i].mean())
             for i in (rng_ci.integers(0, ndt, ndt) for _ in range(5000))]
    o['ci_free_run'] = dict(n_days=ndt, rmse={m: float(np.sqrt(v.mean())) for m, v in se.items()},
                            diff_poly_minus_sindy=float(np.sqrt(se['poly15'].mean()) - np.sqrt(se['sindy3'].mean())),
                            ci95=np.percentile(boots, [2.5, 97.5]).tolist(),
                            days_sindy_better=int(np.sum(se['sindy3'] < se['poly15'])))

    # E. REVERSED SPLIT: test on the first 30% of days instead of the last (daily rollouts)
    f = rb.ident(season, trim=False)
    m = (len(f['T']) // rb.NPD) * rb.NPD
    Tf, Rf = f['T'][:m], f['R'][:m]; dRf = np.gradient(Rf, rb.DT)
    n_te = int(0.30 * (m // rb.NPD)) * rb.NPD
    o['reversed_split'] = {}
    for label, a_tr, b_te in (('test=last30%', slice(0, m - n_te), slice(m - n_te, m)),
                              ('test=first30%', slice(n_te, m), slice(0, n_te))):
        mu2, sd2 = {}, {}
        for key, x in (('th', Tf), ('dr', dRf), ('r', Rf)):
            mu2[key], sd2[key], _ = rb.standardizer(x, a_tr)
        Z3 = np.column_stack([(Tf - mu2['th']) / sd2['th'], (dRf - mu2['dr']) / sd2['dr'], (Rf - mu2['r']) / sd2['r']])
        c3 = rb.lstsq(*rb.weak_system(Tf[a_tr], Rf[a_tr], Z3[a_tr], a, g))
        cp = rb.fit_polynomial(Tf[a_tr], Rf[a_tr], a, g, mu2, sd2)
        models = {'dry model': (rb.dry_model(a, g), None),
                  'SINDy 3 terms': (rb.three_terms(a, g, dict(zip(['th', 'dr', 'r'], c3)), mu2, sd2), None),
                  'polynomial 15 terms': (rb.polynomial(a, g, cp, mu2, sd2), None)}
        df = pd.DataFrame({'T': Tf[b_te], 'R': Rf[b_te], 'dR': dRf[b_te]})
        df['day'] = np.arange(len(df)) // rb.NPD
        days = rb.daily_rollouts(df, models, cols=('R', 'dR'))
        rr = {k: dict(daily_rmse=p['rmse'], diverged=p['diverged'], n_days=p['n'] + p['diverged'])
              for k, p in ((k, rb.pooled(v)) for k, v in days.items())}
        bd = rb.boot_diff(days['SINDy 3 terms'], days['polynomial 15 terms'])
        rr['sindy_minus_poly'] = dict(diff=bd['diff'], ci95=bd['ci95'], n=bd['n'])
        rr['coef_std'] = c3.tolist()
        rr['relax_time_h'] = float(1 / (-c3[0] / sd2['th'] + 4 * g * mu2['th'] ** 3))
        o['reversed_split'][label] = rr
    out[season] = o
    print(f"{season}: test RMSE poly fd {o['poly15_fd']['test']:.3f}, poly weak {o['poly15_weak']['test']:.3f}, "
          f"3 terms fd {o['sindy3_fd']['test']:.3f}, 3 terms weak {o['sindy3_weak']['test']:.3f}, "
          f"climatology {o['T_climatology']:.3f} K")
    rs = o['reversed_split']['test=first30%']
    print(f"     reversed split: 3 terms {rs['SINDy 3 terms']['daily_rmse']:.2f} K, "
          f"polynomial {rs['polynomial 15 terms']['daily_rmse']:.2f} K, tau {rs['relax_time_h']:.1f} h")

rb.save(out, 'checks_2008.json')

print('\nChecks against check_fair.json, check_ci.json, reversed_split.json')
fair, ci, rev = (rb.load(x, rb.REF) for x in ('check_fair.json', 'check_ci.json', 'reversed_split.json'))
for s, o in out.items():
    for k in ('poly15_fd', 'poly15_weak', 'sindy3_fd', 'sindy3_weak', 'dry_model', 'sindy3_weak_causal_dRdt'):
        rb.check(f'{s} {k} test RMSE + fixed points', [o[k]['test']] + o[k]['stable_fp'],
                 [fair[s][k]['test']] + fair[s][k]['stable_fp'], tol=1e-8)
    rb.check(f'{s} T climatology', o['T_climatology'], fair[s]['T_climatology'], tol=1e-10)
    c, r = o['ci_free_run'], ci[s]
    rb.check(f'{s} free-run difference, CI, days better', [c['diff_poly_minus_sindy']] + c['ci95'] + [c['days_sindy_better']],
             [r['diff_poly_minus_sindy']] + r['ci95'] + [r['days_sindy_better']], tol=1e-8)
    for lab in ('test=last30%', 'test=first30%'):
        a_, b_ = o['reversed_split'][lab], rev[s][lab]
        rb.check(f'{s} reversed split {lab}',
                 [a_[m]['daily_rmse'] for m in ('dry model', 'SINDy 3 terms', 'polynomial 15 terms')]
                 + [a_['sindy_minus_poly']['diff']] + a_['sindy_minus_poly']['ci95'] + [a_['relax_time_h']],
                 [b_[m]['daily_rmse'] for m in ('dry model', 'SINDy 3 terms', 'polynomial 15 terms')]
                 + [b_['sindy_minus_poly']['diff']] + b_['sindy_minus_poly']['ci95'] + [b_['relax_time_h']], tol=1e-8)
