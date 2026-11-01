"""
STEP 03 · The final three-term equation
=======================================

IN ONE LINE
    Computes the 3 numbers of the final equation, their uncertainty, and tests it on 2008.

    dT/dt = alpha* R - gamma* T^4 + c_T theta + c_Rdot rdot + c_R r

WHAT HAPPENS
    1. Standardize theta, r, rdot with the training block (first 70% of days).
    2. Weak-form least squares for c_T, c_Rdot, c_R; 1000 day-block bootstrap resamples.
    3. Physical units: a_eff, b, k, relaxation time tau; stable equilibrium (fixed point).
    4. Fit the 15-term polynomial comparator on the same block.
    5. Run three models over the test block (last 30%): no correction, polynomial, three terms.
    6. Compare with the original results (check lines).

INPUT    work/{dry,wet}_res.npz   (step 01)
OUTPUT   results/final_closure.json   (steps 04, 05, 06, 08 and 09 read it)
         work/test_block_runs.npz     (Fig. 1)
TIME     a few seconds
RUN      python 03_final_closure.py
PAPER    Tables I, II, III (free-run column)
"""
import numpy as np

import rebio as rb

out, runs = {}, {}
rng = np.random.default_rng(0)       # one generator for both seasons, dry first (as in the original)
for season in ('dry', 'wet'):
    d = rb.ident(season)
    T, R, dR, alpha, gamma = d['T'], d['R'], d['dR'], d['alpha'], d['gamma']
    n = len(T); n_tr = rb.n_train_steps(n); tr = slice(0, n_tr)

    # 1. STANDARDIZE: subtract the training mean, divide by the training standard deviation
    mu, sd, z = {}, {}, {}
    for k, x in (('th', T), ('dr', dR), ('r', R)):
        mu[k], sd[k], z[k] = rb.standardizer(x, tr)
    Theta = np.column_stack([z['th'], z['dr'], z['r']])

    # 2. FIT the 3 coefficients (weak form) + 1000 day-block bootstrap resamples
    G, b = rb.weak_system(T[tr], R[tr], Theta[tr], alpha, gamma)
    c = rb.lstsq(G, b)
    boot = rb.boot_coefs(G, b, rng, n_boot=1000)

    # 3. PHYSICAL UNITS (a_eff, b, k, tau) and the stable equilibrium
    #    'const' belongs to the form  a_eff R + b dR/dt - k_T T - gamma T^4 + const  (14.2 dry, 21.5 wet K/h);
    #    the paper writes  - k (T - Tbar) + c0, with c0 = const - k_T Tbar (-0.148 dry, -0.215 wet K/h).
    def physical(cT, cRd, cR):
        lam = -cT / sd['th'] + 4 * gamma * mu['th'] ** 3        # total linearized relaxation rate (1/h)
        return dict(a_eff=alpha + cR / sd['r'], b=cRd / sd['dr'], k_T=-cT / sd['th'], lam=lam, tau_h=1 / lam,
                    const=-cT * mu['th'] / sd['th'] - cRd * mu['dr'] / sd['dr'] - cR * mu['r'] / sd['r'])
    phys = physical(*c)
    pb = [physical(*cb) for cb in boot]
    coef = dict(zip(['th', 'dr', 'r'], c.tolist()))
    three = rb.three_terms(alpha, gamma, coef, mu, sd)
    fixed = rb.stable_fixed_points(three, mu['th'], {'R': mu['r'], 'dR': mu['dr']})

    # 4. COMPARATOR: 15-term polynomial (derivative on the whole series, fit on the training rows)
    xi_poly = rb.fit_polynomial(T, R, alpha, gamma, mu, sd, rows=tr)
    poly = rb.polynomial(alpha, gamma, xi_poly, mu, sd)

    # 5. TEST: run the three models continuously over the test block
    te = slice(n_tr, n)
    forcing = {'R': R[te], 'dR': dR[te]}
    sims = {'dry model': rb.rk4(T[te][0], forcing, rb.dry_model(alpha, gamma)),
            'polynomial, 15 terms': rb.rk4(T[te][0], forcing, poly),
            'SINDy, 3 terms': rb.rk4(T[te][0], forcing, three)}
    full = rb.rk4(T[0], {'R': R, 'dR': dR}, three)
    runs.update({f'{season}_obs': T[te]} | {f'{season}_{k}': v for k, v in sims.items()})

    out[season] = dict(alpha=alpha, gamma=gamma, mu=mu, sd=sd, coef_std=coef,
                       coef_std_ci=dict(zip(['th', 'dr', 'r'], np.percentile(boot, [2.5, 97.5], axis=0).T.tolist())),
                       physical={k: float(v) for k, v in phys.items()},
                       physical_ci={k: np.percentile([p[k] for p in pb], [2.5, 97.5]).tolist() for k in phys},
                       fixed_point=fixed, poly_xi=xi_poly.tolist(),
                       poly_fixed_points=rb.stable_fixed_points(poly, mu['th'], {'R': mu['r'], 'dR': mu['dr']}),
                       test_rmse={k: rb.rmse(v, T[te]) for k, v in sims.items()}, test_days=(n - n_tr) // rb.NPD,
                       full_rmse=rb.rmse(full, T))
    o = out[season]
    print(f"{season}: s = {c[0]:+.3f} theta {c[1]:+.3f} rdot {c[2]:+.3f} r  (K/h per s.d.); "
          f"tau = {phys['tau_h']:.1f} h; fixed point {[round(t, 2) for t in fixed]} K")
    print('     test free runs (K):', {k: round(v, 3) for k, v in o['test_rmse'].items()})

rb.save(out, 'final_closure.json')
np.savez(f'{rb.WORK}/test_block_runs.npz', **runs)

print('\nChecks against the original final_equations.json')
ref = rb.load('final_equations.json', rb.REF)
for s in out:
    o, r = out[s], ref[s]
    rb.check(f'{s} coefficients', list(o['coef_std'].values()), list(r['coef_std'].values()), tol=1e-10)
    rb.check(f'{s} coefficient CIs (bootstrap)', list(o['coef_std_ci'].values()), list(r['coef_std_ci'].values()), tol=1e-10)
    for k in ('a_eff', 'b', 'k_T', 'tau_h', 'const'):
        rb.check(f'{s} {k} and its CI', [o['physical'][k]] + o['physical_ci'][k],
                 [r['physical'][k]] + r['physical_ci'][k], tol=1e-10)
    rb.check(f'{s} test-block RMSEs (dry, polynomial, 3 terms)', list(o['test_rmse'].values()),
             list(r['test_rmse'].values()), tol=1e-8)
    rb.check(f'{s} full-season RMSE and fixed point', [o['full_rmse']] + o['fixed_point'],
             [r['full_rmse']] + r['fixed_point'], tol=1e-8)
fair = rb.load('check_fair.json', rb.REF)
for s in out:
    rb.check(f'{s} polynomial stable fixed points (check_fair.json)', out[s]['poly_fixed_points'],
             fair[s]['poly15_fd']['stable_fp'], tol=1e-8)
