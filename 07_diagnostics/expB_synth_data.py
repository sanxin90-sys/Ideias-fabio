"""
07 · expB_synth_data.py · a fake series with a known memory
===========================================================

IN ONE LINE
    Makes a synthetic temperature whose correction has a known 3-hour memory of radiation.

WHAT HAPPENS
    Two coupled equations, driven by the real 2008 radiation:
        dT/dt = a R - g T^4 - kappa (H - Hbar) - lam (T - Tref)
        dH/dt = (R - H) / tau,   tau = 3 h     (H is a slow "reservoir" of radiation)
    Then adds 0.05 K noise and computes the residual s_exp as for the real data.

USED BY  expD_kernel.py (imported, not run on its own)
INPUT    ../work/y2008.pkl, ../work/dry_params.npy (step 01)
"""
import numpy as np, pandas as pd, json

d = pd.read_pickle('../work/y2008.pkl')
R = d['R_liq.4'].interpolate(limit_direction='both').values
a_h, g_h = (float(x) for x in np.load('../work/dry_params.npy') / (1 / (60e-6 * 60)))
tau, lam = 3.0, 1 / 12
Tref = (a_h * R.mean() / g_h) ** 0.25
Hbar = R.mean()


def simulate(kappa, dt=0.5, sub=10):
    T = np.empty(len(R)); H = np.empty(len(R)); T[0], H[0] = Tref, Hbar
    h = dt / sub
    for i in range(1, len(R)):
        x, y, r = T[i - 1], H[i - 1], R[i - 1]
        for _ in range(sub):
            fT = lambda x, y: a_h * r - g_h * x ** 4 - kappa * (y - Hbar) - lam * (x - Tref)
            fH = lambda x, y: (r - y) / tau
            k1 = (fT(x, y), fH(x, y)); k2 = (fT(x + h * k1[0] / 2, y + h * k1[1] / 2), fH(x + h * k1[0] / 2, y + h * k1[1] / 2))
            k3 = (fT(x + h * k2[0] / 2, y + h * k2[1] / 2), fH(x + h * k2[0] / 2, y + h * k2[1] / 2))
            k4 = (fT(x + h * k3[0], y + h * k3[1]), fH(x + h * k3[0], y + h * k3[1]))
            x += h * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
            y += h * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
        T[i], H[i] = x, y
    return T, H


kappa = 0.6 * a_h  # memory term of the same order as the radiative forcing
T, H = simulate(kappa)
s_true = -kappa * (H - Hbar) - lam * (T - Tref)
rng = np.random.default_rng(0)
Tobs = T + rng.normal(0, 0.05, len(T))
s_exp = np.gradient(Tobs, 0.5) - a_h * R + g_h * Tobs ** 4
print('std a R = %.3f, std s_true = %.3f, std memory part = %.3f, std s_exp = %.3f K/h' % (
    (a_h * R).std(), s_true.std(), (kappa * H).std(), s_exp.std()))
print('T range %.1f-%.1f K' % (T.min(), T.max()))

