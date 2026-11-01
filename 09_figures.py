"""
STEP 09 · Figures 1 and 2
=========================

IN ONE LINE
    Draws the two figures of the paper.

WHAT HAPPENS
    Fig. 1  measured vs simulated temperature on the 2008 test blocks.
            Before drawing, it checks the errors equal those in the paper (assert).
    Fig. 2  three-term minus polynomial error in every year, with 95% intervals.
    Colour-blind-safe palette (Okabe-Ito), journal column width (6.69 in).

INPUT    work/test_block_runs.npz, results/final_closure.json (step 03), results/validation.json (step 05)
OUTPUT   results/fig1_test_blocks.{pdf,png}, results/fig2_transfer.{pdf,png}
TIME     under 1 min
RUN      python 09_figures.py
PAPER    Figs. 1-2
"""
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rebio as rb

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
COL = {'obs': '0.1', 'dry': '#999999', 'poly': '#E69F00', 'sindy': '#0072B2'}   # Okabe-Ito
W2 = 6.69   # AIP two-column width, inches

# ---------------- Fig. 1: measured vs simulated T, 2008 test blocks ----------------
runs = np.load(f'{rb.WORK}/test_block_runs.npz')
EQ = rb.load('final_closure.json')
fig, axes = plt.subplots(2, 1, figsize=(W2, 4.4), constrained_layout=True)
for ax, season, tag in zip(axes, ['dry', 'wet'], ['(a)', '(b)']):
    Tobs = runs[f'{season}_obs']
    sims = {k: runs[f'{season}_{k}'] for k in ('dry model', 'polynomial, 15 terms', 'SINDy, 3 terms')}
    for k, v in sims.items():   # same numbers as quoted in the paper
        assert abs(rb.rmse(v, Tobs) - EQ[season]['test_rmse'][k]) < 1e-6, (season, k)
    t_d = np.arange(len(Tobs)) * rb.DT / 24
    ax.plot(t_d, Tobs - 273.15, color=COL['obs'], lw=0.8, label='Observed', zorder=4)
    ax.plot(t_d, sims['dry model'] - 273.15, color=COL['dry'], lw=0.7, label='Uncorrected model')
    ax.plot(t_d, sims['polynomial, 15 terms'] - 273.15, color=COL['poly'], lw=0.7, label='15-term polynomial')
    ax.plot(t_d, sims['SINDy, 3 terms'] - 273.15, color=COL['sindy'], lw=0.7, label='Three-term closure')
    ax.set_ylabel('Air temperature at 62 m (°C)')
    ax.set_xlim(0, t_d[-1])
    ax.text(0.005, 1.0, f'{tag} {"Dry" if season == "dry" else "Wet"} season, {len(Tobs) // 48}-day test block',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8)
axes[1].set_xlabel('Days since start of the 2008 test block')
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=4, frameon=False)
fig.savefig(f'{rb.OUT}/fig1_test_blocks.pdf'); fig.savefig(f'{rb.OUT}/fig1_test_blocks.png', dpi=600)
plt.close(fig)

# ---------------- Fig. 2: three terms minus polynomial, every year ----------------
R = rb.load('validation.json')
ROWS = [('2008', '2008, held-out block (62 m)', 'main'), ('2006', '2006 (22 m)', 'main'),
        ('2008_T22', '2008, temperature at 22 m', 'main'), ('2010', '2010 (62 m), supplementary', 'sup'),
        ('2009', '2009 (62 m), supplementary', 'sup'), ('2007', '2007 (22 m), low coverage', 'sup')]
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7), sharey=True, constrained_layout=True)
for ax, season, tag in zip(axes, ['dry', 'wet'], ['(a) Dry season', '(b) Wet season']):
    y = np.arange(len(ROWS))[::-1]
    for yi, (key, lab, kind) in zip(y, ROWS):
        v = R[season]['years'][key]; d = v['sindy_minus_poly_daily']
        ndiv = v['polynomial 15 terms']['n_days_diverged']; nd = v['SINDy 3 terms']['n_days']
        note = f'{ndiv} of {nd} days' if ndiv else ''
        if d['n'] < 5:
            ax.text(0, yi, 'not estimable', ha='center', va='center', fontsize=7, style='italic')
        else:
            filled = kind == 'main'
            ax.errorbar(d['diff'], yi, xerr=[[d['diff'] - d['ci95'][0]], [d['ci95'][1] - d['diff']]],
                        fmt='o', ms=4, color=COL['sindy'], mfc=COL['sindy'] if filled else 'white',
                        ecolor=COL['sindy'], elinewidth=1, capsize=2)
        if note:
            ax.text(1.02, yi, note, transform=ax.get_yaxis_transform(), ha='left', va='center',
                    fontsize=7, color='#9A6200')
    ax.axvline(0, color='0.4', lw=0.7, ls='--', zorder=0)
    ax.axhline(y[2] - 0.5, color='0.8', lw=0.6)
    ax.set_yticks(y, [r[1] for r in ROWS])
    ax.set_xlim(-1.45, 0.45)
    ax.set_xlabel('Three-term minus polynomial RMSE (K)')
    ax.set_title(tag, loc='left', fontsize=8)
axes[1].text(1.02, 1.0, 'Polynomial\ndiverged', transform=axes[1].transAxes, ha='left', va='bottom',
             fontsize=7, color='#9A6200')
axes[0].text(1.02, 1.0, 'Polynomial\ndiverged', transform=axes[0].transAxes, ha='left', va='bottom',
             fontsize=7, color='#9A6200')
fig.savefig(f'{rb.OUT}/fig2_transfer.pdf'); fig.savefig(f'{rb.OUT}/fig2_transfer.png', dpi=600)
print('saved results/fig1_test_blocks.* and results/fig2_transfer.*')

