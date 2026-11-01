"""
08 · prepare_data.py · export the data PySR needs
=================================================

IN ONE LINE
    Saves the 2008 and 2006 data, already split, into one small file for run_pysr.py.

WHAT HAPPENS
    Same calendar, gap rule, splits and standardization as step 05:
    training = first 70% of the 2008 days (fit = first 56%, validation = 56-70%),
    test = last 30%, plus 2006 for out-of-year scoring. Evaluation sets keep only complete days.

INPUT    spreadsheet, results/final_closure.json (step 03)
OUTPUT   08_pysr/pysr_data.npz
TIME     seconds
RUN      python 08_pysr/prepare_data.py      (from the main folder)
NEEDS    numpy, pandas (no Julia)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import rebio as rb


def complete_days(df):
    days = [d for _, d in df.groupby('day') if len(d) == rb.NPD and not d[['R', 'T']].isna().any().any()]
    return np.array([d['T'].values for d in days]), np.array([d.R.values for d in days])


out = {}
for season in ('dry', 'wet'):
    e = rb.load('final_closure.json')[season]
    d8 = rb.season_frame(2008, season)
    nd = len(d8) // rb.NPD
    n_tr, n_fit = int(rb.TRAIN * nd) * rb.NPD, int(rb.FIT * nd) * rb.NPD
    tr = d8.iloc[:n_tr][['R', 'T']].interpolate()
    assert abs(tr['T'].mean() - e['mu']['th']) < 0.05, 'training mean mismatch'
    out[f'{season}_train_T'], out[f'{season}_train_R'] = tr['T'].values, tr.R.values
    out[f'{season}_n_fit'] = n_fit
    for name, df in (('val', d8.iloc[n_fit:n_tr]), ('test', d8.iloc[n_tr:]), ('y2006', rb.season_frame(2006, season))):
        out[f'{season}_{name}_T'], out[f'{season}_{name}_R'] = complete_days(df)
        print(season, name, out[f'{season}_{name}_T'].shape)
    out[f'{season}_meta'] = json.dumps(dict(a=e['alpha'], g=e['gamma'], mu=e['mu'], sd=e['sd'], sindy3=e['coef_std']))
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pysr_data.npz'), **out)
