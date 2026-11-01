# Verification record

**Date:** 2026-10-30. **Environment:** Linux, Python 3.13.17, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0,
openpyxl 3.1.5, scikit-learn 1.9.1, matplotlib 3.11.2, jax 0.12.0.1, optax 0.2.8, pysindy 2.1.0.
Fresh virtual environment, fresh copy of this folder, spreadsheet copied into `data/`.

| What was run | Result |
| --- | --- |
| `python run_all.py` (steps 01–06, 09) | `Done. Checks: 116 same, 0 different.` |
| `07_diagnostics/expD_kernel.py`, `expC_real.py`, `expF_anomaly.py`, then `compare.py` | largest absolute difference from `reference/`: 0 (expC 30 numbers, expD 40, expF 64) |
| `07_diagnostics/exp_synthetic.py` | output identical to `reference/synthetic_results.txt` |
| `08_pysr/prepare_data.py` | `pysr_data.npz` identical (max abs. difference 0) to the file used for the published PySR run |

**Not run here:** `08_pysr/run_pysr.py` (needs Julia; it was run on the author's computer on
2026-10-27 with PySR 2.7.2, 10 seeds × 60 iterations). The genetic search is not deterministic, so a
re-run is expected to give different expressions with the same ingredients.

An earlier verification (2026-10-28, separate clean folder) gave the same counts and pixel-identical figures.
