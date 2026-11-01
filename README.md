# Three-term closure for a minimal radiative model of air temperature (Rebio Jaru, SW Amazon)

Code for the manuscript *"A three-term closure for a minimal radiative model of air temperature in the
southwestern Amazon, identified by weak-form sparse regression"* (submitted to *Chaos*).

The study revisits the minimal radiative model of da Silva et al. (2026, *Chaos* 36, 033134),

    dT/dt = α* R − γ* T⁴ + s,

whose correction `s` was a neural network later summarized by a 15-term polynomial in T and net
radiation R. Here `s` is identified directly from the 2008 tower record by **weak-form sparse
regression (weak SINDy)**, selecting terms by the error of the *integrated* temperature and requiring a
stable fixed point. The same three terms are retained in the dry and the wet season:

    dT/dt = α* R − γ* T⁴ + c_T θ + c_Ṙ ṙ + c_R r

with θ, r and ṙ the standardized temperature, net radiation and dR/dt (training-block mean and
standard deviation). Time is in hours, T in K, R in W m⁻².

This repository reproduces every number, table and figure of the manuscript from the raw tower
spreadsheet, and checks each result against the original values stored in `reference/`.

## What each file does (start here)

Every file opens with the same short header: **IN ONE LINE**, **WHAT HAPPENS**, **INPUT**,
**OUTPUT**, **TIME**, **RUN**. Read that header first; the code below it follows the same numbered steps.

| File | In one line |
| --- | --- |
| `run_all.py` | The "on" button: runs steps 01–06 and 09 and counts the checks |
| `rebio.py` | The toolbox every step uses (reading data, weak form, models, RK4, bootstrap) |
| `01_prepare.py` | Fits the simple physical model and computes what it gets wrong (the residual s) |
| `02_select_terms.py` | Picks the few terms, out of 58 candidates, that explain s |
| `03_final_closure.py` | Computes the 3 numbers of the final equation and tests it on 2008 |
| `04_checks_2008.py` | Five "what if" tests that try to break the 2008 result |
| `05_validate_years.py` | Applies the 2008 equations, unchanged, to 2006–2010 (Table IV) |
| `06_explore.py` | Tests two alternatives out of year (cycle + anomaly; 11-term wet equation) |
| `07_diagnostics/` | Neural networks (MLP, DeepONet, FNO), memory kernel, synthetic test. Needs jax, pysindy. See its README |
| `08_pysr/` | Symbolic regression: the computer invents formulas. Needs Julia. See its README |
| `09_figures.py` | Draws Figures 1 and 2 |
| `reference/` | The original results; every step compares itself against them |
| `data/README.md` | What the input spreadsheet must look like (data not included) |

## Quick start

```bash
git clone <this repository>
cd <repository folder>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy scipy pandas openpyxl scikit-learn matplotlib
# copy the spreadsheet to data/ (see data/README.md; not distributed here)
python run_all.py
```

The run takes about 5–15 minutes. The last line must read:

    Done. Checks: 116 same, 0 different.

`0 different` means every recomputed number equals the value reported in the manuscript.

## Requirements

- Python ≥ 3.10 (verified with 3.13).
- Steps 01–06 and 09: numpy, scipy, pandas, openpyxl, scikit-learn, matplotlib.
- Step 07 (diagnostics): additionally jax, optax, pysindy.
- Step 08 (symbolic regression): PySR ≥ 0.19 (needs Julia, downloaded on first run), pynumdiff ≥ 0.3,
  sympy. Run it on a local machine; it takes hours.

`requirements.txt` lists all packages and the versions used in the verification run
(see `VERIFICATION.md`).

## Input data

The scripts read `data/Dados Rebio Jaru 2006 a 2008.xlsx`, sheet `DADOS OK`: half-hourly net
radiation, air temperature, relative humidity and wind speed from the Rebio Jaru LBA tower, 2006–2010.
**The raw data are not included.** `data/README.md` describes the expected file, the columns used for
each year, and the known data-quality issues. Request the data from the corresponding author.

## Pipeline: which script gives which result

| Step | Script | What it does | Manuscript | Time |
| --- | --- | --- | --- | --- |
| 01 | `01_prepare.py` | Reads 2008; refits α\*, γ\* by RK4 trajectory; residual s_exp; predictability R² (Table V); condition numbers | Sec. II B, IV C, Table V | 1–5 min |
| 02 | `02_select_terms.py` | Forward selection on a 58-term library (variants A/B/C); 11-term wet-season closure | Sec. II C–D, III A, III D | 2–5 min |
| 03 | `03_final_closure.py` | Final three-term fit, day-block bootstrap, physical parameters, fixed points, 2008 free runs | Tables I–III | ~5 s |
| 04 | `04_checks_2008.py` | Weak vs finite-difference estimator, causal dR/dt, climatology, reversed split | Sec. III B | ~10 s |
| 05 | `05_validate_years.py` | Frozen coefficients on 2006–2010; daily rollouts; day-block bootstrap; re-centring | Table IV, Fig. 2 data | ~1 min |
| 06 | `06_explore.py` | Cycle-plus-anomaly closure and 11-term wet closure out of year | Sec. III C–D | ~1 min |
| 07 | `07_diagnostics/` | Pointwise MLP, DeepONet, FNO, memory kernel, synthetic identifiability test | Sec. III E, IV C | ~15 min |
| 08 | `08_pysr/` | Symbolic-regression check with PySR (10 seeds) | Sec. III F | hours (local) |
| 09 | `09_figures.py` | Figures 1 and 2 | Figs. 1–2 | < 1 min |

`run_all.py` runs steps 01–06 and 09 in order (`python run_all.py 03 04` runs only those). To run a
single step: `python 03_final_closure.py`. Steps 03–06 need step 01 first; 06 also needs 02.

### Step 07: diagnostics

Original code of the published diagnostics, with input paths adapted to `../work/`. Run after step 01,
from inside the folder:

```bash
cd 07_diagnostics
python expD_kernel.py        # memory kernel, seconds
python exp_synthetic.py      # synthetic test, ~4 min; compare output with ../reference/synthetic_results.txt
python expC_real.py          # MLP / DeepONet / FNO on 2008, ~6 min
python expF_anomaly.py       # diurnal cycle + anomaly predictability, ~6 min
python compare.py            # largest absolute difference from ../reference/
```

Neural-network results can move in the last digits with another jax version or CPU; the kernel is exact.

### Step 08: symbolic regression (PySR)

```bash
python 08_pysr/prepare_data.py     # here, after step 03: writes 08_pysr/pysr_data.npz
cd 08_pysr && python run_pysr.py   # on a machine with Julia; env NSEED=10 NITER=60
```

The protocol and the decision rule (fixed before running) are in the header of `run_pysr.py`.
The genetic search is not deterministic: the expressions change between runs, while the ingredients
(θ as a restoring term, r and ṙ) were stable in our runs.

## How the code is organized

```
rebio.py              shared functions (read this first)
01_prepare.py ... 06_explore.py, 09_figures.py
                      one script per result, top to bottom, no hidden state
07_diagnostics/       neural-network and kernel diagnostics (original code)
08_pysr/              PySR check (original code)
reference/            original outputs, used only by the self-checks
data/                 input spreadsheet goes here (not distributed)
work/, results/       created by the scripts (intermediate files; JSON results and figures)
run_all.py            runs 01-06 and 09 and counts the checks
```

`rebio.py` has eight short sections: paths and constants; tower data (one season of one year, with
the calendar correction); 2008 identification data; weak form; models; RK4 integration and error
metrics; day-block bootstrap; and the 58-term candidate library.

Each step ends with lines such as `check dry coefficients ... same`, comparing the new number with the
original result in `reference/` (absolute tolerance 1e-6 unless stated).

## Method in brief

- **Baseline and residual.** α\*, γ\* are refitted per season by minimizing the RK4 trajectory error
  over the 2008 season. Only their ratio is well determined by the data. The residual is
  s_exp = (dT/dt)_obs − α\*R + γ\*T⁴.
- **Standardization.** All candidates use θ = (T − T̄)/σ_T, r = (R − R̄)/σ_R, ṙ = (dR/dt − mean)/σ_Ṙ,
  computed on the training block (raw T and its powers are nearly collinear; condition number ~10¹⁸).
- **Weak form** (`rebio.weak_system`). The equation is multiplied by a test function
  φ(u) = (1 − u²)⁴ on an 8 h window and integrated by parts, so the measured T is never differentiated:
  −∫φ′T dt − ∫φ(α\*R − γ\*T⁴) dt = Σ_k ξ_k ∫φ Θ_k dt. Windows every hour (selection) or half hour
  (final fit) give a linear system Gξ = b.
- **Splits.** Contiguous blocks of whole days per season: fit (first 56%), validation (56–70%),
  test (last 30%, never used for any choice).
- **Selection.** Forward stepwise least squares on the weak system; candidates accepted only with a
  stable fixed point under mean forcing; number of terms = smallest within 5% of the best validation
  error of the integrated temperature.
- **Evaluation.** RK4 with 6 sub-steps per half hour; daily rollouts (each day restarted from its first
  observed T) and free runs; a run leaving 270–330 K counts as diverged.
- **Uncertainty.** Day-block bootstrap: 1000 resamples for coefficients (seed 0), 5000 for differences in
  pooled daily RMSE (seed 1).
- **Comparator.** A 15-term polynomial in θ and r (all monomials of degree ≤ 4), fitted by OLS to the
  central-difference residual on the same training block, with the same α\*, γ\*. It reproduces the
  functional form of Eq. (8) of da Silva et al. (2026), not its fitting procedure.

## Main results (reproduced by `run_all.py`)

| | Dry | Wet |
| --- | --- | --- |
| c_T, c_Ṙ, c_R (K h⁻¹ per s.d.) | −0.205, +0.644, +0.268 | −0.163, +0.737, +0.410 |
| Relaxation time 1/λ (h) | 18.8 [16.9, 20.9] | 13.6 [10.9, 17.6] |
| 2008 test, daily-rollout RMSE: three terms / polynomial (K) | 1.55 / 1.62 | 1.05 / 1.06 |
| Difference [95% CI] (K) | −0.06 [−0.24, +0.13] | −0.00 [−0.13, +0.13] |
| 2006, frozen coefficients, difference [95% CI] (K) | −0.25 [−0.42, −0.09] | −0.03 [−0.10, +0.04] |

The three-term closure never diverged in 2006–2010; the polynomial diverged on some days in seven
season-years. See the manuscript for all tables and caveats.

## Known details

- Gap rule: the 2008 identification interpolates gaps up to 2 h; validation (steps 05–06) up to 1 h.
  This shifts the 2008 test-block boundary by about one day between the two code paths; both values
  are reported in the manuscript.
- Table V, dry season, "hour of day only": the code gives 0.5048 (0.50 in the manuscript).
- α\* and γ\* are fitted on the whole 2008 season, so the 2008 test block is not fully independent of
  them; the other years are.

## Citation

If you use this code, please cite the manuscript (reference to be added on publication) and the
original model:

da Silva et al. (2026). *Chaos* 36, 033134.

## License

Code: MIT (see `LICENSE`). The license covers the code only, not the Rebio Jaru data.
