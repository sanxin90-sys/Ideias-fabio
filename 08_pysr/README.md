# 08 · Symbolic regression (PySR)

**What this folder answers:** if the computer is free to invent *any* formula, does it find
something better than the three-term equation? (Answer in the paper: no; it keeps finding the
same ingredients θ, r and ṙ.)

| # | Command | Where | In one line | Time |
| --- | --- | --- | --- | --- |
| 1 | `python 08_pysr/prepare_data.py` | main folder, after step 03 | Saves the data PySR needs into `pysr_data.npz` | seconds |
| 2 | `cd 08_pysr` then `python run_pysr.py` | a computer with Julia | Runs the search, ranks, refits and tests the formulas | hours |

**Install for step 2:** `pip install "pysr>=0.19" "pynumdiff>=0.3" sympy scipy pandas`.
The first run downloads Julia automatically.
Settings: environment variables `NSEED` (default 10 seeds) and `NITER` (default 60 iterations).

**Outputs:** `results_pysr.json` (the best formulas and their tests), `candidates_pysr.csv`
(every formula found), `run.log`.

**Note:** the search is random, so the exact formulas change between runs. The protocol and the
decision rule were fixed before running; they are written at the top of `run_pysr.py`.
