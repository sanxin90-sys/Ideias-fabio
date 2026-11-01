# 07 · Diagnostics (neural networks, memory kernel, synthetic test)

**What this folder answers:** how much of the residual s can *any* model predict, and does the
method find the right terms when the answer is known? These are diagnostics, not the model.

**Needs:** step 01 run first (`../work/`), plus `pip install jax optax pysindy`.
**Run from inside this folder**, in this order:

| # | Command | In one line | Time |
| --- | --- | --- | --- |
| 1 | `python expD_kernel.py` | How long the air "remembers" radiation (linear memory kernel) | seconds |
| 2 | `python exp_synthetic.py > synthetic_new.txt` | Fake data with a known answer: which method finds it? | ~4 min |
| 3 | `python expC_real.py` | MLP, DeepONet, FNO on real 2008 data | ~6 min |
| 4 | `python expF_anomaly.py` | Is anything predictable beyond the mean daily cycle? | ~6 min |
| 5 | `python compare.py` | Compares 1, 3, 4 with `../reference/` (0 = identical) | seconds |

Check for step 2: `synthetic_new.txt` must be identical to `../reference/synthetic_results.txt`.

**Helper files (not run on their own):**
`ops.py` defines and trains the networks; `common.py` holds the tools of the synthetic test;
`expB_synth_data.py` builds the fake series with a 3-hour memory used by `expD_kernel.py`.
