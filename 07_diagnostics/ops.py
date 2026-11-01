"""
07 · ops.py · the neural networks
=================================

IN ONE LINE
    Defines and trains the four neural networks compared in the paper. Computes nothing by itself.

THE FOUR MODELS (each predicts s on the 48 half hours of one day)
    pointwise MLP   s(t) = f(T(t), R(t))                       the network of the 2026 paper (641 parameters)
    lag MLP         s(t) = f(T(t), R(t), R 1 h ago ... R 24 h ago)
    DeepONet        reads R over two days and T of the previous day, outputs the whole day
    FNO             Fourier neural operator: same inputs, works in frequency space

WHAT IS INSIDE
    layers / models   the four networks (written with jax)
    make_days         cut the series into one sample per day
    standardize       put inputs on the same scale (training statistics only)
    fit               train with AdamW; stop early when the validation days stop improving
    rollout           run dT/dt = a R - g T^4 + s over each test day (RK4) to get T

NEEDS    jax, optax, numpy
"""
import jax, jax.numpy as jnp, numpy as np, optax

NPD = 48  # half-hourly points per day


# ---------------------------------------------------------------- layers
def mlp_init(key, sizes):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / (a + b)), jnp.zeros(b)))
    return ps


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = ps[-1]
    return x @ W + b


def n_params(p):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(p)))


# ---------------------------------------------------------------- models
# Each model: init(key, dims) -> params ; apply(params, batch) -> s_hat (B, 48)
# batch is a dict built by make_batch (all inputs already standardized).

def pw_init(key, d):
    return mlp_init(key, [2, 32, 16, 1])  # 641 params, same as the paper


def pw_apply(p, b):
    x = jnp.stack([b['T_day'], b['R_day']], -1)  # (B,48,2)
    return mlp(p, x)[..., 0]


def lag_init(key, d):
    return mlp_init(key, [2 + d['nlag'], 32, 16, 1])


def lag_apply(p, b):
    x = jnp.concatenate([jnp.stack([b['T_day'], b['R_day']], -1), b['R_lags']], -1)
    return mlp(p, x)[..., 0]


def don_init(key, d, p=32):
    k1, k2 = jax.random.split(key)
    nin = 2 * NPD + NPD  # R over two days + T over previous day
    return {'branch': mlp_init(k1, [nin, 64, p]),
            'trunk': mlp_init(k2, [4, 32, p]),
            'b0': jnp.zeros(())}


def _tfeat(t):
    w = 2 * jnp.pi * t
    return jnp.stack([jnp.sin(w), jnp.cos(w), jnp.sin(2 * w), jnp.cos(2 * w)], -1)


def don_apply(p, b):
    u = jnp.concatenate([b['R_2d'], b['T_prev']], -1)          # (B,144)
    br = mlp(p['branch'], u)                                     # (B,p)
    t = jnp.arange(NPD) / NPD
    tr = jnp.tanh(mlp(p['trunk'], _tfeat(t)))                    # (48,p)
    return br @ tr.T + p['b0']


def fno_init(key, d, width=16, modes=12, layers=3, cin=4):
    ks = jax.random.split(key, 2 * layers + 2)
    sc = 1.0 / (width * width)
    p = {'lift': mlp_init(ks[0], [cin, width]),
         'proj': mlp_init(ks[1], [width, 32, 1]), 'spec': [], 'loc': []}
    for l in range(layers):
        kr, ki = jax.random.split(ks[2 + l])
        p['spec'].append((sc * jax.random.normal(kr, (width, width, modes)),
                          sc * jax.random.normal(ki, (width, width, modes))))
        p['loc'].append(mlp_init(ks[2 + layers + l], [width, width])[0])
    return p


def fno_apply(p, b, pad=32):
    n = 2 * NPD
    t = jnp.linspace(0, 2, n, endpoint=False)
    B = b['R_2d'].shape[0]
    Tch = jnp.concatenate([b['T_prev'], jnp.zeros((B, NPD))], -1)  # current-day T unknown
    mask = jnp.concatenate([jnp.ones(NPD), jnp.zeros(NPD)])
    x = jnp.stack([b['R_2d'], Tch, jnp.broadcast_to(mask, (B, n)),
                   jnp.broadcast_to(jnp.sin(2 * jnp.pi * t), (B, n))], -1)
    x = mlp(p['lift'], x)                               # (B,n,w)
    x = jnp.pad(x, ((0, 0), (0, pad), (0, 0)))         # non-periodic window
    for (wr, wi), (Wl, bl) in zip(p['spec'], p['loc']):
        m = wr.shape[-1]
        xf = jnp.fft.rfft(x, axis=1)[:, :m, :]          # (B,m,w)
        yf = jnp.einsum('bmi,iom->bmo', xf, wr + 1j * wi)
        y = jnp.fft.irfft(yf, n=x.shape[1], axis=1)
        x = jax.nn.gelu(y + x @ Wl + bl)
    x = x[:, NPD:n, :]
    return mlp(p['proj'], x)[..., 0]


MODELS = {'pointwise MLP': (pw_init, pw_apply),
          'lag MLP': (lag_init, lag_apply),
          'DeepONet': (don_init, don_apply),
          'FNO': (fno_init, fno_apply)}

LAGS = list(range(2, 49, 2))  # 1 h ... 24 h, in half-hour steps


# ---------------------------------------------------------------- data
def make_days(T, R, s, days):
    """Build per-day samples for day indices `days` (each needs day-1)."""
    out = {k: [] for k in ['T_day', 'R_day', 'R_lags', 'R_2d', 'T_prev', 's', 'T_obs']}
    for d in days:
        i0 = d * NPD
        out['T_day'].append(T[i0:i0 + NPD]); out['R_day'].append(R[i0:i0 + NPD])
        out['R_2d'].append(R[i0 - NPD:i0 + NPD]); out['T_prev'].append(T[i0 - NPD:i0])
        out['R_lags'].append(np.stack([R[i0 - L:i0 + NPD - L] for L in LAGS], -1))
        out['s'].append(s[i0:i0 + NPD]); out['T_obs'].append(T[i0:i0 + NPD])
    return {k: np.asarray(v) for k, v in out.items()}


def standardize(batch, stats):
    mT, sT, mR, sR, ms, ss = stats
    return {'T_day': (batch['T_day'] - mT) / sT, 'R_day': (batch['R_day'] - mR) / sR,
            'R_lags': (batch['R_lags'] - mR) / sR, 'R_2d': (batch['R_2d'] - mR) / sR,
            'T_prev': (batch['T_prev'] - mT) / sT, 's': (batch['s'] - ms) / ss}


def stats_of(b):
    return (b['T_day'].mean(), b['T_day'].std(), b['R_day'].mean(), b['R_day'].std(),
            b['s'].mean(), b['s'].std())


# ---------------------------------------------------------------- training
def fit(name, tr, va, seed=0, steps=3000, lr=3e-3, wd=1e-4):
    """Full-batch AdamW with early stopping on validation days. Returns best params."""
    init, apply = MODELS[name]
    p = init(jax.random.PRNGKey(seed), {'nlag': len(LAGS)})
    opt = optax.adamw(lr, weight_decay=wd)
    st = opt.init(p)
    loss = lambda p, b: jnp.mean((apply(p, b) - b['s']) ** 2)

    @jax.jit
    def step(p, st):
        l, g = jax.value_and_grad(loss)(p, tr)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    vl = jax.jit(loss)
    best, bp, bad = np.inf, p, 0
    for i in range(steps):
        p, st, _ = step(p, st)
        if i % 25 == 0:
            v = float(vl(p, va))
            if v < best - 1e-5:
                best, bp, bad = v, p, 0
            else:
                bad += 1
                if bad > 20:
                    break
    return bp, apply


def r2(y, yh):
    return 1 - np.mean((y - yh) ** 2) / np.var(y)


# ---------------------------------------------------------------- rollout
def rollout(name, p, apply, raw, stats, a_h, g_h, dt=0.5):
    """Integrate dT/dt = a R - g T^4 + s over each test day (RK4, R held per step).

    Operators predict s for the whole day up front from exogenous/past inputs;
    pointwise and lag MLPs evaluate s on the simulated temperature (feedback).
    """
    mT, sT, mR, sR, ms, ss = stats
    B = raw['T_obs'].shape[0]
    Tm = np.zeros((B, NPD)); Tm[:, 0] = raw['T_obs'][:, 0]
    if name in ('DeepONet', 'FNO'):
        S = np.asarray(apply(p, standardize(raw, stats))) * ss + ms
        sfun = lambda i, T: S[:, i]
    else:
        def sfun(i, T):
            b = {'T_day': ((T - mT) / sT)[:, None], 'R_day': ((raw['R_day'][:, i] - mR) / sR)[:, None],
                 'R_lags': ((raw['R_lags'][:, i] - mR) / sR)[:, None, :]}
            return np.asarray(apply(p, b))[:, 0] * ss + ms
    for i in range(1, NPD):
        r = raw['R_day'][:, i - 1]
        f = lambda T: a_h * r - g_h * T ** 4 + sfun(i - 1, T)
        x = Tm[:, i - 1]
        k1 = f(x); k2 = f(x + dt * k1 / 2); k3 = f(x + dt * k2 / 2); k4 = f(x + dt * k3)
        Tm[:, i] = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return Tm


def rollout_physics(raw, a_h, g_h, dt=0.5, s_const=0.0):
    B = raw['T_obs'].shape[0]
    Tm = np.zeros((B, NPD)); Tm[:, 0] = raw['T_obs'][:, 0]
    for i in range(1, NPD):
        r = raw['R_day'][:, i - 1]
        f = lambda T: a_h * r - g_h * T ** 4 + s_const
        x = Tm[:, i - 1]
        k1 = f(x); k2 = f(x + dt * k1 / 2); k3 = f(x + dt * k2 / 2); k4 = f(x + dt * k3)
        Tm[:, i] = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return Tm
