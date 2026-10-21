"""Shared constants, data, windows and the constrained network for DeepQCDConstraint.

Physics constraint: for momenta i < j at the same omega

    M_i / M_j < j^2 / i^2    <=>    i^2 M_i < j^2 M_j,

i.e. every momentum-pair estimate I_ij = (j^2 M_j - i^2 M_i) / (j^2 - i^2) is positive.

DeepQCD predicts each P on its own, so it cannot see this.  Here ONE network predicts
every P at once.  At each target omega it outputs two numbers, log(M_1 / s) and
log(I / s) (s = the window's seed scale), and

    M_P = I + (M_1 - I) / P^2,    i.e.    P^2 M_P = M_1 + (P^2 - 1) I.

With I > 0 and M_1 > 0 this guarantees, by construction,
    * P^2 M_P rises strictly with P (the constraint above, for every pair),
    * M_P > 0 for every P,
    * every momentum pair gives exactly the same I(omega) -- the generator's own form
      M_P = I + Y / P^2 (Code/synthetic_data_generator.py) with Y = M_1 - I.

A window: for every P, 4 seed points (omega, M), and 6 target omegas shared by all P.
Two seed layouts are trained:
    grid     all P seeded at the same omegas  w0 + k*dw, k = 0..3; targets k = 4..9
             (every rollout block after the first is this layout)
    lattice  P seeded at its own lattice omegas  P * z * d1, z = 0..3 (the real data,
             d1 in [D1_MIN, D1_MAX]); targets dw * (1..6)

Network input (46 values): the 20 seed omegas / OMEGA_MAX, the 20 seed M / s,
the 6 target omegas / OMEGA_MAX.  s = the largest seed M of the window.  No a^2 and
no absolute level input (DeepQCD/RESULTS.md: both stop it generalising to a new FM).

Paths are relative to Helicity/DeepQCDConstraint, so run every script from there.
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

DATA_IN = '../Data/2026/'
OUT_DIR = '../Data/DeepQCDConstraint/'
IMG_DIR = os.path.join(OUT_DIR, 'images')
MODEL_DIR = '../model/DeepQCDConstraint/'
GEN_FILE = os.path.join(DATA_IN, 'generated_Synthetic_exp_data.csv')

FM_TO_A = {9: 0.0888, 12: 0.1207, 15: 0.151}   # fm, as in Code/continuum_M_extrapolation.py
FMS = sorted(FM_TO_A)
P_SET = np.array([1, 2, 3, 4, 5])
N_P = len(P_SET)
P2 = (P_SET ** 2).astype(float)
W_STEP = 0.1
N_W = 201                                # omega = 0.0 .. 20.0
OMEGA_MAX = (N_W - 1) * W_STEP
WIN = 10
N_SEED = 4
N_PRED = WIN - N_SEED
DEFAULT_STRIDES = [1, 2, 5, 10]
D1_MIN, D1_MAX = 0.15, 0.45              # real d1 = omega(P=1, z=1): 0.196 / 0.262 / 0.393
MIN_P = 2                                # smallest P in an I(omega) pair, as in Code/test_error.py
PLOT_DPI = 200

for _d in (OUT_DIR, IMG_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)


def run_tag(test_fm, tag=''):
    return ('dqc' + (f'-{tag}' if tag else '') + '_'
            + ('all' if test_fm is None else f'test_FM{test_fm}'))


def model_path(test_fm, tag=''):
    """Checkpoint trained without ``test_fm`` (None: on every FM)."""
    return os.path.join(MODEL_DIR, f'{run_tag(test_fm, tag)}.pt')


def split_replicas(curves, fms, val_frac, rng):
    """Per FM, a random ``val_frac`` of the replicas for validation, the rest for training."""
    train, val = {}, {}
    for fm in fms:
        perm = rng.permutation(curves[fm]['M'].shape[1])
        n_val = int(round(val_frac * len(perm)))
        val[fm], train[fm] = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    return train, val


def get_device(name='auto'):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_curves(refresh=False):
    """{fm: {'M': (nP, nExp, N_W) replica curves, 'est': (nP, N_W) noise-free Est_M}}
    from the generator output, cached in OUT_DIR/curves.npz.  Replica e of every P is
    taken together as one joint sample (the generator draws the P independently)."""
    cache = os.path.join(OUT_DIR, 'curves.npz')
    if (not refresh and os.path.exists(cache)
            and os.path.getmtime(cache) > os.path.getmtime(GEN_FILE)):
        z = np.load(cache)
        return {fm: {'M': z[f'M_{fm}'], 'est': z[f'est_{fm}']} for fm in FMS}

    d = pd.read_csv(GEN_FILE, usecols=['FM', 'W', 'P', 'Est_M', 'Exp', 'M'])
    if set(d.P.unique()) != set(P_SET) or set(d.FM.unique()) != set(FMS):
        raise ValueError(f'{GEN_FILE}: expected P {list(P_SET)} and FM {FMS}')
    wi = np.rint(d.W.to_numpy() / W_STEP).astype(int)
    out, save = {}, {}
    for fm in FMS:
        sel = (d.FM == fm).to_numpy()
        pi = np.searchsorted(P_SET, d.P.to_numpy()[sel])
        ei = d.Exp.to_numpy()[sel]
        M = np.full((N_P, ei.max() + 1, N_W), np.nan)
        M[pi, ei, wi[sel]] = d.M.to_numpy()[sel]
        est = np.full((N_P, N_W), np.nan)
        est[pi, wi[sel]] = d.Est_M.to_numpy()[sel]
        if np.isnan(M).any() or np.isnan(est).any():
            raise ValueError(f'FM={fm}: generated data does not fill the (P, Exp, omega) grid')
        out[fm] = {'M': M, 'est': est}
        save[f'M_{fm}'], save[f'est_{fm}'] = M, est
    np.savez(cache, **save)
    return out


def interp_grid(curve, w):
    """Linear interpolation of grid curves ``curve`` (..., N_W) at omegas ``w`` (..., k)."""
    x = np.clip(np.asarray(w) / W_STEP, 0, N_W - 1)
    i0 = np.minimum(np.floor(x).astype(int), N_W - 2)
    f = x - i0
    lo = np.take_along_axis(curve, i0, axis=-1)
    hi = np.take_along_axis(curve, i0 + 1, axis=-1)
    return lo + f * (hi - lo)


def i_from_pairs(M, ps=P_SET):
    """Signed I(omega) of M (nP, ..., n_w) averaged over every pair MIN_P <= p_i < p_j.
    On constrained-model output every pair gives the same I (the network's own I)."""
    ps = np.asarray(ps)
    use = [k for k, p in enumerate(ps) if p >= MIN_P]
    E = [(ps[j] ** 2 * M[j] - ps[i] ** 2 * M[i]) / (ps[j] ** 2 - ps[i] ** 2)
         for i in use for j in use if i < j]
    return np.mean(E, axis=0)


def violation_rate(M, axis=1):
    """Fraction of (sample, omega) cells where P^2 M_P is not strictly rising in P,
    for M with the P axis at ``axis`` (cells with every M == 0, i.e. omega = 0, skipped)."""
    q = np.moveaxis(np.asarray(M, dtype=float), axis, 0) * P2.reshape((-1,) + (1,) * (np.ndim(M) - 1))
    bad = (np.diff(q, axis=0) <= 0).any(axis=0)
    live = (q != 0).any(axis=0)
    return float(bad[live].mean()) if live.any() else 0.0


def sample_windows(curves, fms, n, strides, rng, exps=None, lattice_frac=0.5):
    """n random joint windows from ``fms``; ``exps`` = {fm: replica pool} (default all).
    Returns w_seed (n, nP, 4), m_seed (n, nP, 4), w_tgt (n, 6), m_tgt and est_tgt (n, nP, 6)."""
    fm = rng.choice(fms, n)
    dw = rng.choice(strides, n) * W_STEP
    lat = rng.random(n) < lattice_frac
    k = np.arange(WIN)

    start = (rng.random(n) * (OMEGA_MAX - (WIN - 1) * dw) / W_STEP).astype(int) * W_STEP
    w_grid = start[:, None] + dw[:, None] * k                                   # (n, WIN)
    w_seed = np.repeat(w_grid[:, None, :N_SEED], N_P, axis=1)
    w_tgt = w_grid[:, N_SEED:].copy()

    d1 = rng.uniform(D1_MIN, D1_MAX, n)
    w_seed[lat] = (P_SET[None, :, None] * np.arange(N_SEED)[None, None, :]) * d1[lat, None, None]
    w_tgt[lat] = dw[lat, None] * np.arange(1, N_PRED + 1)
    w_tgt = np.round(w_tgt, 10)

    m_seed = np.empty((n, N_P, N_SEED))
    m_tgt = np.empty((n, N_P, N_PRED))
    est_tgt = np.empty((n, N_P, N_PRED))
    for f in fms:
        s = np.flatnonzero(fm == f)
        if not len(s):
            continue
        pool = exps[f] if exps is not None else np.arange(curves[f]['M'].shape[1])
        rep = np.transpose(curves[f]['M'][:, rng.choice(pool, len(s)), :], (1, 0, 2))   # (ns, nP, N_W)
        m_seed[s] = interp_grid(rep, w_seed[s])
        wt = np.repeat(w_tgt[s][:, None, :], N_P, axis=1)
        m_tgt[s] = interp_grid(rep, wt)
        est_tgt[s] = interp_grid(np.broadcast_to(curves[f]['est'], (len(s), N_P, N_W)), wt)
    return w_seed, m_seed, w_tgt, m_tgt, est_tgt


def seed_scale(m_seed):
    """s per window: the largest seed M over every P (n, 1, 1)."""
    s = np.max(m_seed, axis=(1, 2), keepdims=True)
    return np.where(s > 0, s, 1.0)


def make_features(w_seed, m_seed, w_tgt):
    n = len(m_seed)
    return np.column_stack([w_seed.reshape(n, -1) / OMEGA_MAX,
                            (m_seed / seed_scale(m_seed)).reshape(n, -1),
                            w_tgt / OMEGA_MAX]).astype(np.float32)


def loss_weights(m_seed):
    """Per-P error scale (n, nP, 1): that P's largest seed (floored at 1e-3 s), so every
    P counts relative to its own size, as DeepQCD's log(M / last seed) target does."""
    s = seed_scale(m_seed)
    return np.maximum(m_seed.max(axis=2, keepdims=True), 1e-3 * s).astype(np.float32)


class ConstrainedNet(nn.Module):
    """MLP -> (log M_1/s, log I/s) at each target omega -> M_P = I + (M_1 - I) / P^2.
    forward() returns M / s, shape (n, nP, 6); the output scaling lives in buffers so a
    checkpoint is self-contained."""

    def __init__(self, n_in, hidden=256, layers=3):
        super().__init__()
        blocks, d = [], n_in
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.net = nn.Sequential(*blocks, nn.Linear(d, 2 * N_PRED))
        self.register_buffer('x_mean', torch.zeros(n_in))
        self.register_buffer('x_std', torch.ones(n_in))
        self.register_buffer('u_mean', torch.zeros(2 * N_PRED))    # (log M_1/s, log I/s) per target
        self.register_buffer('u_std', torch.ones(2 * N_PRED))
        self.register_buffer('inv_p2', torch.as_tensor(1.0 / P2, dtype=torch.float32))

    def set_scaling(self, X, U):
        """Input standardisation from a training sample X, output scaling from the
        noise-free (log M_1/s, log I/s) U (n, 2*N_PRED): median / IQR, robust to the
        tiny I at large omega."""
        q25, q50, q75 = np.nanpercentile(U, [25, 50, 75], axis=0)
        x_std = X.std(0)
        for name, v in [('x_mean', X.mean(0)), ('x_std', np.where(x_std > 0, x_std, 1.0)),
                        ('u_mean', q50), ('u_std', np.where(q75 > q25, (q75 - q25) / 1.349, 1.0))]:
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32))

    def m1_i(self, x):
        """(M_1 / s, I / s), each (n, 6), both > 0."""
        u = self.net((x - self.x_mean) / self.x_std) * self.u_std + self.u_mean
        u = u.view(-1, N_PRED, 2).clamp(-30.0, 30.0)
        return torch.exp(u[..., 0]), torch.exp(u[..., 1])

    def forward(self, x):
        m1, i = self.m1_i(x)
        return i[:, None, :] + (m1 - i)[:, None, :] * self.inv_p2[None, :, None]


def scaling_targets(m_seed, est_tgt):
    """Noise-free (log M_1/s, log I/s) for set_scaling; I from the P = 1, 2 pair of Est_M."""
    s = seed_scale(m_seed)[:, :, 0]
    m1 = est_tgt[:, 0]
    i = (4 * est_tgt[:, 1] - est_tgt[:, 0]) / 3
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.stack([np.log(m1 / s), np.log(np.maximum(i, 1e-8 * s) / s)], axis=-1)
    return u.reshape(len(m_seed), -1)


def load_model(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = ConstrainedNet(cfg['n_in'], cfg['hidden'], cfg['layers'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict(model, w_seed, m_seed, w_tgt, device, batch=131072):
    """M (n, nP, 6) and I (n, 6) at the target omegas."""
    X = make_features(w_seed, m_seed, w_tgt)
    s = seed_scale(m_seed)
    M, I = [], []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).to(device)
        M.append(model(x).cpu().numpy())
        I.append(model.m1_i(x)[1].cpu().numpy())
    return np.concatenate(M) * s, np.concatenate(I) * s[:, :, 0]


def _roll(model, M, front, dw, device):
    """Grid blocks from ``front`` (last known index) to the end of M (n, nP, n_pts)."""
    n, _, n_pts = M.shape
    while front < n_pts - 1:
        start = min(front - (N_SEED - 1), n_pts - WIN)
        idx = start + np.arange(WIN)
        w = idx * dw
        pred, _ = predict(model, np.broadcast_to(w[:N_SEED], (n, N_P, N_SEED)), M[:, :, idx[:N_SEED]],
                          np.broadcast_to(w[N_SEED:], (n, N_PRED)), device)
        new = idx[N_SEED:] > front
        M[:, :, idx[N_SEED:][new]] = pred[:, :, new]
        front = idx[-1]
    return M


def rollout_grid(model, m_seed, dw, n_pts, device):
    """Extrapolate every P jointly from seeds at omega = 0, dw, 2dw, 3dw on the grid
    omega = k * dw, k = 0 .. n_pts-1.  Returns omega (n_pts,) and M (n, nP, n_pts)."""
    if n_pts < WIN:
        raise ValueError(f'spacing {dw} leaves fewer than {WIN} grid points')
    M = np.full((len(m_seed), N_P, n_pts), np.nan)
    M[:, :, :N_SEED] = m_seed
    return np.arange(n_pts) * dw, _roll(model, M, N_SEED - 1, dw, device)


def rollout_lattice(model, w_seed, m_seed, dw, n_pts, device):
    """Extrapolate from per-P lattice seeds (n, nP, 4) at omega = P z d1: the first block
    predicts the grid points k = 1..6 (k = 0 is the M = 0 anchor), then grid blocks.
    No seed is copied into the output, so every point after omega = 0 is the network's."""
    if n_pts < WIN:
        raise ValueError(f'spacing {dw} leaves fewer than {WIN} grid points')
    n = len(m_seed)
    M = np.full((n, N_P, n_pts), np.nan)
    M[:, :, 0] = 0.0
    M[:, :, 1:N_PRED + 1], _ = predict(model, w_seed, m_seed,
                                       np.broadcast_to(np.arange(1, N_PRED + 1) * dw, (n, N_PRED)), device)
    return np.arange(n_pts) * dw, _roll(model, M, N_PRED, dw, device)
