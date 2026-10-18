"""Shared constants, data loading, window construction and the network for DeepQCD.

A window is 10 points of one synthetic curve M(omega) -- one ensemble FM (lattice
spacing a), momentum P and replica Exp -- on the 0.1 omega grid, spaced by a stride
of s grid steps:

    omega_i, omega_i + 0.1 s, ..., omega_i + 0.9 s

The first N_SEED = 4 M values are given and the network predicts the next N_PRED = 6.

Network input  (17 values per window):
    a^2  (a in fm from FM_TO_A; squared here, not in the data),  P,
    the 10 omegas / OMEGA_MAX,  the 4 seeds divided by the last seed,  log(last seed)
Network output (6 values): log(M_k / last seed) for the 6 predicted points, so every
prediction is positive and the net learns the curve's shape relative to its level.
Two heads produce it:
    direct  the MLP outputs the 6 values freely
    m3      the MLP outputs the M3 parameters (log A, s) of M = A w exp(-s w) for this
            window, and the 6 values are that curve at the target omegas

Paths are relative to Helicity/DeepQCD, so run every script from that directory.
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

DATA_IN = '../Data/2026/'
OUT_DIR = '../Data/DeepQCD/'
IMG_DIR = os.path.join(OUT_DIR, 'Images')
MODEL_DIR = '../model/DeepQCD/'
GEN_FILE = os.path.join(DATA_IN, 'generated_Synthetic_exp_data.csv')
LAG_MODEL_DIR = '../model/2026/'        # the existing per-FM 4-lag models (baseline)

FM_TO_A = {9: 0.0888, 12: 0.1207, 15: 0.151}   # fm, as in Code/continuum_M_extrapolation.py
FMS = sorted(FM_TO_A)
P_SET = np.array([1, 2, 3, 4, 5])
W_STEP = 0.1
N_W = 201                                # omega = 0.0 .. 20.0
OMEGA_MAX = (N_W - 1) * W_STEP
WIN = 10
N_SEED = 4
N_PRED = WIN - N_SEED
DEFAULT_STRIDES = [1, 2, 5, 10]
HEADS = ('direct', 'm3')
# input groups that train.py --drop can switch off (feature column indices, see make_features)
DROPPABLE = {'a': [0], 'level': [2 + WIN + N_SEED]}
PLOT_DPI = 200

for _d in (OUT_DIR, IMG_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)


def model_spec(head, drop=(), tag=''):
    """e.g. 'm3', 'm3-noa', 'direct-noa-nolevel', 'direct-noa-nolevel-wide' (tag 'wide')."""
    return head + ''.join(f'-no{d}' for d in drop) + (f'-{tag}' if tag else '')


def run_tag(test_fm, head, drop=(), tag=''):
    return f'{model_spec(head, drop, tag)}_' + ('all' if test_fm is None else f'test_FM{test_fm}')


def model_path(test_fm, head, drop=(), tag=''):
    """Checkpoint of the ``head`` model (inputs ``drop`` switched off, optional name
    ``tag``) trained without ``test_fm`` (None: on every FM)."""
    return os.path.join(MODEL_DIR, f'deepqcd_{run_tag(test_fm, head, drop, tag)}.pt')


def find_models(test_fm, specs=None):
    """{'DeepQCD_<spec>': path} of every saved checkpoint for ``test_fm`` (None: all-FM
    models), optionally only the given specs."""
    suffix = '_all.pt' if test_fm is None else f'_test_FM{test_fm}.pt'
    out = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, f'deepqcd_*{suffix}'))):
        spec = os.path.basename(path)[len('deepqcd_'):-len(suffix)]
        if specs is None or spec in specs:
            out[f'DeepQCD_{spec}'] = path
    return out


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
    from the generator output, cached in OUT_DIR/curves.npz."""
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
        M = np.full((len(P_SET), ei.max() + 1, N_W), np.nan)
        M[pi, ei, wi[sel]] = d.M.to_numpy()[sel]
        est = np.full((len(P_SET), N_W), np.nan)
        est[pi, wi[sel]] = d.Est_M.to_numpy()[sel]
        if np.isnan(M).any() or np.isnan(est).any():
            raise ValueError(f'FM={fm}: generated data does not fill the (P, Exp, omega) grid')
        out[fm] = {'M': M, 'est': est}
        save[f'M_{fm}'], save[f'est_{fm}'] = M, est
    np.savez(cache, **save)
    return out


def make_features(a, P, w_win, m_seed):
    """Network input, one row per window (see module docstring)."""
    m_last = m_seed[:, -1:]
    return np.column_stack([np.asarray(a, dtype=float) ** 2, P, w_win / OMEGA_MAX,
                            m_seed / m_last, np.log(m_last)]).astype(np.float32)


def to_target(m_pred, m_seed):
    return np.log(m_pred / m_seed[:, -1:]).astype(np.float32)


def from_target(z, m_seed):
    return m_seed[:, -1:] * np.exp(z)


def sample_windows(curves, fms, n, strides, rng, exps=None):
    """n random windows (fm, P, Exp, stride, start) from ``fms``, anywhere on the omega
    grid; ``exps`` = {fm: replica indices to draw from} (default: all replicas).
    Returns a, P, w (n, WIN), m (n, WIN).
    """
    fm = rng.choice(fms, n)
    stride = rng.choice(strides, n)
    start = (rng.random(n) * (N_W - (WIN - 1) * stride)).astype(int)
    p = rng.integers(0, len(P_SET), n)
    idx = start[:, None] + stride[:, None] * np.arange(WIN)
    m = np.empty((n, WIN))
    for f in fms:
        s = fm == f
        pool = exps[f] if exps is not None else np.arange(curves[f]['M'].shape[1])
        exp = rng.choice(pool, s.sum())
        m[s] = curves[f]['M'][p[s][:, None], exp[:, None], idx[s]]
    a = np.array([FM_TO_A[f] for f in FMS])[np.searchsorted(FMS, fm)]
    return a, P_SET[p], idx * W_STEP, m


class WindowNet(nn.Module):
    """MLP from raw window features to the standardised target; the input/target (and,
    for the m3 head, M3-parameter) scaling is kept in buffers so a checkpoint is
    self-contained."""

    def __init__(self, n_in, n_out, hidden=256, layers=3, head='direct', drop=()):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f'head must be one of {HEADS}')
        self.head = head
        mask = torch.ones(n_in)
        for d in drop:
            mask[DROPPABLE[d]] = 0.0                            # dropped inputs are fed as 0
        self.register_buffer('in_mask', mask)
        blocks, d = [], n_in
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.net = nn.Sequential(*blocks, nn.Linear(d, n_out if head == 'direct' else 2))
        self.register_buffer('x_mean', torch.zeros(n_in))
        self.register_buffer('x_std', torch.ones(n_in))
        self.register_buffer('y_mean', torch.zeros(n_out))
        self.register_buffer('y_std', torch.ones(n_out))
        self.register_buffer('p_mean', torch.zeros(2))        # m3 head: (log A, s)
        self.register_buffer('p_std', torch.ones(2))

    def set_scaling(self, X, Y, params=None):
        """Standardisation from a training sample; ``params`` (n, 2) = per-window M3
        (log A, s) seed fits, whose median / IQR scale the m3 head's outputs."""
        stats = [('x_mean', X.mean(0)), ('x_std', X.std(0)), ('y_mean', Y.mean(0)), ('y_std', Y.std(0))]
        if params is not None:
            q25, q50, q75 = np.nanpercentile(params, [25, 50, 75], axis=0)
            stats += [('p_mean', q50), ('p_std', (q75 - q25) / 1.349)]
        for name, v in stats:
            if name.endswith('std'):
                v = np.where(v > 0, v, 1.0)
            getattr(self, name).copy_(torch.as_tensor(v, dtype=torch.float32))

    def forward(self, x):
        h = self.net((x - self.x_mean) / self.x_std * self.in_mask)
        if self.head == 'direct':
            return h
        theta = h * self.p_std + self.p_mean                    # (log A, s)
        w = x[:, 2 + N_SEED:2 + WIN] * OMEGA_MAX                # target omegas (> 0)
        z = theta[:, :1] + torch.log(w) - theta[:, 1:] * w - x[:, -1:]   # log(M / last seed)
        return self.scale_target(z)

    def scale_target(self, y):
        return (y - self.y_mean) / self.y_std

    def unscale_target(self, z):
        return z * self.y_std + self.y_mean


def load_model(path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    model = WindowNet(cfg['n_in'], N_PRED, cfg['hidden'], cfg['layers'], cfg['head'], cfg.get('drop', ()))
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if unexpected or set(missing) - {'in_mask'}:      # in_mask: older checkpoints, no inputs dropped
        raise RuntimeError(f'{path}: missing {missing}, unexpected {unexpected}')
    return model.to(device).eval(), cfg


@torch.no_grad()
def predict_m(model, a, P, w_win, m_seed, device, batch=262144):
    """Predicted M at the 6 points after the seeds, shape (n, N_PRED)."""
    X = make_features(a, P, w_win, m_seed)
    z = np.concatenate([
        model.unscale_target(model(torch.from_numpy(X[i:i + batch]).to(device))).cpu().numpy()
        for i in range(0, len(X), batch)])
    return from_target(z, m_seed)


def rollout(model, a, P, m_seed, stride, device):
    """Extrapolate whole curves from the 4 seeds at omega = 0, 0.1s, 0.2s, 0.3s.

    Each block predicts 6 new points; its last 4 points seed the next block.  The
    final block is shifted back so it ends exactly at OMEGA_MAX.  Returns the
    stride grid omegas and M, shape (n, n_points).
    """
    if (N_W - 1) % stride:
        raise ValueError(f'stride {stride} must divide {N_W - 1}')
    return rollout_dw(model, a, P, m_seed, stride * W_STEP, (N_W - 1) // stride + 1, device)


def rollout_dw(model, a, P, m_seed, dw, n_pts, device):
    """``rollout`` on the grid omega = k * dw, k = 0 .. n_pts-1, for any spacing dw
    (e.g. the real lattice omega = dw * z); the seeds are the first 4 grid points."""
    if n_pts < WIN:
        raise ValueError(f'spacing {dw} leaves fewer than {WIN} grid points')
    m = np.full((len(P), n_pts), np.nan)
    m[:, :N_SEED] = m_seed
    front = N_SEED - 1                           # last known point on the grid
    while front < n_pts - 1:
        start = min(front - (N_SEED - 1), n_pts - WIN)
        idx = start + np.arange(WIN)
        w = np.broadcast_to(idx * dw, (len(P), WIN))
        pred = predict_m(model, a, P, w, m[:, idx[:N_SEED]], device)
        new = idx[N_SEED:] > front
        m[:, idx[N_SEED:][new]] = pred[:, new]
        front = idx[-1]
    return np.arange(n_pts) * dw, m


def m3_seed_params(w_seed, m_seed):
    """M3 for one (FM, P), M = A w exp(-s w) (P^-gamma is absorbed in A), fit to the
    seeds with omega > 0 by least squares on log(M / w) = log A - s w.
    Returns log A and s, each shape (n, 1)."""
    ok = w_seed > 0
    x = np.where(ok, w_seed, np.nan)
    y = np.where(ok, np.log(np.where(ok, m_seed, 1.0) / np.where(ok, w_seed, 1.0)), np.nan)
    xm, ym = np.nanmean(x, 1, keepdims=True), np.nanmean(y, 1, keepdims=True)
    slope = np.nansum((x - xm) * (y - ym), 1, keepdims=True) / np.nansum((x - xm) ** 2, 1, keepdims=True)
    return ym - slope * xm, -slope


def m3_seed_fit(w_seed, m_seed, w_eval):
    """Baseline: the seed-fitted M3 curve evaluated at ``w_eval``."""
    log_a, s = m3_seed_params(w_seed, m_seed)
    return np.exp(log_a) * w_eval * np.exp(-s * w_eval)
