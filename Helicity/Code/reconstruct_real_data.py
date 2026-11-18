"""Reconstruct/extrapolate the real-data M(omega) curves with the trained models.

For every lattice ensemble (FM) and every (P, Exp) replica:

  * seed a 4-point (w, m) lag window at the first 4 omegas of the 0.1 grid
    (0.0-0.3) from that Exp's synthetic replica only.  Real data -- raw or
    filled (fill_real_data.py) -- is never used: real and synthetic M differ
    by up to ~2x per P and the models only know the synthetic curve, so a real
    seed bends both M and the derived I(omega) out of shape;
  * roll the window forward one synthetic-grid step (0.1) at a time.  At each
    step the per-P GB / RF / XGB / NN / GP model predicts M at the next omega and that
    prediction is fed back in as the newest lag.

``anchored`` is therefore 0 everywhere in this default mode; every point
beyond the seed window is a purely autoregressive model extrapolation with no
teacher forcing.  RF is the least reliable in that
tail -- a random forest cannot extrapolate past its training targets and
tends to flat-line under long autoregression.

Every model in ``MODEL_NAMES`` whose ``.sav`` exists for an FM is rolled and
gets an ``M_{name}`` column; missing models are skipped with a note, so the
output only carries the models that have been trained.

The per-P model variant is chosen with ``--exp_id`` (default ``No`` -> the
``*_exp_No`` models, no Exp feature; ``Yes`` -> ``*_exp_Yes``).  The output
file name carries the variant so both can coexist:
    --exp_id No   -> Real_data_projected_{fm}_no_exp_v2.csv
    --exp_id Yes  -> Real_data_projected_{fm}_with_exp_v2.csv
Per train_error_summary.csv the two are within noise; Exp is an arbitrary
replica index, so for ``No`` it is simply not fed through the roll.

Two optional modes, for uncertainty propagation (continuum_M_extrapolation.py):

  --anchor_real   every 0.1-grid omega inside a (FM, P) group's real data reach
                  takes that Exp's own real M, linearly interpolated from its
                  measured (unrounded) omegas, and the roll starts from the last
                  N_LAGS of those points.  Each replica then follows its own
                  gauge configuration across the whole measured range, so the
                  Exp spread is the real statistical spread there.  Because the
                  real data is used, the test error of test_error.py is NOT
                  out-of-sample for these files; they get an ``_anchored``
                  suffix, which test_error.py does not pick up.
  --n_draws K     GP only: K extra rolls from the Exp-mean seed in which every
                  step draws the GP's *latent* prediction (white-noise term
                  removed) instead of taking its mean.  Successive steps are
                  sampled from the bivariate posterior of the current and the
                  previous step's inputs (a Markov approximation of a joint
                  posterior draw), so model uncertainty accumulates smoothly
                  along the roll.  Draw 0 is the same roll without noise, so a
                  draw minus draw 0 is the model-uncertainty shift.  Written to
                  Real_data_projected_{fm}_{token}[_anchored]_draws_v2.csv
                  (FM, P, Draw, W, M_GP).  The model uncertainty is common to
                  every replica, which is why it is kept apart from the Exp
                  rolls rather than added per replica (averaging over Exp would
                  otherwise wash it out).

Run from ``Helicity/Code`` (paths are relative):
    python reconstruct_real_data.py                 # no_exp
    python reconstruct_real_data.py --exp_id Yes    # with_exp
    python reconstruct_real_data.py --anchor_real --n_draws 200 --models GP
"""

import argparse
import os
import shutil

import joblib
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

# Feature order the models were trained on (prepare_train_test_data.py):
# P [, Exp], then 4 lag (w, m) pairs, then w5.  The *_exp_Yes models carry the
# extra Exp column; *_exp_No models do not.
_LAGS = ['w1', 'm1', 'w2', 'm2', 'w3', 'm3', 'w4', 'm4', 'w5']
FEATURES = {'No': ['P'] + _LAGS, 'Yes': ['P', 'Exp'] + _LAGS}
MODEL_NAMES = ['GB', 'RF', 'XGB', 'NN', 'GP', 'BNN', 'NNE']
N_LAGS = 4          # number of (w, m) lag points feeding each prediction
W_DECIMALS = 1      # omega grid is compared after rounding to this many decimals

# exp_id -> output-filename token
EXP_TOKEN = {'No': 'no_exp', 'Yes': 'with_exp'}


def _anchor_curves(real_raw_p, exps, grid):
    """(n_exp, n_grid) each Exp's real M linearly interpolated onto ``grid``.

    ``real_raw_p`` holds one P's rows with unrounded W (incl. the W=0, M=0
    anchor).  Rows of Exps without real data are NaN.
    """
    out = np.full((len(exps), len(grid)), np.nan)
    piv = real_raw_p.pivot_table(index='Exp', columns='W', values='M', aggfunc='mean')
    piv = piv.reindex(index=exps)
    w = piv.columns.to_numpy(dtype=float)
    for i, row in enumerate(piv.to_numpy(dtype=float)):
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            out[i] = np.interp(grid, w[ok], row[ok])
    return out


class GPDrawRoller:
    """Rolls K+1 lag windows through one per-P GP pipeline, sampling its latent function.

    Row 0 is the noiseless roll; rows 1..K add a draw of the latent GP error at
    every step, correlated with the previous step's draw through the posterior
    covariance of the two input rows.
    """

    def __init__(self, pipe, rng):
        self.scale = pipe.named_steps['scale']
        self.gp = pipe.named_steps['gp']
        noise = [k.noise_level for k in (getattr(self.gp.kernel_, 'k1', None),
                                         getattr(self.gp.kernel_, 'k2', None))
                 if k is not None and hasattr(k, 'noise_level')]
        y_std = float(np.atleast_1d(getattr(self.gp, '_y_train_std', 1.0))[0])
        self.noise_var = (noise[0] if noise else 0.0) * y_std ** 2
        self.rng = rng
        self.prev_X, self.prev_delta = None, None

    def step(self, X):
        """X: (K+1, n_feat) inputs -> sampled latent M, shape (K+1,)."""
        n = len(X)
        Xs = self.scale.transform(X)
        if self.prev_X is None:
            mu, cov = self.gp.predict(Xs, return_cov=True)
            var = np.clip(np.diag(cov) - self.noise_var, 0.0, None)
            delta = np.sqrt(var) * self.rng.standard_normal(n)
        else:
            mu2, cov = self.gp.predict(np.vstack([self.prev_X, Xs]), return_cov=True)
            mu = mu2[n:]
            v_prev = np.clip(np.diag(cov)[:n] - self.noise_var, 1e-30, None)
            var = np.clip(np.diag(cov)[n:] - self.noise_var, 0.0, None)
            c = np.diag(cov[:n, n:])
            # latent error | previous step's latent error (bivariate Gaussian)
            cond_mean = c / v_prev * self.prev_delta
            cond_var = np.clip(var - c ** 2 / v_prev, 0.0, None)
            delta = cond_mean + np.sqrt(cond_var) * self.rng.standard_normal(n)
        delta[0] = 0.0                                  # row 0: noiseless reference roll
        self.prev_X, self.prev_delta = Xs, delta
        return mu + delta


class GPReplicaSampler:
    """Per-replica latent-GP sampling inside the Exp roll (--propagate).

    Each row (replica) gets its own latent error at every step, drawn from the
    posterior conditional on that row's previous-step error (bivariate
    Gaussian of the current and previous inputs, Markov approximation), so the
    model uncertainty accumulates along omega independently per replica.
    Only the diagonal / row-wise cross covariances are computed, so this scales
    to all ~1900 replicas.  Assumes kernel = Constant * RBF + White.
    """

    def __init__(self, pipe, rng):
        self.scale = pipe.named_steps['scale']
        gp = pipe.named_steps['gp']
        k = gp.kernel_
        if not (hasattr(k, 'k1') and hasattr(k.k1, 'k1') and hasattr(k.k1.k2, 'length_scale')):
            raise ValueError(f'GPReplicaSampler: unexpected kernel {k}')
        self.gp = gp
        self.latent = k.k1                          # Constant * RBF (white noise excluded)
        self.amp = float(k.k1.k1.constant_value)
        self.ls = np.asarray(k.k1.k2.length_scale, dtype=float)
        self.y_std = float(np.atleast_1d(getattr(gp, '_y_train_std', 1.0))[0])
        self.y_mean = float(np.atleast_1d(getattr(gp, '_y_train_mean', 0.0))[0])
        self.rng = rng
        self.prev_Xs, self.prev_V, self.prev_delta = None, None, None

    def _moments(self, Xs):
        """Mean (orig. units), latent var (normalised units), V = L^-1 k_*."""
        from scipy.linalg import solve_triangular
        K_trans = self.latent(Xs, self.gp.X_train_)   # white term is 0 off the training points
        mu = K_trans @ self.gp.alpha_
        V = solve_triangular(self.gp.L_, K_trans.T, lower=True, check_finite=False)
        var = np.clip(self.latent.diag(Xs) - np.einsum('ij,ij->j', V, V), 0.0, None)
        return np.ravel(mu) * self.y_std + self.y_mean, var, V

    def step(self, X):
        Xs = self.scale.transform(np.asarray(X, dtype=float))
        mu, var, V = self._moments(Xs)
        z = self.rng.standard_normal(len(Xs))
        if self.prev_Xs is None:
            delta = np.sqrt(var) * z
        else:
            v_prev = np.clip(self.prev_var, 1e-30, None)
            k_pc = self.amp * np.exp(-0.5 * np.sum(((self.prev_Xs - Xs) / self.ls) ** 2, axis=1))
            c = k_pc - np.einsum('ij,ij->j', self.prev_V, V)
            cond_var = np.clip(var - c ** 2 / v_prev, 0.0, None)
            delta = c / v_prev * self.prev_delta + np.sqrt(cond_var) * z
        self.prev_Xs, self.prev_V, self.prev_var, self.prev_delta = Xs, V, var, delta
        return mu + delta * self.y_std


class BNNReplicaSampler:
    """Per-replica BNN posterior draw (--propagate): replica i uses one fixed
    HMC weight sample for its whole roll, i.e. a joint posterior function draw."""

    def __init__(self, est, n_rows, rng):
        self.est = est
        n_d = len(est.samples_)
        self.draw = np.resize(rng.permutation(n_d), n_rows)

    def step(self, X):
        Xs = (np.asarray(X, dtype=float) - self.est.x_mean_) / self.est.x_scale_
        out = np.empty(len(Xs))
        for d in np.unique(self.draw):
            rows = np.flatnonzero(self.draw == d)
            out[rows] = self.est._f_std(self.est.samples_[d:d + 1], Xs[rows])[0]
        return out * self.est.y_scale_ + self.est.y_mean_


class EnsembleReplicaSampler:
    """Per-replica NNE member (--propagate NNE): replica i uses one fixed
    bootstrap member for its whole roll, so the member spread propagates."""

    def __init__(self, est, n_rows, rng):
        self.est = est
        self.member = np.resize(rng.permutation(len(est.weights_[0])), n_rows)

    def step(self, X):
        out = np.empty(len(X))
        for k in np.unique(self.member):
            rows = np.flatnonzero(self.member == k)
            out[rows] = self.est.predict_member(X[rows], k)
        return out


class Reconstruct_real_data:
    def __init__(self, data_dir='../Data', model_dir='../model', exp_id='No',
                 anchor_real=False, n_draws=0, models=None, seed=12345,
                 out_dir=None, target='level', clip_dir=None,
                 seed_noise=0.0, noise_models=('NN', 'GP', 'BNN'),
                 propagate=()):
        self.data_dir = data_dir
        # propagate: models (GP, BNN) whose own predictive uncertainty is sampled
        # per replica inside the roll.  Output defaults to {data_dir}/propagate.
        self.propagate = set(propagate)
        if out_dir is None and self.propagate:
            out_dir = os.path.join(data_dir, 'propagate')
        # seed_noise K > 0: the seed M of noise_models gets N(0, K * omega * replica
        # std at that omega) added (one draw shared by those models); other models keep
        # the clean seed.  Output defaults to {data_dir}/seednoise_{K}.
        self.seed_noise = seed_noise
        self.noise_models = set(noise_models)
        self.noise_rng = np.random.default_rng(seed + 1)
        if out_dir is None and seed_noise > 0:
            out_dir = os.path.join(data_dir, f'seednoise_{seed_noise:g}')
        self.out_dir = out_dir or data_dir
        os.makedirs(self.out_dir, exist_ok=True)
        if os.path.abspath(self.out_dir) != os.path.abspath(data_dir):
            # test_error.py --data_dir {out_dir} also needs these two
            for f in ('All_p_w_m.csv', 'train_error_summary.csv'):
                src, dst = os.path.join(data_dir, f), os.path.join(self.out_dir, f)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)
        # target='logratio': models predict log(M5/M4) (SegTest); the roll rebuilds
        # M5 = M4 * exp(prediction), the prediction clipped per (P, omega) to
        # {clip_dir}/logratio_clip_{fm}.csv when clip_dir is given
        self.target = target
        self.clip_dir = clip_dir
        if target == 'logratio' and n_draws > 0:
            raise ValueError('--n_draws is not supported with --target logratio')
        self.model_dir = model_dir
        self.exp_id = exp_id
        self.features = FEATURES[exp_id]
        self.use_exp = exp_id == 'Yes'
        self.anchor_real = anchor_real
        self.n_draws = n_draws
        self.model_names = models or MODEL_NAMES
        self.rng = np.random.default_rng(seed)

    def _load_models(self, fm, ordered_ps):
        """{model_name: {P: estimator}} for one FM.

        model.py appends one estimator per P in ``sorted(P.unique())`` order,
        so element i corresponds to ``ordered_ps[i]`` -- NOT to ``P == i + 1``.
        Models without a saved .sav for this FM are left out.
        """
        models = {}
        for name in self.model_names:
            path = os.path.join(self.model_dir, str(fm), f'{name}_exp_{self.exp_id}.sav')
            if not os.path.exists(path):
                print(f'FM={fm} {name}: {path} not found, skipped')
                continue
            est_list = joblib.load(path)
            if len(est_list) != len(ordered_ps):
                raise ValueError(
                    f'FM={fm} {name}: {len(est_list)} per-P models but '
                    f'{len(ordered_ps)} P values {ordered_ps} in the data')
            models[name] = dict(zip(ordered_ps, est_list))
        return models

    def _load_synthetic(self, fm):
        """Synthetic replicas for one FM as columns [P, Exp, W, M], omega rounded."""
        per_fm = os.path.join(self.data_dir, f'generated_Synthetic_exp_data_{fm}.csv')
        if os.path.exists(per_fm):
            df = pd.read_csv(per_fm, usecols=['P', 'Exp', 'W', 'M'])
        else:
            df = pd.read_csv(os.path.join(self.data_dir, 'generated_Synthetic_exp_data.csv'),
                             usecols=['FM', 'P', 'Exp', 'W', 'M'])
            df = df[df.FM == fm][['P', 'Exp', 'W', 'M']].copy()
        df['W'] = np.round(df['W'], W_DECIMALS)
        return df

    def _load_clip(self, fm):
        """{P: (W grid, lo, hi)} log-ratio clip table, or None."""
        if self.target != 'logratio' or not self.clip_dir:
            return None
        path = os.path.join(self.clip_dir, f'logratio_clip_{fm}.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found')
        t = pd.read_csv(path)
        return {p: (g.W.to_numpy(), g.lo.to_numpy(), g.hi.to_numpy(), g.M_hi.to_numpy())
                for p, g in t.sort_values('W').groupby('P')}

    def _to_level(self, pred, name, m4, clip, p, w):
        """Model output -> M at w5 (identity for target='level').  logratio: step clipped
        to [lo, hi] and M capped at M_hi of the clip table (SegTest common.logratio_to_level)."""

        if self.seed_noise > 0 and name in self.noise_models:
            # add N(0, K * omega * replica std at that omega) to the seed M of noise_models
            sd = np.nanstd(pred, axis=0) * w
            pred += self.noise_rng.normal(size=pred.shape) * self.seed_noise * sd
        
        if self.target != 'logratio':
            return pred
        m4 = np.maximum(m4, 1e-6)       # a negative M4 would turn exp growth into a fall to -inf
        if clip is None or p not in clip:
            return m4 * np.exp(pred)
        ws, lo, hi, m_hi = clip[p]
        i = int(np.argmin(np.abs(ws - w)))
        return np.minimum(m4 * np.exp(np.clip(pred, lo[i], hi[i])), m_hi[i])

    @staticmethod
    def _interleave(ws, ms):
        """row-wise [w1..w4] + [m1..m4] -> [w1, m1, w2, m2, w3, m3, w4, m4]."""
        out = np.empty((ws.shape[0], 2 * ws.shape[1]), dtype=float)
        out[:, 0::2] = ws
        out[:, 1::2] = ms
        return out

    def reconstruct_data(self):
        df_real_all = pd.read_csv(os.path.join(self.data_dir, 'All_p_w_m.csv'))
        df_real_all['W_raw'] = df_real_all['W']          # unrounded, for --anchor_real
        df_real_all['W'] = np.round(df_real_all['W'], W_DECIMALS)
        suffix = '_anchored' if self.anchor_real else ''

        for fm in sorted(df_real_all.FM.unique()):
            df_syn = self._load_synthetic(fm)
            df_real = df_real_all[df_real_all.FM == fm]

            ordered_ps = sorted(df_syn.P.unique())
            models = self._load_models(fm, ordered_ps)
            names = list(models)
            for name in [n for n in names if n in self.propagate]:
                models[f'{name}_draw'] = models[name]
                names.append(f'{name}_draw')
            if not names:
                print(f'FM={fm}: no trained *_exp_{self.exp_id}.sav models, nothing saved')
                continue
            all_ws = np.array(sorted(df_syn.W.unique()), dtype=float)
            clip = self._load_clip(fm)

            P_parts, E_parts, W_parts = [], [], []
            anchored_parts = []
            pred_parts = {name: [] for name in names}
            draw_parts = []                     # (P, Draw, W, M_GP) frames for --n_draws

            for p in ordered_ps:
                gen_piv = (df_syn[df_syn.P == p]
                           .pivot_table(index='Exp', columns='W', values='M', aggfunc='mean')
                           .sort_index())
                gen_piv = gen_piv.reindex(columns=sorted(gen_piv.columns))
                gen_ws = np.array(gen_piv.columns, dtype=float)

                if len(gen_ws) <= N_LAGS:
                    print(f'FM={fm} P={p}: only {len(gen_ws)} omega points (<= {N_LAGS}), skipped')
                    continue

                exps = gen_piv.index.to_numpy()
                n_exp = len(exps)
                real_p = df_real[df_real.P == p]

                if self.anchor_real and not real_p.empty:
                    # every grid omega inside the real reach comes from that Exp's data
                    head_ws = gen_ws[gen_ws <= real_p.W_raw.max() + 1e-9]
                    if len(head_ws) < N_LAGS:
                        head_ws = gen_ws[:N_LAGS]
                    head_M = _anchor_curves(real_p.assign(W=real_p.W_raw), exps, head_ws)
                    head_real = np.isfinite(head_M)
                    fallback = gen_piv.reindex(columns=head_ws).to_numpy(dtype=float)
                    head_M = np.where(head_real, head_M, fallback)
                else:
                    # seed M: the first N_LAGS grid omegas of that Exp's synthetic
                    # replica only.  Real data (raw or filled) is never used: it is
                    # not on the synthetic curve the models learned, so any real
                    # seed bends both M and the derived I(omega) out of shape.
                    head_ws = gen_ws[:N_LAGS]
                    head_M = np.array(gen_piv.iloc[:, :N_LAGS].to_numpy(dtype=float), copy=True)
                    head_real = np.zeros_like(head_M, dtype=bool)
                if not np.isfinite(head_M).all():
                    head_M = (pd.DataFrame(head_M).ffill(axis=1).bfill(axis=1)
                              .fillna(0.0).to_numpy())
                seed_ws, seed_M = head_ws[-N_LAGS:], head_M[:, -N_LAGS:]

                # per-model head: noisy copy for noise_models when --seed_noise > 0
                head_by = {name: head_M for name in names}
                if self.seed_noise > 0:
                    # std grows with omega: K * omega * (replica spread at that omega)
                    #sd = np.nanstd(head_M, axis=0) * head_ws
                    #print(f'SD={sd}')
                    noisy = head_M #+ self.noise_rng.normal(size=head_M.shape) * self.seed_noise * head_ws
                    for name in names:
                        if name in self.noise_models:
                            head_by[name] = noisy

                p_col = np.full(n_exp, p, dtype=float)
                exp_col = exps.astype(float)

                # head rows (seed window, or the whole anchored range): the
                # (possibly real) value goes into every model column
                for j, w in enumerate(head_ws):
                    P_parts.append(p_col.copy())
                    E_parts.append(exps)
                    W_parts.append(np.full(n_exp, w))
                    anchored_parts.append(head_real[:, j])
                    for name in names:
                        pred_parts[name].append(head_by[name][:, j].copy())

                # per-model rolling lag window [w1, m1, ... w4, m4]
                lag_ws = np.tile(seed_ws, (n_exp, 1))
                state = {name: self._interleave(lag_ws, head_by[name][:, -N_LAGS:].copy())
                         for name in names}

                roll_ws = all_ws[all_ws > seed_ws[-1]]
                if self.n_draws > 0 and 'GP' in names:
                    draw_parts.append(self._roll_draws(models['GP'][p], p, head_ws, head_M,
                                                       seed_ws, seed_M, roll_ws))

                # propagated models roll twice: M_{X} deterministic (mean prediction)
                # and M_{X}_draw (sampled); test_error.py takes the model error from
                # their per-replica difference
                samplers = {}
                if 'GP_draw' in names:
                    samplers['GP_draw'] = GPReplicaSampler(models['GP'][p], self.rng)
                if 'BNN_draw' in names:
                    samplers['BNN_draw'] = BNNReplicaSampler(models['BNN'][p], n_exp, self.rng)
                if 'NNE_draw' in names:
                    samplers['NNE_draw'] = EnsembleReplicaSampler(models['NNE'][p], n_exp, self.rng)

                for w in roll_ws:
                    w5 = np.full(n_exp, w, dtype=float)

                    step_val = {}
                    for name in names:
                        cols = ([p_col, exp_col, state[name], w5] if self.use_exp
                                else [p_col, state[name], w5])
                        X = pd.DataFrame(np.column_stack(cols), columns=self.features)
                        if name in samplers:
                            pred = samplers[name].step(X.to_numpy(dtype=float))
                        else:
                            pred = np.asarray(models[name][p].predict(X), dtype=float)
                        step_val[name] = self._to_level(pred, name, state[name][:, -1], clip, p, w)

                    P_parts.append(p_col.copy())
                    E_parts.append(exps)
                    W_parts.append(w5.copy())
                    anchored_parts.append(np.zeros(n_exp, dtype=bool))
                    for name in names:
                        pred_parts[name].append(step_val[name])
                        # slide: drop (w1, m1), append (w5, value used)
                        state[name] = np.column_stack(
                            [state[name][:, 2:], w5, step_val[name]])

                print(f'FM={fm} P={p} done  (seed omega<={seed_ws[-1]:.1f}, '
                      f'{len(roll_ws)} pure-autoregressive extrapolated steps)')

            if not P_parts:
                print(f'FM={fm}: no usable (P, Exp) groups, nothing saved')
                continue

            df_out = pd.DataFrame({
                'FM': fm,
                'P': np.concatenate(P_parts).astype(int),
                'Exp': np.concatenate(E_parts).astype(int),
                'W': np.round(np.concatenate(W_parts), W_DECIMALS),
                'anchored': np.concatenate(anchored_parts).astype(int),
                **{f'M_{name}': np.concatenate(pred_parts[name]) for name in names},
            }).sort_values(['P', 'Exp', 'W']).reset_index(drop=True)

            token = EXP_TOKEN[self.exp_id]
            out_path = os.path.join(self.out_dir,
                                    f'Real_data_projected_{fm}_{token}{suffix}_v2.csv')
            df_out.to_csv(out_path, index=False)
            print(f'FM={fm} saved -> {out_path}  ({len(df_out)} rows)')

            if draw_parts:
                df_d = pd.concat(draw_parts, ignore_index=True)
                df_d.insert(0, 'FM', fm)
                d_path = os.path.join(self.out_dir,
                                      f'Real_data_projected_{fm}_{token}{suffix}_draws_v2.csv')
                df_d.to_csv(d_path, index=False)
                print(f'FM={fm} GP draws saved -> {d_path}  ({len(df_d)} rows)')

    def _roll_draws(self, pipe, p, head_ws, head_M, seed_ws, seed_M, roll_ws):
        """Draw 0 (noiseless) + n_draws latent-GP rolls from the Exp-mean seed.

        Returns a frame P, Draw, W, M_GP over head_ws (Exp-mean, identical in
        every draw) and roll_ws (sampled).  Uses the no-Exp feature layout; with
        --exp_id Yes the Exp column is set to 0.
        """
        k = self.n_draws + 1
        roller = GPDrawRoller(pipe, self.rng)
        state = self._interleave(np.tile(seed_ws, (k, 1)),
                                 np.tile(seed_M.mean(axis=0), (k, 1)))
        p_col, e_col = np.full(k, p, dtype=float), np.zeros(k)
        vals = [np.tile(m, (k, 1)) for m in head_M.mean(axis=0)]
        for w in roll_ws:
            w5 = np.full(k, w, dtype=float)
            cols = [p_col, e_col, state, w5] if self.use_exp else [p_col, state, w5]
            X = pd.DataFrame(np.column_stack(cols), columns=self.features)
            pred = roller.step(X)
            vals.append(pred[:, None] if pred.ndim == 1 else pred)
            state = np.column_stack([state[:, 2:], w5, pred])
        M = np.hstack([v.reshape(k, 1) for v in vals])          # (k, n_w)
        ws = np.concatenate([head_ws, roll_ws])
        return pd.DataFrame({'P': p,
                             'Draw': np.repeat(np.arange(k), len(ws)),
                             'W': np.round(np.tile(ws, k), W_DECIMALS),
                             'M_GP': M.ravel()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='../Data/2026',
                        help="Directory with All_p_w_m.csv and generated_Synthetic_exp_data*.csv")
    parser.add_argument("--model_dir", default='../model/2026',
                        help="Directory with {model_dir}/{fm}/{GB,RF,XGB,NN,GP}_exp_{id}.sav")
    parser.add_argument("--exp_id", default='No', choices=['No', 'Yes'],
                        help="Which per-P model variant to roll with (default: No / without_exp)")
    parser.add_argument("--anchor_real", action='store_true',
                        help="Use each Exp's real M over the whole measured omega range")
    parser.add_argument("--n_draws", type=int, default=0,
                        help="GP latent-function draws rolled from the Exp-mean seed (0 = off)")
    parser.add_argument("--models", nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES,
                        help="Models to roll (default: all with a saved .sav)")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for --n_draws")
    parser.add_argument("--out_dir", default=None,
                        help="Where Real_data_projected_*.csv go (default: --data_dir)")
    parser.add_argument("--target", default='level', choices=['level', 'logratio'],
                        help="What the models predict: M5 (default) or log(M5/M4) (SegTest logratio runs)")
    parser.add_argument("--clip_dir", default=None,
                        help="logratio only: directory with logratio_clip_{fm}.csv (per-P, per-omega step limits)")
    parser.add_argument("--seed_noise", type=float, default=0.0,
                        help="Add N(0, K * omega * replica std at each seed omega) to the seed M of "
                             "--noise_models (0 = off). Output -> {data_dir}/seednoise_{K} unless --out_dir")
    parser.add_argument("--noise_models", nargs='+', choices=MODEL_NAMES, default=['NN', 'GP', 'BNN'],
                        help="Models whose seed gets --seed_noise (others keep the clean seed)")
    parser.add_argument("--propagate", nargs='*', choices=['GP', 'BNN', 'NNE'], default=[],
                        help="Sample these models' own predictive uncertainty per replica inside the "
                             "roll (GP latent draw per step, BNN one posterior draw / NNE one ensemble member per replica). "
                             "Each adds an M_{X}_draw column next to the deterministic M_{X}. "
                             "Output -> {data_dir}/propagate unless --out_dir")
    args = parser.parse_args()

    rd = Reconstruct_real_data(data_dir=args.data_dir, model_dir=args.model_dir,
                               exp_id=args.exp_id, anchor_real=args.anchor_real,
                               n_draws=args.n_draws, models=args.models, seed=args.seed,
                               out_dir=args.out_dir, target=args.target, clip_dir=args.clip_dir,
                               seed_noise=args.seed_noise, noise_models=args.noise_models,
                               propagate=args.propagate)
    rd.reconstruct_data()
