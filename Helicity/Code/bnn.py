"""Bayesian neural network regressor sampled with Hamiltonian Monte Carlo (HMC).

Same inputs/target as the other per-P models (P [, Exp], 4 lag (w, m) pairs,
w5 -> target), so it slots into model.py / train.py / reconstruct_real_data.py
as model type ``BNN``.

Model (inputs and target standardised, like ScaledMLPRegressor):
    f_theta(x) = tanh MLP, hidden layers (width, width/2, ...) like the NN model
    y ~ Normal(f_theta(x), sigma^2)
    W_l ~ Normal(0, prior_scale^2 / fan_in),  b_l ~ Normal(0, 1)
    log sigma ~ Normal(log 0.1, 1)            (sigma in standardised-y units)
tanh, not relu: HMC needs a smooth log posterior (relu kinks cost accuracy in
the leapfrog integrator).

Sampling (pure torch, float64, full batch):
  * each chain starts from its own MAP fit (Adam), from a different random
    initialisation, so separate chains can land in separate posterior modes;
  * warmup: step size tuned by dual averaging (Hoffman & Gelman 2014) to a
    target acceptance of ``target_accept``; halfway through, the diagonal
    mass matrix is set to the posterior variance seen so far (Stan-style
    shrinkage) and the step size is re-tuned;
  * sampling: fixed number of leapfrog steps, step size jittered +-20 %,
    thinned so ``n_keep`` posterior draws are kept over all chains.

Data: exact HMC needs the full-batch gradient at every leapfrog step, so like
the GP it is fit on a seeded subsample of ``max_samples`` rows per P.  The
rows of these frames are overlapping sliding windows over ~1000 replicas of
one curve, i.e. far from independent, so the posterior width (the model
uncertainty) depends on ``max_samples`` and is not a calibrated error on its
own -- compare it across models / settings, do not quote it as a sigma.

predict(X) returns the posterior predictive mean (average of f over the kept
draws), which keeps the autoregressive roll deterministic like the other
models.  predict(X, return_std=True) adds the spread of f over the draws
(model uncertainty, observation noise excluded); predict_samples(X) gives
every draw.  The kept draws are stored as numpy arrays and prediction runs in
numpy, so the pickled model does not need torch to be used.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

PREDICT_CHUNK = 5000        # rows per numpy forward pass in predict()


def _hidden_sizes(width, depth):
    return tuple(max(int(width) // 2 ** k, 4) for k in range(int(depth)))


def _layer_shapes(n_in, hidden):
    sizes = [n_in, *hidden, 1]
    return [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def _rhat(draws):
    """Gelman-Rubin R-hat per column; draws (n_chains, n_draws, k).

    A single chain is split in halves.
    """
    if draws.shape[0] == 1:
        h = draws.shape[1] // 2
        draws = np.stack([draws[0, :h], draws[0, h:2 * h]])
    m, n = draws.shape[:2]
    if n < 2:
        return np.full(draws.shape[2], np.nan)
    chain_mean = draws.mean(axis=1)
    W = draws.var(axis=1, ddof=1).mean(axis=0)
    B = n * chain_mean.var(axis=0, ddof=1)
    var_plus = (n - 1) / n * W + B / n
    return np.sqrt(var_plus / np.where(W > 0, W, np.nan))


class BayesianMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, width=32, depth=2, prior_scale=1.0, max_samples=3000,
                 n_chains=2, n_warmup=600, n_samples=600, n_leapfrog=30, n_keep=200,
                 target_accept=0.8, map_steps=2000, random_state=0, verbose=0):
        self.width = width
        self.depth = depth
        self.prior_scale = prior_scale
        self.max_samples = max_samples
        self.n_chains = n_chains
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.n_leapfrog = n_leapfrog
        self.n_keep = n_keep
        self.target_accept = target_accept
        self.map_steps = map_steps
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ torch
    def _setup(self, n_in):
        self.hidden_ = _hidden_sizes(self.width, self.depth)
        self.shapes_ = _layer_shapes(n_in, self.hidden_)
        self.n_params_ = sum(a * b + b for a, b in self.shapes_) + 1     # +1: log sigma

    def _unpack(self, theta):
        """theta (n_params,) -> [(W, b), ...], log_sigma."""
        out, k = [], 0
        for a, b in self.shapes_:
            W = theta[k:k + a * b].reshape(a, b)
            k += a * b
            out.append((W, theta[k:k + b]))
            k += b
        return out, theta[k]

    def _forward(self, theta, X):
        layers, _ = self._unpack(theta)
        h = X
        for i, (W, b) in enumerate(layers):
            h = h @ W + b
            if i < len(layers) - 1:
                h = h.tanh()
        return h[:, 0]

    def _potential(self, theta, X, y):
        """-log posterior (up to a constant)."""
        layers, log_sigma = self._unpack(theta)
        r = y - self._forward(theta, X)
        nll = len(y) * log_sigma + 0.5 * (r * r).sum() * (-2 * log_sigma).exp()
        prior = 0.5 * (log_sigma - np.log(0.1)) ** 2
        for W, b in layers:
            var_w = self.prior_scale ** 2 / W.shape[0]
            prior = prior + 0.5 * (W * W).sum() / var_w + 0.5 * (b * b).sum()
        return nll + prior

    def _init_theta(self, gen):
        import torch
        parts = []
        for a, b in self.shapes_:
            parts.append(torch.randn(a * b, generator=gen, dtype=torch.float64) / np.sqrt(a))
            parts.append(torch.zeros(b, dtype=torch.float64))
        parts.append(torch.tensor([np.log(0.1)], dtype=torch.float64))
        return torch.cat(parts)

    def _map(self, X, y, gen):
        import torch
        theta = self._init_theta(gen).requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.map_steps)
        for _ in range(self.map_steps):
            opt.zero_grad()
            loss = self._potential(theta, X, y) / len(y)
            loss.backward()
            opt.step()
            sched.step()
        return theta.detach()

    def _run_chain(self, X, y, chain):
        """One HMC chain -> (kept draws (k, n_params), stats dict)."""
        import torch
        gen = torch.Generator().manual_seed(int(self.random_state) * 1000 + chain)
        rng = np.random.default_rng(int(self.random_state) * 1000 + chain)

        def U_grad(th):
            th = th.detach().requires_grad_(True)
            u = self._potential(th, X, y)
            g, = torch.autograd.grad(u, th)
            return u.detach(), g

        theta = self._map(X, y, gen)
        u, g = U_grad(theta)
        minv = torch.ones(self.n_params_, dtype=torch.float64)            # inverse diagonal mass

        def leapfrog(th, p, g, eps, n_steps):
            p = p - 0.5 * eps * g
            for i in range(n_steps):
                th = th + eps * minv * p
                u_new, g = U_grad(th)
                if not torch.isfinite(u_new):
                    return th, p, g, u_new
                if i < n_steps - 1:
                    p = p - eps * g
            p = p - 0.5 * eps * g
            return th, p, g, u_new

        def transition(th, u, g, eps, n_steps):
            p0 = torch.randn(self.n_params_, generator=gen, dtype=torch.float64) / minv.sqrt()
            h0 = u + 0.5 * (p0 * p0 * minv).sum()
            th1, p1, g1, u1 = leapfrog(th, p0, g, eps, n_steps)
            h1 = u1 + 0.5 * (p1 * p1 * minv).sum()
            dh = float(h1 - h0)
            if not np.isfinite(dh):
                return th, u, g, 0.0, True
            acc = min(1.0, np.exp(-dh))
            if rng.uniform() < acc:
                return th1, u1, g1, acc, dh > 1000
            return th, u, g, acc, dh > 1000

        # initial step size: double / halve until acceptance crosses 0.5 (one leapfrog step)
        eps = 1e-3
        _, _, _, a, _ = transition(theta, u, g, eps, 1)
        direction = 1 if a > 0.5 else -1
        for _ in range(30):
            eps_try = eps * 2.0 ** direction
            _, _, _, a, _ = transition(theta, u, g, eps_try, 1)
            if (direction == 1 and a < 0.5) or (direction == -1 and a > 0.5):
                break
            eps = eps_try

        def dual_avg_init(eps):
            return {'mu': np.log(10 * eps), 'hbar': 0.0, 'log_eps_bar': 0.0, 't': 0}

        def dual_avg_step(st, acc, gamma=0.05, t0=10, kappa=0.75):
            st['t'] += 1
            t = st['t']
            st['hbar'] = (1 - 1 / (t + t0)) * st['hbar'] + (self.target_accept - acc) / (t + t0)
            log_eps = st['mu'] - np.sqrt(t) / gamma * st['hbar']
            w = t ** -kappa
            st['log_eps_bar'] = w * log_eps + (1 - w) * st['log_eps_bar']
            return float(np.exp(log_eps))

        st = dual_avg_init(eps)
        half = self.n_warmup // 2
        warm_draws = []
        n_div = 0
        for it in range(self.n_warmup):
            theta, u, g, acc, div = transition(theta, u, g, eps, self.n_leapfrog)
            n_div += div
            eps = dual_avg_step(st, acc)
            if it >= self.n_warmup // 4:
                warm_draws.append(theta.clone())
            if it == half - 1 and len(warm_draws) > 10:
                d = torch.stack(warm_draws)
                n = len(d)
                var = d.var(dim=0)
                minv = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
                u, g = U_grad(theta)
                st = dual_avg_init(eps)
        eps = float(np.exp(st['log_eps_bar']))

        keep_per_chain = max(1, self.n_keep // self.n_chains)
        thin = max(1, self.n_samples // keep_per_chain)
        kept, accs, n_div_s = [], [], 0
        for it in range(self.n_samples):
            e = eps * rng.uniform(0.8, 1.2)
            theta, u, g, acc, div = transition(theta, u, g, e, self.n_leapfrog)
            accs.append(acc)
            n_div_s += div
            if (it + 1) % thin == 0 and len(kept) < keep_per_chain:
                kept.append(theta.clone())
        stats = {'step_size': eps, 'accept_rate': float(np.mean(accs)),
                 'n_divergent_warmup': int(n_div), 'n_divergent': int(n_div_s)}
        return torch.stack(kept).numpy(), stats

    # ---------------------------------------------------------------- sklearn
    def fit(self, X, y):
        import torch
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if len(X) > self.max_samples:
            idx = np.sort(np.random.RandomState(int(self.random_state))
                          .choice(len(X), size=self.max_samples, replace=False))
            X, y = X[idx], y[idx]

        self.x_mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.x_scale_ = np.where(sd > 1e-12, sd, 1.0)      # constant columns (P) -> 0
        self.y_mean_ = float(y.mean())
        self.y_scale_ = float(y.std()) or 1.0
        Xs = torch.from_numpy((X - self.x_mean_) / self.x_scale_)
        ys = torch.from_numpy((y - self.y_mean_) / self.y_scale_)
        self._setup(X.shape[1])

        start = time.time()
        draws, stats = [], []
        for c in range(self.n_chains):
            d, s = self._run_chain(Xs, ys, c)
            draws.append(d)
            stats.append(s)
            if self.verbose:
                print(f'      chain {c}: step={s["step_size"]:.2e} accept={s["accept_rate"]:.2f} '
                      f'divergent={s["n_divergent"]}  ({time.time() - start:.0f}s)', flush=True)
        self.samples_ = np.concatenate(draws)            # (n_kept, n_params)

        # convergence check on predictions at 200 fit rows: R-hat across chains
        n_k = min(len(d) for d in draws)
        probe = Xs.numpy()[np.linspace(0, len(Xs) - 1, min(200, len(Xs))).astype(int)]
        per_chain = np.stack([self._f_std(d[:n_k], probe) for d in draws])  # (c, k, n)
        rhat = _rhat(per_chain)
        sigma = np.exp(self.samples_[:, -1]) * self.y_scale_
        self.diagnostics_ = {
            'hidden': list(self.hidden_), 'n_params': int(self.n_params_ - 1),
            'n_fit_rows': int(len(X)), 'n_kept_draws': int(len(self.samples_)),
            'chains': stats,
            'rhat_pred_median': float(np.nanmedian(rhat)),
            'rhat_pred_max': float(np.nanmax(rhat)),
            'noise_sigma_mean': float(sigma.mean()),
            'fit_seconds': round(time.time() - start, 1),
        }
        self.n_features_in_ = X.shape[1]
        return self

    def _f_std(self, samples, Xs):
        """(n_draws, n_rows) network output in standardised-y units (numpy)."""
        out = np.empty((len(samples), len(Xs)))
        for s, theta in enumerate(samples):
            k, h = 0, Xs
            for i, (a, b) in enumerate(self.shapes_):
                W = theta[k:k + a * b].reshape(a, b)
                k += a * b
                h = h @ W + theta[k:k + b]
                k += b
                if i < len(self.shapes_) - 1:
                    h = np.tanh(h)
            out[s] = h[:, 0]
        return out

    def predict_samples(self, X):
        """(n_draws, n_rows) f(X) for every kept posterior draw, original M units."""
        Xs = (np.asarray(X, dtype=float) - self.x_mean_) / self.x_scale_
        parts = [self._f_std(self.samples_, Xs[i:i + PREDICT_CHUNK])
                 for i in range(0, len(Xs), PREDICT_CHUNK)]
        return np.concatenate(parts, axis=1) * self.y_scale_ + self.y_mean_

    def predict(self, X, return_std=False):
        f = self.predict_samples(X)
        if return_std:
            return f.mean(axis=0), f.std(axis=0)
        return f.mean(axis=0)
