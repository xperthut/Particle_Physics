"""NNE: bootstrap ensemble of small MLPs, all members trained at once on the GPU.

Each of the ``n_members`` networks gets its own block bootstrap of the Exp
replicas (rows of one replica are correlated, so whole replicas are
resampled) and is early-stopped on its own out-of-bag replicas.  The members
are stacked into batched weight tensors (K, n_in, h) and trained together with
``torch.bmm``, so one small ensemble costs about one training run on the GPU
(MPS / CUDA, CPU fallback).

The architecture is deliberately small (default one hidden layer of 8 units)
so the members follow the mean curve rather than the replica noise; tanh
(default) keeps the output bounded, so no member runs away under the
autoregressive roll (ReLU members extrapolate linearly and can diverge); the
member-to-member spread is then the model (bootstrap) uncertainty.
reconstruct_real_data.py --propagate NNE gives every replica one member for
its whole roll so that spread propagates along omega.

The fitted weights are stored as numpy arrays and prediction runs in numpy,
so a saved model loads and predicts without torch or a GPU.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class NNEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_members=16, width=8, depth=1, alpha=1e-3,
                 learning_rate_init=3e-3, n_rows=50000, batch_size=4096,
                 max_epochs=1500, patience=30, n_val=5000, random_state=0,
                 activation='tanh', verbose=0):
        self.n_members = n_members
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.n_rows = n_rows
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.n_val = n_val
        self.random_state = random_state
        self.activation = activation
        self.verbose = verbose

    def _hidden(self):
        return [max(int(self.width) // 2 ** k, 4) for k in range(int(self.depth))]

    def _bootstrap(self, rid, rng):
        """(K, n_rows) train and (K, n_val) out-of-bag row indices."""
        reps = np.unique(rid)
        rows_of = {r: np.flatnonzero(rid == r) for r in reps}
        n_tr = min(self.n_rows, len(rid))
        tr, va = [], []
        for _ in range(self.n_members):
            pick = rng.choice(reps, size=len(reps), replace=True)
            idx = np.concatenate([rows_of[r] for r in pick])
            tr.append(rng.choice(idx, size=n_tr, replace=len(idx) < n_tr))
            oob = np.setdiff1d(reps, pick)
            oob_idx = (np.concatenate([rows_of[r] for r in oob]) if len(oob)
                       else idx)
            va.append(rng.choice(oob_idx, size=min(self.n_val, len(oob_idx)),
                                 replace=False))
        n_v = min(len(v) for v in va)
        return np.stack(tr), np.stack([v[:n_v] for v in va])

    def fit(self, X, y, replica_id=None):
        import torch
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        rid = np.arange(len(X)) if replica_id is None else np.asarray(replica_id)
        rng = np.random.RandomState(int(self.random_state))
        torch.manual_seed(int(self.random_state))

        self.x_mean_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.x_scale_ = np.where(sd > 1e-12, sd, 1.0)      # constant P column -> 0
        self.y_mean_ = float(y.mean())
        self.y_scale_ = float(y.std()) or 1.0
        Xs = ((X - self.x_mean_) / self.x_scale_).astype(np.float32)
        ys = ((y - self.y_mean_) / self.y_scale_).astype(np.float32)

        tr_idx, va_idx = self._bootstrap(rid, rng)
        dev = _device()
        K, n_tr = tr_idx.shape
        Xtr = torch.from_numpy(Xs[tr_idx]).to(dev)            # (K, n_tr, d)
        ytr = torch.from_numpy(ys[tr_idx]).to(dev)            # (K, n_tr)
        Xva = torch.from_numpy(Xs[va_idx]).to(dev)
        yva = torch.from_numpy(ys[va_idx]).to(dev)

        sizes = [X.shape[1]] + self._hidden() + [1]
        params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            W = torch.randn(K, a, b, device=dev) * np.sqrt(2.0 / a)
            params += [W.requires_grad_(), torch.zeros(K, 1, b, device=dev).requires_grad_()]

        def forward(h):
            for i in range(0, len(params), 2):
                h = torch.baddbmm(params[i + 1], h, params[i])
                if i < len(params) - 2:
                    h = torch.tanh(h) if self.activation == 'tanh' else torch.relu(h)
            return h[..., 0]

        opt = torch.optim.Adam(params, lr=self.learning_rate_init)
        best = torch.full((K,), float('inf'), device=dev)
        best_par = [p.detach().clone() for p in params]
        since = torch.zeros(K, dtype=torch.long, device=dev)
        start = time.time()
        epoch = 0
        for epoch in range(self.max_epochs):
            perm = torch.randperm(n_tr, device=dev)
            for s in range(0, n_tr, self.batch_size):
                b = perm[s:s + self.batch_size]
                loss_k = ((forward(Xtr[:, b]) - ytr[:, b]) ** 2).mean(dim=1)
                l2 = sum((W ** 2).sum() for W in params[0::2])
                loss = loss_k.sum() + self.alpha * l2 / n_tr
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                v = ((forward(Xva) - yva) ** 2).mean(dim=1)       # (K,)
                better = v < best
                best = torch.where(better, v, best)
                since = torch.where(better, torch.zeros_like(since), since + 1)
                for p, bp in zip(params, best_par):
                    bp[better] = p.detach()[better]
            if bool((since >= self.patience).all()):
                break
        self.n_epochs_ = epoch + 1
        self.val_mse_ = (best.cpu().numpy() * self.y_scale_ ** 2).tolist()
        self.weights_ = [p.cpu().numpy().astype(float) for p in best_par]
        self.fit_seconds_ = round(time.time() - start, 1)
        self.device_ = str(dev)
        if self.verbose:
            print(f'      NNE {K} members {self._hidden()} on {dev}: {self.n_epochs_} epochs, '
                  f'{self.fit_seconds_}s', flush=True)
        self.n_features_in_ = X.shape[1]
        return self

    # ------------------------------------------------------------- predict
    def _forward_np(self, Xs, members):
        h = np.broadcast_to(Xs, (len(members),) + Xs.shape)
        for i in range(0, len(self.weights_), 2):
            W, b = self.weights_[i][members], self.weights_[i + 1][members]
            h = np.matmul(h, W) + b
            if i < len(self.weights_) - 2:
                h = (np.tanh(h) if getattr(self, 'activation', 'relu') == 'tanh'
                     else np.maximum(h, 0.0))
        return h[..., 0] * self.y_scale_ + self.y_mean_

    def _scale(self, X):
        return (np.asarray(X, dtype=float) - self.x_mean_) / self.x_scale_

    def predict_all(self, X):
        """(n_members, n_rows) every member's prediction."""
        return self._forward_np(self._scale(X), np.arange(len(self.weights_[0])))

    def predict_member(self, X, k):
        return self._forward_np(self._scale(X), np.array([k]))[0]

    def predict(self, X):
        return self.predict_all(X).mean(axis=0)
