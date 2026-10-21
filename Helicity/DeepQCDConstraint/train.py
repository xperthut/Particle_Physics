"""Train the constrained joint-P network on the synthetic curves of every FM except --test_fm.

Each epoch draws fresh random joint windows (FM, Exp, layout, stride, start) from the
training FMs.  A random --val_frac of each training FM's replicas is held back for
validation (learning-rate schedule and early stopping); the held-out FM is not touched
here (evaluate.py scores it).

Loss: MSE of (M_pred - M_replica) / (that P's largest seed), over every P and target,
so P = 5 counts as much as P = 1.

Writes
    ../model/DeepQCDConstraint/dqc_test_FM{fm}.pt      (dqc_all.pt with --test_fm none)
    ../Data/DeepQCDConstraint/train_history_{tag}.csv, train_log_{tag}.txt
    ../Data/DeepQCDConstraint/images/train_loss_{tag}.png

Run from Helicity/DeepQCDConstraint with the gluon env:
    python train.py --test_fm 9
    python train.py --test_fm none       # final model on all FMs, for real data
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from common import (DEFAULT_STRIDES, FMS, OUT_DIR, ConstrainedNet, get_device,
                    load_curves, loss_weights, make_features, model_path, run_tag, sample_windows,
                    scaling_targets, split_replicas)
from plot import plot_train_loss


def windows_tensors(curves, fms, n, strides, rng, exps, lattice_frac, device):
    w_seed, m_seed, w_tgt, m_tgt, _ = sample_windows(curves, fms, n, strides, rng, exps, lattice_frac)
    X = make_features(w_seed, m_seed, w_tgt)
    wt = loss_weights(m_seed)
    Y = (m_tgt / wt).astype(np.float32)                   # target in units of each P's own scale
    s_over_w = (np.max(m_seed, axis=(1, 2), keepdims=True) / wt).astype(np.float32)
    return [torch.from_numpy(a).to(device) for a in (X, Y, s_over_w)]


def loss_fn(model, X, Y, s_over_w):
    return torch.mean((model(X) * s_over_w - Y) ** 2)      # model gives M / s


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w')

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def main(args):
    test_fm = None if args.test_fm == 'none' else int(args.test_fm)
    train_fms = [fm for fm in FMS if fm != test_fm]
    strides = [int(s) for s in args.strides.split(',')]
    tag = run_tag(test_fm, args.tag)
    sys.stdout = Tee(os.path.join(OUT_DIR, f'train_log_{tag}.txt'))
    device = get_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(f'constrained joint-P net: train FMs {train_fms}, test FM {test_fm}, strides {strides}, '
          f'lattice layout {args.lattice_frac:.0%}, val = {args.val_frac:.0%} of replicas, device {device}')

    curves = load_curves()
    tr_exps, va_exps = split_replicas(curves, train_fms, args.val_frac, rng)
    w_seed, m_seed, w_tgt, _, est_tgt = sample_windows(curves, train_fms, args.n_val, strides, rng,
                                                       tr_exps, args.lattice_frac)
    X0 = make_features(w_seed, m_seed, w_tgt)
    model = ConstrainedNet(X0.shape[1], args.hidden, args.layers)
    model.set_scaling(X0, scaling_targets(m_seed, est_tgt))
    model.to(device)
    val = windows_tensors(curves, train_fms, args.n_val, strides, rng, va_exps, args.lattice_frac, device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.patience // 3)

    best, best_state, wait, hist = np.inf, None, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr = 0.0
        for _ in range(args.steps_per_epoch):
            batch = windows_tensors(curves, train_fms, args.batch, strides, rng, tr_exps,
                                    args.lattice_frac, device)
            loss = loss_fn(model, *batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += loss.item()
        tr /= args.steps_per_epoch

        model.eval()
        with torch.no_grad():
            va = loss_fn(model, *val).item()
        sched.step(va)
        lr = opt.param_groups[0]['lr']
        hist.append({'epoch': epoch, 'train_loss': tr, 'val_loss': va, 'lr': lr})
        print(f'epoch {epoch:4d}  train {tr:.5f}  val {va:.5f}  lr {lr:.1e}  {time.time() - t0:.1f}s')

        if va < best:
            best, wait = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print(f'early stop: no val improvement in {args.patience} epochs')
                break

    cfg = {'n_in': X0.shape[1], 'hidden': args.hidden, 'layers': args.layers, 'strides': strides,
           'lattice_frac': args.lattice_frac, 'test_fm': test_fm, 'train_fms': train_fms,
           'val_frac': args.val_frac, 'best_val_loss': best, 'seed': args.seed}
    path = model_path(test_fm, args.tag)
    torch.save({'state_dict': best_state, 'config': cfg}, path)
    print(f'saved {path} (best val loss {best:.5f})')

    hist = pd.DataFrame(hist)
    hist.to_csv(os.path.join(OUT_DIR, f'train_history_{tag}.csv'), index=False)
    plot_train_loss(tag)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_fm', required=True, choices=[str(f) for f in FMS] + ['none'])
    ap.add_argument('--tag', default='', help="extra checkpoint name, e.g. 'wide' -> dqc-wide_all.pt")
    ap.add_argument('--strides', default=','.join(map(str, DEFAULT_STRIDES)),
                    help='window strides in grid steps of 0.1 omega')
    ap.add_argument('--lattice_frac', type=float, default=0.5,
                    help='fraction of windows with per-P lattice seeds (the rest: common grid seeds)')
    ap.add_argument('--val_frac', type=float, default=0.1,
                    help="fraction of each training FM's replicas held back for validation")
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--steps_per_epoch', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--n_val', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--device', default='auto')
    main(ap.parse_args())
