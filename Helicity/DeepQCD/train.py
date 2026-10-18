"""Train the DeepQCD window network on the synthetic curves of every FM except --test_fm.

Each epoch draws fresh random windows (FM, P, Exp, stride, start) from the training
FMs, anywhere on the omega grid.  A random --val_frac of each training FM's replicas
is held back for validation (learning-rate schedule and early stopping); the held-out
FM is not touched here (evaluate.py scores it).

--head direct  the MLP predicts the 6 log(M / last seed) values freely
--head m3      the MLP predicts the M3 parameters (log A, s) of the window's curve
--drop a,level switch inputs off (fed as 0): a = the a^2 input, level = log(last seed);
               the spec gets '-noa' / '-nolevel' (e.g. m3-noa-nolevel)

Writes
    ../model/DeepQCD/deepqcd_{spec}_test_FM{fm}.pt      (..._all.pt with --test_fm none)
    ../Data/DeepQCD/train_history_{spec}_test_FM{fm}.csv
    ../Data/DeepQCD/Images/train_loss_{spec}_test_FM{fm}.png

Run from Helicity/DeepQCD with the gluon env:
    python train.py --test_fm 15 --head m3
    python train.py --test_fm none --head m3     # final model on all FMs, for new ensembles
"""

import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import (DEFAULT_STRIDES, DROPPABLE, FMS, HEADS, IMG_DIR, N_PRED, N_SEED, OUT_DIR, PLOT_DPI,
                    WindowNet, get_device, load_curves, m3_seed_params, make_features,
                    model_path, run_tag, sample_windows, split_replicas, to_target)


def windows_xy(curves, fms, n, strides, rng, exps):
    a, P, w, m = sample_windows(curves, fms, n, strides, rng, exps)
    return make_features(a, P, w, m[:, :N_SEED]), to_target(m[:, N_SEED:], m[:, :N_SEED]), w, m


def main(args):
    test_fm = None if args.test_fm == 'none' else int(args.test_fm)
    train_fms = [fm for fm in FMS if fm != test_fm]
    strides = [int(s) for s in args.strides.split(',')]
    drop = [d for d in args.drop.split(',') if d]
    if not set(drop) <= set(DROPPABLE):
        raise ValueError(f'--drop must be from {list(DROPPABLE)}')
    tag = run_tag(test_fm, args.head, drop, args.tag)
    path = model_path(test_fm, args.head, drop, args.tag)
    device = get_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    print(f'{args.head} head, dropped inputs {drop}: train FMs {train_fms}, test FM {test_fm}, strides {strides}, '
          f'val = {args.val_frac:.0%} of replicas, device {device}')

    curves = load_curves()
    tr_exps, va_exps = split_replicas(curves, train_fms, args.val_frac, rng)
    X0, Y0, w0, m0 = windows_xy(curves, train_fms, args.n_val, strides, rng, tr_exps)
    Xv, Yv, _, _ = windows_xy(curves, train_fms, args.n_val, strides, rng, va_exps)

    model = WindowNet(X0.shape[1], N_PRED, args.hidden, args.layers, args.head, drop)
    params = np.hstack(m3_seed_params(w0[:, :N_SEED], m0[:, :N_SEED])) if args.head == 'm3' else None
    model.set_scaling(X0, Y0, params)
    model.to(device)
    Xv, Yv = torch.from_numpy(Xv).to(device), torch.from_numpy(Yv).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.patience // 3)
    loss_fn = torch.nn.MSELoss()

    best, best_state, wait, hist = np.inf, None, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr = 0.0
        for _ in range(args.steps_per_epoch):
            X, Y, _, _ = windows_xy(curves, train_fms, args.batch, strides, rng, tr_exps)
            X, Y = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
            loss = loss_fn(model(X), model.scale_target(Y))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += loss.item()
        tr /= args.steps_per_epoch

        model.eval()
        with torch.no_grad():
            va = loss_fn(model(Xv), model.scale_target(Yv)).item()
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

    cfg = {'n_in': X0.shape[1], 'hidden': args.hidden, 'layers': args.layers, 'head': args.head,
           'drop': drop, 'strides': strides, 'test_fm': test_fm, 'train_fms': train_fms,
           'val_frac': args.val_frac, 'best_val_loss': best, 'seed': args.seed}
    torch.save({'state_dict': best_state, 'config': cfg}, path)
    print(f'saved {path} (best val loss {best:.5f})')

    hist = pd.DataFrame(hist)
    hist.to_csv(os.path.join(OUT_DIR, f'train_history_{tag}.csv'), index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(hist.epoch, hist.train_loss, label='train')
    ax.semilogy(hist.epoch, hist.val_loss, label=f'val ({args.val_frac:.0%} held-out replicas)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE (standardised target)')
    ax.set_title(f'DeepQCD {tag.rsplit("_", 2)[0]}, train FM {train_fms}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, f'train_loss_{tag}.png'), dpi=PLOT_DPI)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_fm', required=True, choices=[str(f) for f in FMS] + ['none'])
    ap.add_argument('--head', default='direct', choices=HEADS)
    ap.add_argument('--drop', default='', help=f'comma list of inputs to switch off, from {list(DROPPABLE)}')
    ap.add_argument('--tag', default='', help="extra name for the checkpoint, e.g. 'wide' -> direct-noa-nolevel-wide")
    ap.add_argument('--strides', default=','.join(map(str, DEFAULT_STRIDES)),
                    help='window strides in grid steps of 0.1 omega')
    ap.add_argument('--val_frac', type=float, default=0.1,
                    help='fraction of each training FM\'s replicas held back for validation')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--steps_per_epoch', type=int, default=200)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--n_val', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--device', default='auto')
    main(ap.parse_args())
