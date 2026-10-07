"""CNN surrogate: map -> (runner win rate, timeout rate, mean round time).

Simulated labels are noisy binomial proportions (16-48 games per map), so the
loss is BCE against the soft proportion weighted by the number of games — an
unbiased fit to the underlying rate. Maps are augmented by the horizontal and
vertical flips, which the canvas's exit layout makes game-equivalent. An
ensemble of K nets gives a mean and a spread.

    python -m mapgen.surrogate --data mapgen/data/random.jsonl --out mapgen/runs/surrogate.pt
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from .design import N_CODES, MapState, encode_map

TIME_SCALE = 90.0


class Net(nn.Module):
    def __init__(self, ch=N_CODES + 1, width=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(2 * width + ch, 128), nn.ReLU(),
                                  nn.Linear(128, 3))

    def forward(self, x):
        h = self.body(x)
        feats = torch.cat([h.mean((2, 3)), h.amax((2, 3)), x.mean((2, 3))], 1)
        return self.head(feats)


def load_records(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            recs += [json.loads(line) for line in f if line.strip()]
    return recs


def tensors(recs, augment=False):
    xs, ys, ns = [], [], []
    for r in recs:
        x = encode_map(MapState.from_rows(r["rows"]), r["mode"])
        y = [r["win"], r["timeout"], min(1.0, r["time"] / TIME_SCALE)]
        views = [x]
        if augment:
            views += [x[:, :, ::-1], x[:, ::-1, :], x[:, ::-1, ::-1]]
        for v in views:
            xs.append(np.ascontiguousarray(v))
            ys.append(y)
            ns.append(r["n"])
    return (torch.tensor(np.stack(xs)), torch.tensor(ys, dtype=torch.float32),
            torch.tensor(ns, dtype=torch.float32))


def fit(recs, epochs=40, seed=0, lr=2e-3, batch=128, verbose=False):
    torch.manual_seed(seed)
    X, Y, N = tensors(recs, augment=True)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            out = net(X[idx])
            loss = (lossf(out, Y[idx]).mean(1) * N[idx]).sum() / N[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1}: loss {tot / len(X):.4f}", flush=True)
    net.eval()
    return net


class Surrogate:
    def __init__(self, nets):
        self.nets = nets

    @torch.no_grad()
    def predict_full(self, states, modes):
        x = torch.tensor(np.stack([encode_map(s, m) for s, m in zip(states, modes)]))
        outs = torch.stack([torch.sigmoid(n(x)) for n in self.nets])
        return outs.mean(0).numpy(), outs.std(0).numpy()

    def predict(self, states, modes):
        return self.predict_full(states, modes)[0][:, :2]

    def save(self, path):
        # mapgen/runs/ is gitignored, so it does not exist on a fresh clone
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save([n.state_dict() for n in self.nets], path)

    @classmethod
    def load(cls, path):
        nets = []
        for sd in torch.load(path):
            n = Net()
            n.load_state_dict(sd)
            n.eval()
            nets.append(n)
        return cls(nets)


def report(sur, recs, label):
    states = [MapState.from_rows(r["rows"]) for r in recs]
    pred, _ = sur.predict_full(states, [r["mode"] for r in recs])
    out = {}
    for mode in ("1v1", "2v4"):
        idx = [i for i, r in enumerate(recs) if r["mode"] == mode]
        if not idx:
            continue
        for k, col in (("win", 0), ("timeout", 1)):
            y = np.array([recs[i][k] for i in idx])
            p = pred[idx, col]
            n = np.array([recs[i]["n"] for i in idx])
            noise = np.mean(y * (1 - y) / np.maximum(n - 1, 1))  # label noise variance
            mse = np.mean((y - p) ** 2)
            r2_ceiling = 1 - noise / max(np.var(y), 1e-9)
            out[f"{mode}/{k}"] = dict(mae=float(np.mean(np.abs(y - p))),
                                      r=float(np.corrcoef(y, p)[0, 1]),
                                      r2=float(1 - mse / max(np.var(y), 1e-9)),
                                      r2_ceiling=float(r2_ceiling), n=len(idx))
    print(label)
    for k, v in out.items():
        print(f"  {k:12s} MAE {v['mae']:.3f}  r {v['r']:.3f}  R2 {v['r2']:.3f} "
              f"(noise ceiling {v['r2_ceiling']:.3f}, n={v['n']})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--refit", action="store_true",
                   help="refit on all maps after the holdout report (doubles the cost)")
    args = p.parse_args()
    torch.set_num_threads(8)
    recs = load_records(args.data)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(recs))
    n_val = int(len(recs) * args.holdout)
    val = [recs[i] for i in idx[:n_val]]
    train = [recs[i] for i in idx[n_val:]]
    print(f"{len(train)} train / {len(val)} holdout maps")
    nets = [fit(train, epochs=args.epochs, seed=s, verbose=(s == 0)) for s in range(args.k)]
    sur = Surrogate(nets)
    metrics = report(sur, val, "holdout:")
    if args.refit:
        sur = Surrogate([fit(recs, epochs=args.epochs, seed=s) for s in range(args.k)])
    sur.save(args.out)
    with open(args.out + ".json", "w") as f:
        json.dump({"holdout": metrics, "n_maps": len(recs)}, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
