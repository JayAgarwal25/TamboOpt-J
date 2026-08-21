"""Readout probes: how much of a target is linearly present in a representation.

A probe fits a readout from some feature block to a target and reports held-out
R^2. One probe alone means little; the reading comes from the CONTRAST between
probes, so they share a split and a scoring rule:

  * a LINEAR probe measures what is linearly decodable,
  * an MLP probe on the same features upper-bounds what is decodable at all,
  * the same pair on a trivial feature block (raw coordinates, say) gives the
    baseline any richer representation has to beat.

If the trivial block already scores near 1, the comparison carries no
information about the richer one and the experiment is inconclusive rather
than negative.

Both probes standardise features and target on the TRAIN split only, and return
(held-out R^2, predictions for every row). The predictions are for plotting; the
reported number is always the test score.
"""

import torch
import torch.nn as nn


def r2(pred, true):
    """Coefficient of determination against the variance of `true`."""
    ss_res = torch.sum((true - pred) ** 2)
    ss_tot = torch.sum((true - true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def train_test_split(n, device, test_frac=0.2, seed=0):
    """Index split. Seeded, so probes in one run are comparable across features."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sh = torch.randperm(n, generator=g).to(device)
    cut = int((1.0 - test_frac) * n)
    return sh[:cut], sh[cut:]


def subsample(n, k, device, seed=0):
    """Up to `k` row indices out of `n`, for plots that cannot take every point."""
    if n <= k:
        return torch.arange(n, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(n, generator=g)[:k].to(device)


def linear_probe(feat_tr, y_tr, feat_te, y_te, feat_all, ridge=1e-3):
    """Closed-form ridge regression, fitted on train and scored on test."""
    mu, sd = feat_tr.mean(0, keepdim=True), feat_tr.std(0, keepdim=True).clamp_min(1e-6)
    a = torch.cat([(feat_tr - mu) / sd, torch.ones_like(feat_tr[:, :1])], dim=1).double()
    b = torch.cat([(feat_te - mu) / sd, torch.ones_like(feat_te[:, :1])], dim=1).double()
    ym, ys = y_tr.mean(), y_tr.std().clamp_min(1e-6)
    gram = (a.T @ a
            + ridge * a.shape[0] * torch.eye(a.shape[1], dtype=torch.float64, device=a.device))
    w = torch.linalg.solve(gram, a.T @ ((y_tr - ym) / ys).double())
    score = r2((b @ w).float() * ys + ym, y_te)
    with torch.no_grad():
        allf = torch.cat([(feat_all - mu) / sd,
                          torch.ones_like(feat_all[:, :1])], dim=1).double()
        pred_all = (allf @ w).float() * ys + ym
    return score, pred_all


def mlp_probe(feat_tr, y_tr, feat_te, y_te, feat_all, hidden=256, epochs=60,
              lr=1e-3, seed=0):
    """Small MLP readout, as the flexible-decoder counterpart to `linear_probe`."""
    torch.manual_seed(seed)
    dev = feat_tr.device
    mu, sd = feat_tr.mean(0, keepdim=True), feat_tr.std(0, keepdim=True).clamp_min(1e-6)
    ym, ys = y_tr.mean(), y_tr.std().clamp_min(1e-6)
    net = nn.Sequential(nn.Linear(feat_tr.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xtr, ytr = (feat_tr - mu) / sd, ((y_tr - ym) / ys).unsqueeze(1)
    n, bs = xtr.shape[0], 8192
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for lo in range(0, n, bs):
            idx = perm[lo:lo + bs]
            opt.zero_grad()
            nn.functional.mse_loss(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        score = r2(net((feat_te - mu) / sd).squeeze(1) * ys + ym, y_te)
        pred_all = torch.cat([net(((feat_all[lo:lo + 65536] - mu) / sd)).squeeze(1)
                              for lo in range(0, feat_all.shape[0], 65536)]) * ys + ym
    return score, pred_all


def probe_pair(features, target, tr, te, label, ridge=1e-3, **mlp_kw):
    """Run both probes on one feature block. Returns {name: (score, preds)}."""
    lin = linear_probe(features[tr], target[tr], features[te], target[te],
                       features, ridge=ridge)
    mlp = mlp_probe(features[tr], target[tr], features[te], target[te],
                    features, **mlp_kw)
    return {f"linear_from_{label}": lin, f"mlp_from_{label}": mlp}
