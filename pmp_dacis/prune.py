"""Channel statistics, importance criteria, and budget allocation for block-internal pruning."""
import numpy as np
import torch
import torch.nn.functional as F

from .model import params_for_mids, count_params

FULL_MIDS = [64, 64, 128, 128, 256, 256, 512, 512]
CRITERIA = ("random", "l1", "bn_gamma", "taylor", "G", "V", "D", "GV", "GD", "VD", "dacis")


def rank_norm(v):
    """Within-layer percentile rank in [0, 1]."""
    n = v.numel()
    if n == 1:
        return torch.zeros(1)
    order = torch.argsort(v)
    r = torch.empty(n)
    r[order] = torch.arange(n, dtype=torch.float32)
    return r / (n - 1)


def minmax_norm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-12)


def calib_indices(ds, classes, per_class, rng):
    """Class-balanced calibration subset (indices into ds) and local labels 0..len(classes)-1."""
    idx, lab = [], []
    by = ds.idx_by_class()
    for li, c in enumerate(classes):
        pool = by[c]
        take = min(per_class, len(pool))
        sel = pool[torch.from_numpy(rng.choice(len(pool), take, replace=False))]
        idx.append(sel)
        lab.append(torch.full((take,), li, dtype=torch.long))
    return torch.cat(idx), torch.cat(lab)


def collect_stats(model, ds, idx, labels, device, bs=64, tau=10.0, blocks=None):
    """One calibration pass over (idx, labels).

    Per block, per mid channel:
      G  mean L2 norm of d(loss)/d(conv1 filter), loss = CE of a cosine nearest-prototype classifier
      T  mean |sum_xy a * dL/da| per image (first-order Taylor term)
      V  variance over images of the globally pooled post-ReLU activation
      D  Fisher ratio S_B/S_W of the pooled activation over the class labels
      L1 L1 norm of the conv1 filter;  GAMMA |bn1 weight|
    The prototypes come from the current model, so no trained head is needed.
    """
    model.eval()
    blocks = model.blocks if blocks is None else blocks
    for b in blocks:
        b.capture = True
    n = len(idx)
    K = int(labels.max()) + 1
    pooled = [[] for _ in blocks]
    protos = None
    feats_sum = None
    with torch.no_grad():                                   # pass A: prototypes + pooled activations
        for s in range(0, n, bs):
            x = ds.fetch(idx[s:s + bs], device, train=False)
            f = F.normalize(model.features(x).float(), dim=1)
            if feats_sum is None:
                feats_sum = torch.zeros(K, f.shape[1], device=device)
            feats_sum.index_add_(0, labels[s:s + bs].to(device), f)
            for bi, b in enumerate(blocks):
                pooled[bi].append(b.mid.float().mean((2, 3)).cpu())
        protos = F.normalize(feats_sum, dim=1)
    G = [torch.zeros(b.conv1.out_channels) for b in blocks]
    T = [torch.zeros(b.conv1.out_channels) for b in blocks]
    nb = 0
    perm = torch.randperm(n)
    for s in range(0, n, bs):                                # pass B: gradients
        sel = perm[s:s + bs]
        x = ds.fetch(idx[sel], device, train=False)
        y = labels[sel].to(device)
        model.zero_grad(set_to_none=True)
        f = F.normalize(model.features(x).float(), dim=1)
        loss = F.cross_entropy(tau * f @ protos.t(), y)
        loss.backward()
        for bi, b in enumerate(blocks):
            G[bi] += b.conv1.weight.grad.flatten(1).norm(dim=1).cpu()
            T[bi] += (b.mid.float() * b.mid.grad.float()).sum((2, 3)).abs().sum(0).cpu() * len(sel)
        nb += 1
    model.zero_grad(set_to_none=True)
    out = {"G": [g / nb for g in G], "T": [t / n for t in T], "V": [], "D": [], "L1": [], "GAMMA": []}
    lab = labels.cpu()
    for bi, b in enumerate(blocks):
        A = torch.cat(pooled[bi])                           # [N, C]
        out["V"].append(A.var(0, unbiased=True))
        mu = A.mean(0)
        sb = torch.zeros_like(mu)
        sw = torch.zeros_like(mu)
        for k in range(K):
            m = lab == k
            if m.sum() == 0:
                continue
            Ak = A[m]
            muk = Ak.mean(0)
            sb += m.sum() * (muk - mu) ** 2
            sw += ((Ak - muk) ** 2).sum(0)
        out["D"].append(sb / (sw + 1e-8))
        out["L1"].append(b.conv1.weight.detach().abs().flatten(1).sum(1).cpu())
        out["GAMMA"].append(b.bn1.weight.detach().abs().cpu())
    for b in blocks:
        b.capture = False
        b.mid = None
    return out


def criterion_scores(name, stats, cfg, rng, gmeta=None, norm="rank", refine_gamma=0.0):
    """Per-block score tensors (higher = keep) for criterion `name`.
    gmeta: optional per-block mean gradient norms accumulated during episodic training (stage 2).
    """
    nb = len(stats["G"])
    nf = rank_norm if norm == "rank" else minmax_norm
    out = []
    for b in range(nb):
        C = stats["G"][b].numel()
        if name == "random":
            s = torch.from_numpy(rng.random(C)).float()
        elif name == "l1":
            s = stats["L1"][b]
        elif name == "bn_gamma":
            s = stats["GAMMA"][b]
        elif name == "taylor":
            s = stats["T"][b]
        elif name in ("G", "V", "D"):
            s = stats[name][b]
        elif name in ("GV", "GD", "VD"):
            s = 0.5 * (nf(stats[name[0]][b]) + nf(stats[name[1]][b]))
        elif name == "dacis":
            lg, lv, ld = cfg["lambda_g"], cfg["lambda_v"], cfg["lambda_d"]
            s = lg * nf(stats["G"][b]) + lv * nf(stats["V"][b]) + ld * nf(stats["D"][b])
            if refine_gamma > 0 and gmeta is not None:
                s = s * (1.0 + refine_gamma * nf(gmeta[b]))
        else:
            raise ValueError(name)
        out.append(s)
    return out


def allocate_mids(model, target_frac, dense_params, tilt=0.5, caps=None, min_keep=4):
    """Per-block mid-channel counts so that params / dense_params ~= target_frac.

    The keep fraction of block b is kappa * (1 - tilt * d_b), d_b = b / (B-1): later blocks lose more
    channels (tilt=0 gives the same fraction everywhere). kappa is found by bisection. Counts never exceed
    `caps` (the current widths), so a later stage can only remove channels.
    """
    B = len(model.blocks)
    caps = list(caps) if caps is not None else [b.conv1.out_channels for b in model.blocks]
    d = np.arange(B) / max(1, B - 1)

    def mids_for(k):
        return [int(min(caps[i], max(min_keep, round(k * (1 - tilt * d[i]) * FULL_MIDS[i])))) for i in range(B)]

    lo, hi = 0.0, 2.0 / max(1e-6, 1 - tilt)
    for _ in range(60):
        k = 0.5 * (lo + hi)
        if params_for_mids(model, mids_for(k)) / dense_params > target_frac:
            hi = k
        else:
            lo = k
    mids = mids_for(hi if params_for_mids(model, mids_for(hi)) / dense_params <= target_frac * 1.003 else lo)
    return mids


def allocate_linear(coefs, cur, full, params_now, dense_params, target_frac, tilt=0.5, min_keep=4):
    """Same rule as allocate_mids for networks whose parameter count is linear in each block width (MobileNetV2):
    block b keeps a fraction kappa * (1 - tilt * b / (B - 1)) of its full hidden width, kappa found by bisection."""
    B = len(coefs)
    d = np.arange(B) / max(1, B - 1)

    def mids_for(k):
        return [int(min(cur[i], max(min_keep, round(k * (1 - tilt * d[i]) * full[i])))) for i in range(B)]

    def frac(m):
        return (params_now + sum(c * (mi - ci) for c, mi, ci in zip(coefs, m, cur))) / dense_params

    lo, hi = 0.0, 2.0 / max(1e-6, 1 - tilt)
    for _ in range(60):
        k = 0.5 * (lo + hi)
        if frac(mids_for(k)) > target_frac:
            hi = k
        else:
            lo = k
    return mids_for(hi if frac(mids_for(hi)) <= target_frac * 1.003 else lo)


def select_keep(scores, mids):
    """Top-m channels per block, as sorted index tensors."""
    keep = []
    for s, m in zip(scores, mids):
        idx = torch.topk(s, m).indices.sort().values
        keep.append(idx)
    return keep
