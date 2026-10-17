"""Experiment orchestration: profiles, run grids, schedules, caching."""
import copy
import hashlib
import json
import os
import time
import zlib

import numpy as np
import torch
import torch.nn as nn

from . import data as D
from .evaluate import embed, episode_accuracy, host_info, latency_ms, make_episodes, peak_mem_mb
from .model import CompactNet, count_macs, count_params, make_compact, make_model, mbv2_views, prune_model
from .prune import allocate_linear, allocate_mids, calib_indices, collect_stats, criterion_scores, select_keep
from .train import train_ce, train_episodic

PROFILES = {
    "paper": dict(
        seeds=[0, 1, 2, 3, 4], pv_res=128, es_res=64, n_base=24, n_val=6,
        s0_epochs=6, s0_lr=0.02, r_epochs=1, r_lr=0.01, m_episodes=1000, m_lr="auto",
        m_lr_grid=[1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3],
        probe_episodes=200, eval_episodes=2000, shots=[1, 5, 10], query=15, way=5, bs=128,
        calib_per_class=40, tilt=0.5, k1=0.6, gamma=1.0, lambdas=(0.3, 0.2, 0.5),
        es_s0_epochs=8, es_r_epochs=2, es_m_episodes=800, es_n_base=5,
        rs_res=128, rs_n_base=25, rs_n_val=5, rs_s0_epochs=10, rs_r_epochs=2, rs_m_episodes=1000,
        res_sweep=[64, 96, 128, 160, 224], amp=True, limit=None,
    ),
    "smoke": dict(
        seeds=[0], pv_res=64, es_res=64, n_base=24, n_val=6,
        s0_epochs=1, s0_lr=0.02, r_epochs=1, r_lr=0.01, m_episodes=20, m_lr=0.00025, m_lr_grid=[2.5e-4, 1e-3],
        probe_episodes=10, eval_episodes=50, shots=[1, 5], query=5, way=5, bs=64,
        calib_per_class=8, tilt=0.5, k1=0.6, gamma=1.0, lambdas=(0.3, 0.2, 0.5),
        es_s0_epochs=1, es_r_epochs=1, es_m_episodes=20, es_n_base=5,
        rs_res=64, rs_n_base=25, rs_n_val=5, rs_s0_epochs=1, rs_r_epochs=1, rs_m_episodes=20,
        res_sweep=[64, 96], amp=False, limit=6000, max_specs=3,
    ),
}

# Test profile: the full grid on a few hundred images per dataset, seconds per run. It checks that every code path
# (all variants, all specs, report, paper) runs; its numbers mean nothing.
PROFILES["micro"] = dict(
    PROFILES["smoke"], seeds=[0, 1], pv_res=128, es_res=64, rs_res=128, bs=32, m_lr="auto", m_lr_grid=[2.5e-4, 1e-3],
    m_episodes=10, es_m_episodes=10, rs_m_episodes=10, probe_episodes=5, eval_episodes=30, shots=[1, 5], query=5,
    calib_per_class=8, amp=True, limit=None, max_specs=None, subsample=1800, res_sweep=[64, 128],
)

# keys that decide what the dense S0 network looks like (its cache must not depend on later stages)
_S0_KEYS = ("pv_res", "es_res", "n_base", "n_val", "s0_epochs", "s0_lr", "bs", "amp", "limit", "es_s0_epochs", "es_n_base",
            "rs_res", "rs_n_base", "rs_n_val", "rs_s0_epochs")
_RS_KEYS = ("rs_res", "rs_n_base", "rs_n_val", "rs_s0_epochs", "rs_r_epochs", "rs_m_episodes")   # only the RESISC45 variant depends on these


def _h(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def cfg_hash(cfg, keys=None, variant=None):
    keys = keys or [k for k in cfg if k not in ("seeds", "max_specs")]
    if variant != "resisc45":                       # keeps the hashes (and cached runs) of the other variants unchanged
        keys = [k for k in keys if k not in _RS_KEYS]
    return _h({k: cfg[k] for k in keys if k in cfg})


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def ops_for(sched, t, k1):
    return {
        "compact": [], "compactR": [("R",)], "dense": [], "dense_m": [("M", 1)], "dense_rmr": [("R",), ("M", 1), ("R",)],
        "P": [("P", t), ("R",)],
        "PM": [("P", t), ("R",), ("M", 1)],
        "PM2x": [("P", t), ("R",), ("M", 2)],
        "MP": [("M", 1), ("P", t), ("R",)],
        "PMP": [("P", k1), ("R",), ("M", 1), ("P", t), ("R",)],
        "PMPM": [("P", k1), ("R",), ("M", 1), ("P", t), ("R",), ("M", 1)],
        "PGP": [("P", k1), ("R",), ("G", 1), ("P", t), ("R",)],      # gradient probe: no weight update in the meta stage
    }[sched]


class Ctx:
    """Everything for one (dataset variant, seed): data, class split, cached dense model, eval episodes."""

    def __init__(self, variant, seed, cfg, device, datasets, out_root, fh=None):
        self.variant, self.seed, self.cfg, self.device, self.fh = variant, seed, cfg, device, fh
        self.dir = os.path.join(out_root, "runs", variant, f"seed{seed}")
        os.makedirs(self.dir, exist_ok=True)
        self.h_all = cfg_hash(cfg, variant=variant)
        self.h_s0 = cfg_hash(cfg, _S0_KEYS, variant)
        if variant == "plantvillage":
            ds = datasets["plantvillage"]
            self.n_base = cfg["n_base"]
            self.split = D.class_split(len(ds.class_names), seed, cfg["n_base"], cfg["n_val"])
            self.ds = ds
            self.res = cfg["pv_res"]
            self.e0, self.r_ep, self.m_ep = cfg["s0_epochs"], cfg["r_epochs"], cfg["m_episodes"]
        elif variant == "resisc45":
            ds = datasets["resisc45"]
            self.split = D.class_split(len(ds.class_names), seed, cfg["rs_n_base"], cfg["rs_n_val"])
            self.ds = ds
            self.res = cfg["rs_res"]
            self.e0, self.r_ep, self.m_ep = cfg["rs_s0_epochs"], cfg["rs_r_epochs"], cfg["rs_m_episodes"]
        else:
            ds = datasets["eurosat"]
            ds = ds.with_bands(D.ES_RGB) if variant == "eurosat3" else ds.with_bands(None)
            self.split = D.class_split(len(ds.class_names), seed, cfg["es_n_base"], 0)
            self.ds = ds
            self.res = cfg["es_res"]
            self.e0, self.r_ep, self.m_ep = cfg["es_s0_epochs"], cfg["es_r_epochs"], cfg["es_m_episodes"]
        self.base, self.val, self.novel = self.split
        by = self.ds.idx_by_class()
        base_idx = torch.cat([by[c] for c in self.base])
        self.ds.res = self.res
        self.ds.set_norm(base_idx)
        self.eval_sets = {}
        nov_idx = torch.cat([by[c] for c in self.novel])
        lut = {c: i for i, c in enumerate(self.novel)}
        self.eval_sets["novel"] = (self.ds, nov_idx, np.array([lut[int(c)] for c in self.ds.y[nov_idx]]))
        if variant == "plantvillage" and datasets.get("plantdoc") is not None:
            pdoc = datasets["plantdoc"]
            pdoc.res = self.res
            need = max(cfg["shots"]) + cfg["query"]            # classes too small for an episode are dropped
            keep_c = [c for c, ix in pdoc.idx_by_class().items() if len(ix) >= need]
            sel = torch.cat([pdoc.idx_by_class()[c] for c in keep_c])
            lut_pd = {c: i for i, c in enumerate(keep_c)}
            self.eval_sets["plantdoc"] = (pdoc, sel, np.array([lut_pd[int(c)] for c in pdoc.y[sel]]))
            self.plantdoc_classes = len(keep_c)
        self.val_set = None
        if len(self.val):
            vidx = torch.cat([by[c] for c in self.val])
            vlut = {c: i for i, c in enumerate(self.val)}
            self.val_set = (self.ds, vidx, np.array([vlut[int(c)] for c in self.ds.y[vidx]]))
        self._eps = {}
        self.dense_params = None

    def episodes(self, name, shot):
        key = (name, shot)
        if key not in self._eps:
            lab = self.eval_sets[name][2] if name != "val" else self.val_set[2]
            self._eps[key] = make_episodes(lab, self.cfg["eval_episodes"], self.cfg["way"], shot, self.cfg["query"],
                                           seed=self.seed * 100003 + shot * 101 + (7 if name == "plantdoc" else 0)
                                           + (13 if name == "val" else 0))
        return self._eps[key]

    # ---- cached dense models -------------------------------------------------------------
    def _rng(self, tag):
        return np.random.default_rng([self.seed, zlib.crc32(tag.encode())])

    def dense(self):
        path = os.path.join(self.dir, f"cache_dense_s0_{self.h_s0}.pt")
        m = make_model(self.ds.in_ch, pretrained=True).to(self.device)
        if os.path.exists(path):
            ck = torch.load(path, map_location=self.device)
            m.fc = nn.Linear(512, ck["n_fc"]).to(self.device)
            m.load_state_dict(ck["sd"])
            self.dense_params = count_params(m)
            return m, ck["log"]
        torch.manual_seed(self.seed)
        lg = train_ce(m, self.ds, self.base, self.e0, self.cfg["s0_lr"], self.device, self._rng("s0"),
                      bs=self.cfg["bs"], amp=self.cfg["amp"], new_head=True)
        torch.save({"sd": m.state_dict(), "n_fc": len(self.base), "log": lg}, path)
        self.dense_params = count_params(m)
        log(f"  [{self.variant} s{self.seed}] dense S0: {lg['steps']} steps, {lg['seconds']:.0f}s, loss {lg['loss']:.3f}", self.fh)
        return m, lg

    def dense_arch(self, arch):
        """Cached S0 network of an off-the-shelf architecture (the starting point of its pruning runs)."""
        path = os.path.join(self.dir, f"cache_dense_{arch}_s0_{self.h_s0}.pt")
        m = make_compact(arch, self.ds.in_ch).to(self.device)
        if os.path.exists(path):
            ck = torch.load(path, map_location=self.device)
            m.fc = nn.Linear(m.feat_dim, ck["n_fc"]).to(self.device)
            m.load_state_dict(ck["sd"])
            return m
        torch.manual_seed(self.seed)
        lg = train_ce(m, self.ds, self.base, self.e0, self.cfg["s0_lr"], self.device, self._rng("s0_" + arch),
                      bs=self.cfg["bs"], amp=self.cfg["amp"], new_head=True)
        torch.save({"sd": m.state_dict(), "n_fc": len(self.base), "log": lg}, path)
        log(f"  [{self.variant} s{self.seed}] {arch} S0: {lg['steps']} steps, {lg['seconds']:.0f}s, loss {lg['loss']:.3f}", self.fh)
        return m

    def dense_m(self):
        path = os.path.join(self.dir, f"cache_dense_m_{self.h_all}.pt")
        m, _ = self.dense()
        if os.path.exists(path):
            ck = torch.load(path, map_location=self.device)
            if all(k in ck for k in ("sd", "gmeta", "log")):
                m.load_state_dict(ck["sd"])
                return m, ck["gmeta"], ck["log"]
        torch.manual_seed(self.seed)
        lg = train_episodic(m, self.ds, self.base, self.m_ep, self.cfg["m_lr"], self.device, self._rng("dm"),
                            way=self.cfg["way"], query=self.cfg["query"], amp=self.cfg["amp"])
        gm = lg.pop("gmeta")
        torch.save({"sd": m.state_dict(), "gmeta": gm, "log": lg}, path)
        return m, gm, lg


# ---- tuning of the one shared knob that the first experiments showed to matter ----------------
def tune_m_lr(cfg, device, datasets, out_root, fh=None, variant="plantvillage"):
    """Choose the learning rate of the episodic stage on the *validation classes* of the first seed.

    Only the dense network is used and no novel class is touched. The result is written to tuning_m_lr.json
    (PlantVillage, the value every variant uses) or tuning_m_lr_<variant>.json (a check on another dataset).
    """
    path = os.path.join(out_root, "tuning_m_lr.json" if variant == "plantvillage" else f"tuning_m_lr_{variant}.json")
    key = cfg_hash(cfg, [k for k in cfg if k not in ("seeds", "max_specs", "m_lr")], variant)
    if os.path.exists(path):
        rec = json.load(open(path))
        if rec.get("cfg_hash") == key:
            return rec["best_lr"]
    seed = cfg["seeds"][0]
    probe_cfg = dict(cfg, m_lr=cfg["m_lr_grid"][0])
    ctx = Ctx(variant, seed, probe_cfg, device, datasets, out_root, fh)
    ds_e, vidx, _ = ctx.val_set
    shots = [1, 5]

    def val_acc(model):
        f = embed(model, ds_e, vidx, device, amp=cfg["amp"])
        return {str(k): float(episode_accuracy(f, ctx.episodes("val", k), k).mean() * 100) for k in shots}

    base, _ = ctx.dense()
    rows = {"S0 (no episodic stage)": val_acc(base)}
    for lr in cfg["m_lr_grid"]:
        m, _ = ctx.dense()
        torch.manual_seed(seed)
        t0 = time.time()
        train_episodic(m, ctx.ds, ctx.base, ctx.m_ep, lr, device, ctx._rng("tune"), way=cfg["way"], query=cfg["query"], amp=cfg["amp"])
        rows[f"{lr:g}"] = val_acc(m)
        log(f"  [tune M lr {variant}] {lr:g}: val 1-shot {rows[f'{lr:g}']['1']:.2f}  5-shot {rows[f'{lr:g}']['5']:.2f}  ({time.time() - t0:.0f}s)", fh)
    best = max(cfg["m_lr_grid"], key=lambda lr: rows[f"{lr:g}"]["5"])
    json.dump({"cfg_hash": key, "variant": variant, "seed": seed, "val_classes": ctx.val, "rows": rows, "best_lr": best},
              open(path, "w"), indent=1)
    log(f"  [tune M lr {variant}] best validation 5-shot accuracy at {best:g}", fh)
    return best


# ---- one run ------------------------------------------------------------------------------
def _stage_prune(ctx, model, spec, frac, state, rng):
    cfg = ctx.cfg
    idx, lab = calib_indices(ctx.ds, ctx.base, cfg["calib_per_class"], ctx._rng("calib"))
    compact = isinstance(model, CompactNet)
    if compact:
        with mbv2_views(model) as views:
            stats = collect_stats(model, ctx.ds, idx, lab, ctx.device, blocks=views)
    else:
        stats = collect_stats(model, ctx.ds, idx, lab, ctx.device)
    lam = spec.get("lambdas", cfg["lambdas"])
    scfg = {"lambda_g": lam[0], "lambda_v": lam[1], "lambda_d": lam[2]}
    gm = state.get("gmeta")
    use_refine = spec["crit"] == "dacis" and spec.get("refine", True) and gm is not None
    scores = criterion_scores(spec["crit"], stats, scfg, rng, gmeta=gm, norm=spec.get("norm", "rank"),
                              refine_gamma=cfg["gamma"] if use_refine else 0.0)
    if compact:
        coefs, cur, full = model.mid_spec()
        mids = allocate_linear(coefs, cur, full, count_params(model), state["dense_params"], frac, tilt=spec.get("tilt", cfg["tilt"]))
    else:
        mids = allocate_mids(model, frac, ctx.dense_params, tilt=spec.get("tilt", cfg["tilt"]), caps=model.mids())
    keep = select_keep(scores, mids)
    new = prune_model(model, keep)
    state["stages"].append({"op": "P", "frac": frac, "mids": mids, "params": count_params(new),
                            "refined": bool(use_refine), "keep": [k.tolist() for k in keep]})
    return new


def _evaluate(ctx, model, spec):
    cfg, out, arrs = ctx.cfg, {}, {}
    sets = dict(ctx.eval_sets)
    if spec.get("val_eval") and ctx.val_set is not None:
        sets["val"] = ctx.val_set
    for name, (ds_e, idx, lab) in sets.items():
        feats = embed(model, ds_e, idx, ctx.device, amp=cfg["amp"])
        for shot in cfg["shots"]:
            acc = episode_accuracy(feats, ctx.episodes(name, shot), shot)
            arrs[f"{name}_{shot}"] = acc.astype(np.float32)
            out[f"{name}_{shot}"] = {"mean": float(acc.mean() * 100), "std": float(acc.std() * 100),
                                     "ci95": float(1.96 * acc.std() / np.sqrt(len(acc)) * 100)}
    if spec.get("res_sweep"):
        ds_e, idx, lab = ctx.eval_sets["novel"]
        sweep = {}
        for r in cfg["res_sweep"]:
            feats = embed(model, ds_e, idx, ctx.device, res=r, amp=cfg["amp"])
            acc = episode_accuracy(feats, ctx.episodes("novel", 5), 5)
            sweep[str(r)] = float(acc.mean() * 100)
            arrs[f"res{r}_5"] = acc.astype(np.float32)
        out["res_sweep_5shot"] = sweep
    return out, arrs


def _efficiency(ctx, model, spec):
    dev = ctx.device
    cin = ctx.ds.in_ch
    eff = {"params": count_params(model), "macs_128": count_macs(model, 128, cin), "macs_224": count_macs(model, 224, cin),
           "macs_res": count_macs(model, ctx.res, cin)}
    if not spec.get("bench"):
        return eff
    cpu_model = copy.deepcopy(model).cpu().float()
    r = ctx.res
    eff["cpu_1thread_b1_ms"] = latency_ms(cpu_model, cin, r, torch.device("cpu"), batch=1, threads=1, iters=60)
    eff["cpu_all_threads_b1_ms"] = latency_ms(cpu_model, cin, r, torch.device("cpu"), batch=1,
                                              threads=os.cpu_count(), iters=100)
    if dev.type == "cuda":
        eff["gpu_b1_ms"] = latency_ms(model, cin, r, dev, batch=1, iters=200)
        g = latency_ms(model, cin, r, dev, batch=128, iters=30, warm=5)
        eff["gpu_b128_ms"] = g
        eff["gpu_b128_img_per_s"] = 128000.0 / g["median"]
        eff["gpu_peak_mem_b1_mb"] = peak_mem_mb(model, cin, r, dev, 1)
    return eff


KEEP_CKPT = ("dense_ce", "A_random_PMP_30", "A_l1_PMP_30", "A_dacis_PMP_30")


def run_spec(ctx, spec):
    path = os.path.join(ctx.dir, f"{spec['id']}.json")
    key = _h([ctx.h_all, spec])
    if os.path.exists(path):
        rec = json.load(open(path))
        if rec.get("run_hash") == key:
            return rec                                    # finished earlier with the same configuration
    t0 = time.time()
    cfg = ctx.cfg
    torch.manual_seed(ctx.seed)
    rng = ctx._rng(spec["id"])
    state = {"gmeta": None, "stages": []}
    ops = ops_for(spec["sched"], spec["t"], cfg["k1"])
    r_epochs = spec.get("r_epochs", ctx.r_ep)
    train_s, train_img = 0.0, 0
    if spec.get("arch"):                                  # off-the-shelf compact network: S0 recipe, then R, no pruning
        model = make_compact(spec["arch"], ctx.ds.in_ch).to(ctx.device)
        if ctx.dense_params is None:
            ctx.dense_params = count_params(make_model(ctx.ds.in_ch, pretrained=False))
        lg = train_ce(model, ctx.ds, ctx.base, ctx.e0, cfg["s0_lr"], ctx.device, rng, bs=cfg["bs"], amp=cfg["amp"], new_head=True)
        state["stages"].append({"op": "S0", "epochs": ctx.e0, **{k: lg[k] for k in ("steps", "images", "seconds", "loss")}})
        ops = [("R",)]
    elif spec.get("prune_arch"):                          # compact network: cached S0, then prune / recover as for the ResNet
        model = ctx.dense_arch(spec["prune_arch"])
        state["dense_params"] = count_params(model)
    elif spec["sched"] in ("MP", "dense_m"):
        model, gm, lg = ctx.dense_m()
        state["gmeta"] = gm
        state["stages"].append({"op": "M", "episodes": ctx.m_ep, **{k: lg[k] for k in ("steps", "images", "seconds", "loss")}})
        ops = ops[1:]
    else:
        model, _ = ctx.dense()
    if ctx.dense_params is None:
        ctx.dense_params = count_params(ctx.dense()[0])
    for op in ops:
        if op[0] == "P":
            model = _stage_prune(ctx, model, spec, op[1], state, rng)
        elif op[0] == "R":
            lg = train_ce(model, ctx.ds, ctx.base, r_epochs, cfg["r_lr"], ctx.device, rng, bs=cfg["bs"], amp=cfg["amp"])
            state["stages"].append({"op": "R", "epochs": r_epochs, **{k: lg[k] for k in ("steps", "images", "seconds", "loss")}})
        elif op[0] == "G":
            lg = train_episodic(model, ctx.ds, ctx.base, cfg["probe_episodes"], 0.0, ctx.device, rng,
                                way=cfg["way"], query=cfg["query"], amp=cfg["amp"])
            state["gmeta"] = lg.pop("gmeta")
            state["stages"].append({"op": "G", "episodes": cfg["probe_episodes"], **{k: lg[k] for k in ("steps", "images", "seconds", "loss")}})
        elif op[0] == "M":
            lg = train_episodic(model, ctx.ds, ctx.base, ctx.m_ep * op[1], cfg["m_lr"], ctx.device, rng,
                                way=cfg["way"], query=cfg["query"], amp=cfg["amp"])
            state["gmeta"] = lg.pop("gmeta")
            state["stages"].append({"op": "M", "episodes": ctx.m_ep * op[1], **{k: lg[k] for k in ("steps", "images", "seconds", "loss")}})
    for st in state["stages"]:
        if st["op"] in ("R", "M", "G"):
            train_s += st["seconds"]
            train_img += st["images"]
    acc, arrs = _evaluate(ctx, model, spec)
    eff = _efficiency(ctx, model, spec)
    rec = {"spec": spec, "variant": ctx.variant, "seed": ctx.seed, "run_hash": key,
           "split": {"base": ctx.base, "val": ctx.val, "novel": ctx.novel},
           "acc": acc, "eff": eff, "param_frac": eff["params"] / (state.get("dense_params") or ctx.dense_params), "mids": model.mids(),
           "stages": state["stages"], "train_seconds_after_s0": train_s, "train_images_after_s0": train_img,
           "wall_seconds": time.time() - t0}
    np.savez_compressed(os.path.join(ctx.dir, f"{spec['id']}.npz"), **arrs)
    if spec["id"] in KEEP_CKPT:                           # weights are not needed to rebuild any table; keep a few for inspection
        torch.save({"mids": model.mids(), "sd": model.state_dict(), "in_ch": ctx.ds.in_ch},
                   os.path.join(ctx.dir, f"ckpt_{spec['id']}.pt"))
    json.dump(rec, open(path, "w"))
    a5 = acc.get("novel_5", {}).get("mean", float("nan"))
    log(f"  [{ctx.variant} s{ctx.seed}] {spec['id']:<28s} params {rec['param_frac']*100:5.1f}%  novel5 {a5:5.1f}  "
        f"({rec['wall_seconds']:.0f}s)", ctx.fh)
    return rec


# ---- grids --------------------------------------------------------------------------------
def _spec(id_, sched, crit, t, tier, **kw):
    d = dict(id=id_, sched=sched, crit=crit, t=t, tier=tier)
    d.update(kw)
    return d


def grid_for(variant):
    L = [_spec("dense_ce", "dense", "-", 1.0, 1, bench=True, res_sweep=True),
         _spec("dense_m", "dense_m", "-", 1.0, 1, res_sweep=True),
         _spec("dense_rmr", "dense_rmr", "-", 1.0, 1, res_sweep=True)]
    if variant == "plantvillage":
        for c in ["random", "l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"]:
            L.append(_spec(f"A_{c}_PMP_30", "PMP", c, 0.30, 1, bench=c in ("l1", "dacis"), res_sweep=c in ("l1", "dacis")))
        for c in ["random", "l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"]:        # same scores, plain prune-and-recover
            L.append(_spec(f"A2_{c}_P_30", "P", c, 0.30, 1, bench=False, res_sweep=c in ("l1", "dacis")))
        for s in ["PM", "MP", "PGP", "PMPM", "PM2x"]:
            L.append(_spec(f"B_dacis_{s}_30", s, "dacis", 0.30, 1))
        L.append(_spec("B_l1_PM_30", "PM", "l1", 0.30, 1))
        for t in (0.5, 0.2, 0.1):
            for c in ("l1", "dacis"):
                L.append(_spec(f"C_{c}_PMP_{int(round(t * 100))}", "PMP", c, t, 2, bench=True))
        L.append(_spec("E_dacis_PMP_30_norefine", "PMP", "dacis", 0.30, 2, refine=False))
        L.append(_spec("E_dacis_PMP_30_uniform", "PMP", "dacis", 0.30, 2, tilt=0.0))
        L.append(_spec("E_dacis_PMP_30_minmax", "PMP", "dacis", 0.30, 2, norm="minmax"))
        for c in ("GV", "GD", "VD"):
            L.append(_spec(f"A_{c}_PMP_30", "PMP", c, 0.30, 2))
        for c in ("random", "l1", "taylor", "dacis"):                       # does longer recovery erase the differences?
            for n in (3, 6):
                L.append(_spec(f"R_{c}_P_30_r{n}", "P", c, 0.30, 2, r_epochs=n))
        for arch in ("mobilenet_v3_small", "mobilenet_v2", "efficientnet_b0"):      # what a small pretrained net achieves
            L.append(_spec(f"K_{arch}", "compact", "-", 1.0, 2, arch=arch, bench=True))
        for i, lam in enumerate([(0.3, 0.2, 0.5), (1 / 3, 1 / 3, 1 / 3), (0.5, 0.2, 0.3), (0.2, 0.5, 0.3), (0.2, 0.2, 0.6),
                                 (0.1, 0.1, 0.8), (0.0, 0.0, 1.0), (0.5, 0.5, 0.0)]):
            L.append(_spec(f"S_dacis_PMP_30_lam{i}", "PMP", "dacis", 0.30, 3, lambdas=lam, val_eval=True))
    else:
        for c in ["random", "l1", "taylor", "D", "dacis"]:
            L.append(_spec(f"A_{c}_PMP_30", "PMP", c, 0.30, 1, bench=c in ("l1", "dacis"),
                           res_sweep=variant == "resisc45" and c in ("l1", "dacis")))
        for c in ["random", "l1", "taylor", "D", "dacis"]:
            L.append(_spec(f"A2_{c}_P_30", "P", c, 0.30, 1))
        for s in ["PM", "MP", "PGP"]:
            L.append(_spec(f"B_dacis_{s}_30", s, "dacis", 0.30, 1))
        for c in ["bn_gamma", "G", "V"]:
            L.append(_spec(f"A_{c}_PMP_30", "PMP", c, 0.30, 2))
        for t in (0.5, 0.2, 0.1):
            for c in ("l1", "dacis"):
                L.append(_spec(f"C_{c}_PMP_{int(round(t * 100))}", "PMP", c, t, 2, bench=True))
            for c in ("random", "taylor"):
                L.append(_spec(f"C_{c}_PMP_{int(round(t * 100))}", "PMP", c, t, 2))
        for arch in ("mobilenet_v3_small", "mobilenet_v2", "efficientnet_b0"):
            L.append(_spec(f"K_{arch}", "compact", "-", 1.0, 2, arch=arch, bench=True))
    for c in ("random", "l1", "bn_gamma", "taylor", "dacis"):      # MobileNetV2 pruned like the ResNet: tier 4
        for t in (0.6, 0.4, 0.25):
            L.append(_spec(f"MP_{c}_P_{int(round(t * 100))}", "P", c, t, 4, prune_arch="mobilenet_v2", bench=c in ("l1", "dacis")))
    L.append(_spec("MK_mobilenet_v2_dense", "compactR", "-", 1.0, 4, prune_arch="mobilenet_v2", bench=True))
    return L


# the extension run: more class partitions for the Earth-observation comparisons, and the pruned MobileNetV2 on ten
EXT_EXTRA = ("A_bn_gamma_PMP_30", "A_G_PMP_30", "A_V_PMP_30")


def ext_jobs(variants=("resisc45", "eurosat13"), new_seeds=(5, 6, 7, 8, 9), all_seeds=tuple(range(10))):
    jobs = []
    for v in variants:
        g = grid_for(v)
        main = [s for s in g if s["tier"] == 1 or s["id"] in EXT_EXTRA or s["id"].startswith("K_")]
        jobs.append((v, main, list(new_seeds)))
        jobs.append((v, [s for s in g if s["tier"] == 4], list(all_seeds)))
    return jobs


def load_datasets(data_root, cache_dir, cfg, which=("plantvillage", "plantdoc", "eurosat", "resisc45")):
    out = {}
    lim = cfg.get("limit")
    if "plantvillage" in which:
        out["plantvillage"] = D.load_plantvillage(data_root, cache_dir, cfg["pv_res"], limit=lim)
    if "plantdoc" in which:
        out["plantdoc"] = D.load_plantdoc(data_root, cache_dir, cfg["pv_res"], limit=lim)
    if "eurosat" in which:
        out["eurosat"] = D.load_eurosat(data_root, cache_dir, limit=lim)
    if "resisc45" in which:
        out["resisc45"] = D.load_resisc45(data_root, cache_dir, cfg["rs_res"], limit=lim)
    if cfg.get("subsample"):                              # micro profile only; PlantDoc is already small
        out = {k: (v if k == "plantdoc" else D.subsample(v, cfg["subsample"])) for k, v in out.items()}
    return out


def run_all(cfg, device, data_root, cache_dir, out_root, variants=("resisc45", "eurosat13", "eurosat3", "plantvillage"),
            tiers=(1, 2, 3), fh=None, jobs=None):
    cfg = dict(cfg)
    which = ["plantvillage", "plantdoc"] if "plantvillage" in variants else []
    if any(v.startswith("eurosat") for v in variants):
        which.append("eurosat")
    if "resisc45" in variants:
        which.append("resisc45")
    if cfg["m_lr"] == "auto" and "plantvillage" not in variants:
        which.append("plantvillage")                      # the shared learning rate is tuned on PlantVillage validation classes
    datasets = load_datasets(data_root, cache_dir, cfg, which)
    if cfg["m_lr"] == "auto":
        cfg["m_lr"] = tune_m_lr(cfg, device, datasets, out_root, fh)
    log(f"episodic learning rate in use: {cfg['m_lr']:g}", fh)
    if jobs is None and "resisc45" in variants and len(cfg.get("m_lr_grid", [])) > 1:      # same sweep on aerial-scene validation classes, for the record
        tune_m_lr(cfg, device, datasets, out_root, fh, variant="resisc45")
    failures = []
    if jobs is None:
        jobs = []
        for tier in tiers:
            for variant in variants:
                grid = [s for s in grid_for(variant) if s["tier"] == tier]
                if variant == "eurosat3":
                    grid = [s for s in grid if s["id"] in ("dense_ce", "dense_m", "dense_rmr", "A_l1_PMP_30", "A_dacis_PMP_30", "K_mobilenet_v3_small")]
                jobs.append((variant, grid, cfg["seeds"]))
    for variant, grid, seeds in jobs:
        if True:
            if cfg.get("max_specs"):                      # smoke profile: only check that the code path runs
                grid = grid[: cfg["max_specs"]]
            for seed in seeds:
                ctx = Ctx(variant, seed, cfg, device, datasets, out_root, fh)
                ctx.dense()
                for spec in grid:
                    try:
                        run_spec(ctx, spec)
                    except Exception as e:                    # one failing run must not throw away hours of finished ones
                        import traceback
                        failures.append({"variant": variant, "seed": seed, "id": spec["id"], "error": repr(e),
                                         "trace": traceback.format_exc()[-1500:]})
                        log(f"  [{variant} s{seed}] {spec['id']} FAILED: {e!r}", fh)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
    json.dump({k: v for k, v in cfg.items()}, open(os.path.join(out_root, "config_used.json"), "w"), indent=1, default=str)
    json.dump(failures, open(os.path.join(out_root, "failures.json"), "w"), indent=1)
    if failures:
        log(f"{len(failures)} runs FAILED (see failures.json); the report skips them", fh)
    return datasets
