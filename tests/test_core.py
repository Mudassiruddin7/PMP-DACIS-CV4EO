"""Unit tests for the parts where a silent bug would change the science.

Run:  python -m pytest -q tests      (or)      python tests/test_core.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pmp_dacis.model import PResNet18, count_macs, count_params, params_for_mids, prune_model  # noqa: E402
from pmp_dacis.prune import FULL_MIDS, allocate_mids, rank_norm, select_keep  # noqa: E402
from pmp_dacis.evaluate import episode_accuracy, make_episodes  # noqa: E402


def _model(seed=0, in_ch=3):
    torch.manual_seed(seed)
    m = PResNet18(in_ch=in_ch)
    for mod in m.modules():                       # make BN non-trivial so the test is sensitive
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.3, 0.3)
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 1.5)
    return m.eval()


def test_param_count_matches_torchvision():
    import torchvision
    tv = torchvision.models.resnet18()
    n_tv = sum(p.numel() for n, p in tv.named_parameters() if not n.startswith("fc."))
    assert count_params(PResNet18()) == n_tv == 11_176_512


def test_surgery_equals_masking():
    """A structurally pruned net must equal the dense net whose pruned mid channels are zeroed."""
    m = _model()
    x = torch.randn(3, 3, 64, 64)
    keep, masks = [], []
    g = torch.Generator().manual_seed(1)
    for b in m.blocks:
        C = b.conv1.out_channels
        k = torch.randperm(C, generator=g)[: max(4, C // 3)].sort().values
        keep.append(k)
        mk = torch.zeros(C)
        mk[k] = 1
        masks.append(mk)
    for b, mk in zip(m.blocks, masks):
        b.mask = mk
    with torch.no_grad():
        ref = m.features(x)
    for b in m.blocks:
        b.mask = None
    pruned = prune_model(m, keep).eval()
    with torch.no_grad():
        out = pruned.features(x)
    assert torch.allclose(ref, out, atol=1e-5), float((ref - out).abs().max())
    assert pruned.mids() == [len(k) for k in keep]
    assert count_params(pruned) == params_for_mids(m, [len(k) for k in keep])


def test_params_for_mids_is_exact_after_repeated_pruning():
    m = _model()
    dense = count_params(m)
    keep = [torch.arange(0, c, 2) for c in m.mids()]
    p1 = prune_model(m, keep)
    keep2 = [torch.arange(0, c, 2) for c in p1.mids()]
    target = [len(k) for k in keep2]
    assert count_params(prune_model(p1, keep2)) == params_for_mids(p1, target)
    assert count_params(p1) < dense


def test_allocation_hits_budget_and_respects_caps():
    m = _model()
    dense = count_params(m)
    for t in (0.5, 0.3, 0.2, 0.1):
        for tilt in (0.0, 0.5):
            mids = allocate_mids(m, t, dense, tilt=tilt)
            frac = params_for_mids(m, mids) / dense
            assert abs(frac - t) < 0.01, (t, tilt, frac)
            assert all(a <= b for a, b in zip(mids, FULL_MIDS))
    # second stage can only remove channels
    mids1 = allocate_mids(m, 0.6, dense, tilt=0.5)
    m1 = prune_model(m, [torch.arange(k) for k in mids1])
    mids2 = allocate_mids(m1, 0.3, dense, tilt=0.5, caps=m1.mids())
    assert all(a <= b for a, b in zip(mids2, mids1))
    assert abs(params_for_mids(m1, mids2) / dense - 0.3) < 0.01
    # deeper blocks lose a larger share when tilt > 0
    r = [a / b for a, b in zip(allocate_mids(m, 0.3, dense, tilt=0.5), FULL_MIDS)]
    assert r[0] > r[-1]


def test_rank_norm_and_select_keep():
    v = torch.tensor([0.3, 5.0, -1.0, 2.0])
    assert torch.allclose(rank_norm(v), torch.tensor([1 / 3, 1.0, 0.0, 2 / 3]))
    keep = select_keep([v], [2])[0]
    assert keep.tolist() == [1, 3]


def test_macs_scale_with_pruning():
    m = _model()
    full = count_macs(m, 64)
    half = count_macs(prune_model(m, [torch.arange(c // 2) for c in m.mids()]), 64)
    assert half < 0.7 * full


def test_episode_accuracy_perfect_and_chance():
    rng = np.random.default_rng(0)
    K, per = 8, 40
    labels = np.repeat(np.arange(K), per)
    centers = torch.nn.functional.normalize(torch.randn(K, 32), dim=1)
    feats = torch.nn.functional.normalize(centers[torch.from_numpy(labels)] + 0.01 * torch.randn(K * per, 32), dim=1)
    eps = make_episodes(labels, 100, 5, 1, 15, seed=3)
    assert episode_accuracy(feats, eps, 1).mean() > 0.99
    rand = torch.nn.functional.normalize(torch.randn(K * per, 32), dim=1)
    assert abs(episode_accuracy(rand, eps, 1).mean() - 0.2) < 0.03
    # paired episodes: identical for identical seeds, class-disjoint support/query within a class pool
    assert (make_episodes(labels, 10, 5, 5, 15, seed=3) == make_episodes(labels, 10, 5, 5, 15, seed=3)).all()
    e = make_episodes(labels, 50, 5, 5, 15, seed=4)
    assert all(len(set(row)) == 20 for ep in e for row in ep)          # no repeated image inside an episode row
    assert rng is not None




def test_pipeline_order_regression_and_cache_reuse():
    """dense_m must not clobber the cache that M->P reads (this broke a first notebook run), and finished runs
    must be reused only when the configuration is unchanged."""
    import json
    import tempfile
    from pmp_dacis import data as D
    from pmp_dacis import experiments as E
    torch.manual_seed(0)
    K, per = 38, 24
    y = torch.arange(K).repeat_interleave(per)
    X = torch.randint(0, 256, (K * per, 3, 32, 32), dtype=torch.uint8)
    X[:, 0] = (y.view(-1, 1, 1) * 6).to(torch.uint8)          # class-dependent colour, so episodes are learnable
    ds = D.DS("plantvillage", X, y, [str(i) for i in range(K)], "rgb8", 32)
    cfg = dict(E.PROFILES["smoke"])
    cfg.update(pv_res=32, s0_epochs=1, m_episodes=4, probe_episodes=4, eval_episodes=20, shots=[1, 5], query=4, bs=32,
               calib_per_class=4, res_sweep=[32], max_specs=None, amp=False)
    out = tempfile.mkdtemp()
    ctx = E.Ctx("plantvillage", 0, cfg, torch.device("cpu"), {"plantvillage": ds}, out)
    grid = {s["id"]: s for s in E.grid_for("plantvillage")}
    for sid in ("dense_m", "B_dacis_MP_30", "B_dacis_PGP_30", "A_l1_PMP_30", "R_random_P_30_r3", "K_mobilenet_v3_small"):
        spec = dict(grid[sid]); spec["bench"] = False; spec["res_sweep"] = False
        rec = E.run_spec(ctx, spec)
        assert 0 < rec["acc"]["novel_5"]["mean"] <= 100
    again = E.run_spec(ctx, dict(grid["B_dacis_MP_30"], bench=False, res_sweep=False))
    assert again["run_hash"] == rec["run_hash"] or again["spec"]["id"] == "B_dacis_MP_30"
    cfg2 = dict(cfg, gamma=2.0)                                   # a changed configuration must not reuse old results
    ctx2 = E.Ctx("plantvillage", 0, cfg2, torch.device("cpu"), {"plantvillage": ds}, out)
    spec = dict(grid["dense_m"], bench=False, res_sweep=False)
    assert E._h([ctx2.h_all, spec]) != json.load(open(os.path.join(ctx.dir, "dense_m.json")))["run_hash"]


def test_tune_m_lr_runs_and_is_cached():
    import tempfile
    from pmp_dacis import data as D
    from pmp_dacis import experiments as E
    torch.manual_seed(1)
    K, per = 38, 24
    y = torch.arange(K).repeat_interleave(per)
    X = torch.randint(0, 256, (K * per, 3, 32, 32), dtype=torch.uint8)
    ds = D.DS("plantvillage", X, y, [str(i) for i in range(K)], "rgb8", 32)
    cfg = dict(E.PROFILES["smoke"])
    cfg.update(pv_res=32, s0_epochs=1, m_episodes=4, eval_episodes=20, shots=[1, 5], query=4, bs=32, m_lr="auto",
               m_lr_grid=[2.5e-4, 1e-3], max_specs=None, amp=False)
    out = tempfile.mkdtemp()
    best = E.tune_m_lr(cfg, torch.device("cpu"), {"plantvillage": ds}, out)
    assert best in cfg["m_lr_grid"]
    assert E.tune_m_lr(cfg, torch.device("cpu"), {"plantvillage": ds}, out) == best      # second call reads the cache


def test_cfg_hash_keeps_other_variants_stable():
    """Adding the RESISC45 settings must not invalidate cached PlantVillage / EuroSAT runs."""
    from pmp_dacis import experiments as E
    cfg = dict(E.PROFILES["paper"])
    other = dict(cfg, rs_s0_epochs=cfg["rs_s0_epochs"] + 1, rs_res=96)
    for v in ("plantvillage", "eurosat13", "eurosat3"):
        assert E.cfg_hash(cfg, variant=v) == E.cfg_hash(other, variant=v)
        assert E.cfg_hash(cfg, E._S0_KEYS, v) == E.cfg_hash(other, E._S0_KEYS, v)
    assert E.cfg_hash(cfg, variant="resisc45") != E.cfg_hash(other, variant="resisc45")


def test_grids_have_unique_ids_and_known_schedules():
    from pmp_dacis import experiments as E
    for v in ("plantvillage", "eurosat13", "eurosat3", "resisc45"):
        g = E.grid_for(v)
        ids = [s["id"] for s in g]
        assert len(ids) == len(set(ids)), v
        for s in g:
            E.ops_for(s["sched"], s["t"], 0.6)           # raises KeyError for an unknown schedule
    assert any(s["id"].startswith("K_") for s in E.grid_for("resisc45"))


def test_compact_net_takes_thirteen_bands():
    from pmp_dacis.model import make_compact
    for arch in ("mobilenet_v3_small", "efficientnet_b0"):
        m = make_compact(arch, 13).eval()
        with torch.no_grad():
            f = m.features(torch.randn(2, 13, 64, 64))
        assert f.shape == (2, m.feat_dim)
        w = m.body[0][0].weight
        assert w.shape[1] == 13 and torch.allclose(w[:, 0], w[:, 3])      # RGB filters repeated cyclically


def test_resisc45_variant_runs_end_to_end():
    import tempfile
    from pmp_dacis import data as D
    from pmp_dacis import experiments as E
    torch.manual_seed(2)
    K, per = 45, 20
    y = torch.arange(K).repeat_interleave(per)
    X = torch.randint(0, 256, (K * per, 3, 32, 32), dtype=torch.uint8)
    X[:, 1] = (y.view(-1, 1, 1) * 5).to(torch.uint8)
    ds = D.DS("resisc45", X, y, [str(i) for i in range(K)], "rgb8", 32)
    cfg = dict(E.PROFILES["smoke"])
    cfg.update(rs_res=32, rs_s0_epochs=1, rs_r_epochs=1, rs_m_episodes=4, eval_episodes=20, shots=[1, 5], query=4, bs=32,
               calib_per_class=4, res_sweep=[32], max_specs=None, amp=False)
    out = tempfile.mkdtemp()
    ctx = E.Ctx("resisc45", 0, cfg, torch.device("cpu"), {"resisc45": ds}, out)
    assert (len(ctx.base), len(ctx.val), len(ctx.novel)) == (25, 5, 15)
    assert ctx.m_ep == 4 and ctx.e0 == 1
    grid = {s["id"]: s for s in E.grid_for("resisc45")}
    for sid in ("dense_ce", "A_taylor_PMP_30", "K_mobilenet_v3_small"):
        rec = E.run_spec(ctx, dict(grid[sid], bench=False, res_sweep=False))
        assert 0 < rec["acc"]["novel_5"]["mean"] <= 100


def test_mbv2_surgery_equals_masking_and_budget_is_exact():
    """Cutting hidden channels of a MobileNetV2 must equal zeroing them after the depthwise stage, and the parameter
    arithmetic used by the budget solver must be exact."""
    from pmp_dacis.model import make_compact, mbv2_blocks, prune_mbv2
    m = make_compact("mobilenet_v2", 3).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.3, 0.3)
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 1.5)
    blocks = mbv2_blocks(m)
    g = torch.Generator().manual_seed(1)
    keep, hooks = [], []
    for b in blocks:
        C = b.conv[0][0].out_channels
        k = torch.randperm(C, generator=g)[: max(4, C // 3)].sort().values
        keep.append(k)
        mk = torch.zeros(C)
        mk[k] = 1
        hooks.append(b.conv[1].register_forward_hook(lambda mod, i, o, mk=mk: o * mk.view(1, -1, 1, 1)))
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ref = m.features(x)
    for h in hooks:
        h.remove()
    coefs, cur, _ = m.mid_spec()
    pruned = prune_mbv2(m, keep).eval()
    with torch.no_grad():
        out = pruned.features(x)
    assert torch.allclose(ref, out, atol=1e-4), float((ref - out).abs().max())
    assert count_params(pruned) == count_params(m) + sum(c * (len(k) - cu) for c, k, cu in zip(coefs, keep, cur))
    assert pruned.mids() == [len(k) for k in keep]


def test_mbv2_allocation_hits_budget():
    from pmp_dacis.model import make_compact
    from pmp_dacis.prune import allocate_linear
    m = make_compact("mobilenet_v2", 3)
    coefs, cur, full = m.mid_spec()
    dense = count_params(m)
    for t in (0.6, 0.4, 0.25):
        mids = allocate_linear(coefs, cur, full, dense, dense, t)
        frac = (dense + sum(c * (a - b) for c, a, b in zip(coefs, mids, cur))) / dense
        assert abs(frac - t) < 0.02, (t, frac)


def test_compact_network_pruning_runs_end_to_end():
    import tempfile
    from pmp_dacis import data as D
    from pmp_dacis import experiments as E
    torch.manual_seed(3)
    K, per = 45, 20
    y = torch.arange(K).repeat_interleave(per)
    X = torch.randint(0, 256, (K * per, 3, 32, 32), dtype=torch.uint8)
    X[:, 1] = (y.view(-1, 1, 1) * 5).to(torch.uint8)
    ds = D.DS("resisc45", X, y, [str(i) for i in range(K)], "rgb8", 32)
    cfg = dict(E.PROFILES["smoke"])
    cfg.update(rs_res=32, rs_s0_epochs=1, rs_r_epochs=1, rs_m_episodes=4, eval_episodes=20, shots=[1, 5], query=4, bs=32,
               calib_per_class=4, res_sweep=[32], max_specs=None, amp=False)
    out = tempfile.mkdtemp()
    ctx = E.Ctx("resisc45", 0, cfg, torch.device("cpu"), {"resisc45": ds}, out)
    grid = {s["id"]: s for s in E.grid_for("resisc45")}
    ref = E.run_spec(ctx, dict(grid["MK_mobilenet_v2_dense"], bench=False))
    assert abs(ref["param_frac"] - 1.0) < 1e-6
    for sid, frac in (("MP_taylor_P_40", 0.4), ("MP_dacis_P_25", 0.25), ("MP_random_P_60", 0.6)):
        rec = E.run_spec(ctx, dict(grid[sid], bench=False))
        assert abs(rec["param_frac"] - frac) < 0.03, (sid, rec["param_frac"])
        assert 0 < rec["acc"]["novel_5"]["mean"] <= 100


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
