"""Build assets/js/data.js for the project page from the saved runs of the study.

    python tools/make_data.py <path to the PMP-DACIS-CV4EO repository> <out data.js>

Every number on the page is read from results/runs/*/seed*/*.json (one file per trained network and class partition) and
from results_replication/latency_named_cpu.json. Paired differences use the paper's recipe: mean of the per-partition
differences with a 95% t-interval over partitions. The script also checks itself against the numbers printed in the paper
and stops if one of them does not come out the same.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

REPO = sys.argv[1]
OUT = sys.argv[2]
RUNS = os.path.join(REPO, "results", "runs")

RESISC45 = ["airplane", "airport", "baseball diamond", "basketball court", "beach", "bridge", "chaparral", "church",
            "circular farmland", "cloud", "commercial area", "dense residential", "desert", "forest", "freeway", "golf course",
            "ground track field", "harbor", "industrial area", "intersection", "island", "lake", "meadow", "medium residential",
            "mobile home park", "mountain", "overpass", "palace", "parking lot", "railway", "railway station",
            "rectangular farmland", "river", "roundabout", "runway", "sea ice", "ship", "snowberg", "sparse residential",
            "stadium", "storage tank", "tennis court", "terrace", "thermal power station", "wetland"]


# ---------------------------------------------------------------------------------------------------------------- loading
def load():
    runs = {}
    for variant in sorted(os.listdir(RUNS)):
        for sd in sorted(glob.glob(os.path.join(RUNS, variant, "seed*"))):
            seed = int(os.path.basename(sd)[4:])
            for f in glob.glob(os.path.join(sd, "*.json")):
                d = json.load(open(f, encoding="utf-8"))
                runs.setdefault(variant, {}).setdefault(d["spec"]["id"], {})[seed] = d
    return runs


R = load()


def acc(variant, rid, key="novel_5"):
    return {s: d["acc"][key]["mean"] for s, d in R[variant].get(rid, {}).items() if key in d["acc"]}


def ms(variant, rid, key="novel_5"):
    v = np.array(list(acc(variant, rid, key).values()))
    return dict(m=float(v.mean()), sd=float(v.std(ddof=1)), n=int(len(v)))


def paired(variant, a, b, key="novel_5"):
    x, y = acc(variant, a, key), acc(variant, b, key)
    seeds = sorted(set(x) & set(y))
    dif = np.array([x[s] - y[s] for s in seeds])
    m = dif.mean()
    se = dif.std(ddof=1) / np.sqrt(len(dif))
    t = stats.t.ppf(0.975, len(dif) - 1)
    return dict(m=float(m), lo=float(m - t * se), hi=float(m + t * se), n=int(len(dif)),
                ahead=int((dif > 0).sum()), seeds=seeds)


def r1(x):
    """round half away from zero to one decimal, as the paper's tables do"""
    v = float(np.sign(x) * np.floor(abs(x) * 10 + 0.5 + 1e-9) / 10)
    return 0.0 if v == 0 else v


def pr(d):
    return {k: (r1(v) if k in ("m", "lo", "hi") else v) for k, v in d.items() if k != "seeds"}


# ---------------------------------------------------------------------------------------------------------------- pieces
SC = ["random", "l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"]
SC_LABEL = {"random": "Random", "l1": "Filter L1", "bn_gamma": "BatchNorm scale", "taylor": "Taylor", "G": "Gradient",
            "V": "Variance", "D": "Fisher", "dacis": "DACIS"}
DS = ["resisc45", "eurosat13", "plantvillage"]

data = {}

# class partitions of RESISC45 (the same split for every run of a seed)
splits = []
for s in range(10):
    d = R["resisc45"]["dense_ce"][s]["split"]
    splits.append({"base": d["base"], "val": d["val"], "novel": d["novel"]})
data["partitions"] = {"classes": RESISC45, "splits": splits}

# channel counts kept per residual block (the same for every score)
orig = [64, 64, 128, 128, 256, 256, 512, 512]
prof = {"orig": orig, "kept": {"100": orig}}
for b, rid in ((50, "C_dacis_PMP_50"), (30, "A_dacis_PMP_30"), (20, "C_dacis_PMP_20"), (10, "C_dacis_PMP_10")):
    d = R["resisc45"][rid][0]
    prof["kept"][str(b)] = d["mids"]
    prof.setdefault("params", {})[str(b)] = d["param_frac"]
data["profile"] = prof

# which channels four scores actually kept: one pruning step to 30% (schedule P), partition 0, indices into the original blocks
picks = {"seed": 0, "orig": orig, "scores": {}, "acc": {}}
for sc in ("random", "l1", "taylor", "dacis"):
    d = R["resisc45"][f"A2_{sc}_P_30"][0]
    st = next(s for s in d["stages"] if s["op"] == "P")
    picks["scores"][sc] = [sorted(int(i) for i in k) for k in st["keep"]]
    picks["acc"][sc] = ms("resisc45", f"A2_{sc}_P_30")["m"]
ids = list(picks["scores"])
jac = {}
for a in ids:
    for b in ids:
        x, y = picks["scores"][a], picks["scores"][b]
        inter = sum(len(set(p) & set(q)) for p, q in zip(x, y))
        union = sum(len(set(p) | set(q)) for p, q in zip(x, y))
        jac[f"{a}|{b}"] = round(inter / union, 2)
picks["jaccard"] = jac
data["picks"] = picks

# strip chart: 8 scores at 30% (PMP), one value per partition
strip = {}
for ds, key in (("resisc45", "novel_5"), ("eurosat13", "novel_5"), ("plantvillage", "novel_5"), ("plantvillage", "plantdoc_5")):
    name = "plantdoc" if key == "plantdoc_5" else ds
    rows = []
    for sc in SC:
        rid = f"A_{sc}_PMP_30"
        a = acc(ds, rid, key)
        if not a:
            continue
        seeds = sorted(a)
        v = np.array([a[s] for s in seeds])
        rows.append({"id": sc, "label": SC_LABEL[sc], "seeds": seeds, "vals": [round(float(x), 2) for x in v],
                     "m": float(v.mean()), "sd": float(v.std(ddof=1))})
    dense = acc(ds, "dense_ce", key)
    dv = np.array(list(dense.values()))
    strip[name] = {"rows": rows, "dense": {"m": float(dv.mean()), "sd": float(dv.std(ddof=1)), "vals": [round(float(dense[s]), 2) for s in sorted(dense)]}}
data["strip"] = strip

# forest: gain over random in three settings, two datasets
settings = [
    ("r18_30", "ResNet-18 at 30% of the weights", "PMP", {"l1": "A_l1_PMP_30", "bn_gamma": "A_bn_gamma_PMP_30", "taylor": "A_taylor_PMP_30", "dacis": "A_dacis_PMP_30"}, "A_random_PMP_30"),
    ("mb_40", "MobileNetV2 at 40% of the weights", "P", {s: f"MP_{s}_P_40" for s in ("l1", "bn_gamma", "taylor", "dacis")}, "MP_random_P_40"),
    ("mb_25", "MobileNetV2 at 25% of the weights", "P", {s: f"MP_{s}_P_25" for s in ("l1", "bn_gamma", "taylor", "dacis")}, "MP_random_P_25"),
]
forest = []
for sid, label, sched, ids, base in settings:
    ent = {"id": sid, "label": label, "sched": sched, "ds": {}}
    for ds in ("resisc45", "eurosat13"):
        ent["ds"][ds] = {sc: pr(paired(ds, rid, base)) for sc, rid in ids.items()}
    forest.append(ent)
data["forest"] = forest

# lead over random as pruning gets harsher
lead = {}
for ds in ("resisc45", "eurosat13"):
    r18 = {"budgets": [50, 30, 20, 10], "scores": {}}
    for sc in ("l1", "taylor", "dacis"):
        arr = []
        for b in r18["budgets"]:
            pre = "A" if b == 30 else "C"
            arr.append(pr(paired(ds, f"{pre}_{sc}_PMP_{b}", f"{pre}_random_PMP_{b}")))
        r18["scores"][sc] = arr
    mb = {"budgets": [60, 40, 25], "scores": {}}
    for sc in ("l1", "bn_gamma", "taylor", "dacis"):
        mb["scores"][sc] = [pr(paired(ds, f"MP_{sc}_P_{b}", f"MP_random_P_{b}")) for b in mb["budgets"]]
    lead[ds] = {"r18": r18, "mbv2": mb}
data["lead"] = lead

# partition by partition: taylor against random
race = {}
for sid, ds, tay, rnd, bn in (("r18_30", "resisc45", "A_taylor_PMP_30", "A_random_PMP_30", "A_bn_gamma_PMP_30"),
                              ("mb_40", "resisc45", "MP_taylor_P_40", "MP_random_P_40", "MP_bn_gamma_P_40"),
                              ("mb_25", "resisc45", "MP_taylor_P_25", "MP_random_P_25", "MP_bn_gamma_P_25")):
    t, r, b = acc(ds, tay), acc(ds, rnd), acc(ds, bn)
    seeds = sorted(t)
    race[sid] = {"seeds": seeds, "taylor": [round(t[s], 2) for s in seeds], "random": [round(r[s], 2) for s in seeds],
                 "bn": [round(b[s], 2) for s in seeds], "ahead": int(sum(t[s] > r[s] for s in seeds))}
data["race"] = race

# schedules against plain prune-and-recover (DACIS at 30%)
sch = {}
for ds in ("resisc45", "eurosat13"):
    sch[ds] = [
        {"id": "PMP", "label": "prune, meta-train, prune", **pr(paired(ds, "A_dacis_PMP_30", "A2_dacis_P_30"))},
        {"id": "MP", "label": "meta-train, then prune", **pr(paired(ds, "B_dacis_MP_30", "A2_dacis_P_30"))},
        {"id": "PGP", "label": "prune, probe gradients, prune (no weight updates)", **pr(paired(ds, "B_dacis_PGP_30", "A2_dacis_P_30"))},
        {"id": "PM", "label": "prune, then meta-train", **pr(paired(ds, "B_dacis_PM_30", "A2_dacis_P_30"))},
    ]
data["schedules"] = sch
data["episodic"] = {
    "resisc45": {"k5": pr(paired("resisc45", "dense_m", "dense_ce", "novel_5")), "k1": pr(paired("resisc45", "dense_m", "dense_ce", "novel_1"))},
    "eurosat13": {"k5": pr(paired("eurosat13", "dense_m", "dense_ce", "novel_5"))},
}

# cost
lat = json.load(open(os.path.join(REPO, "results_replication", "latency_named_cpu.json")))
cost = {"cpu": lat["cpu"], "rows": [{"key": r["key"], "name": r["name"], "params_m": r["params_m"], "macs_g": r["macs_g"],
                                    "cpu_ms": r["cpu1_ms"]} for r in lat["rows"]]}
dense = next(r for r in cost["rows"] if r["key"] == "dense")
for r in cost["rows"]:
    r["macs_red"] = 100 * (1 - r["macs_g"] / dense["macs_g"])
    r["cpu_red"] = 100 * (1 - r["cpu_ms"] / dense["cpu_ms"])
data["cost"] = cost

# accuracy against multiply-accumulates: every pruned network, plus the unpruned ones
import re

PRUNED = re.compile(r"^(A2|A|B|C|MP)_(random|l1|bn_gamma|taylor|dacis|G|V|D)_([A-Za-z0-9]+)_(\d+)$")
DENSE = {"dense_ce": ("r18", "ResNet-18, unpruned"), "MK_mobilenet_v2_dense": ("mbv2", "MobileNetV2, unpruned"),
         "K_mobilenet_v3_small": ("mbv3s", "MobileNetV3-Small, unpruned"), "K_efficientnet_b0": ("eff", "EfficientNet-B0, unpruned")}
trade = {}
for ds in DS:
    pts = []
    for rid, per in R[ds].items():
        s0 = next(iter(per.values()))
        v = np.array(list(acc(ds, rid).values()))
        base = {"id": rid, "n": int(len(v)), "m": float(v.mean()), "sd": float(v.std(ddof=1)),
                "macs": s0["eff"]["macs_res"] / 1e9, "params": s0["eff"]["params"] / 1e6}
        m = PRUNED.match(rid)
        if rid in DENSE:
            pts.append({**base, "kind": "dense", "family": DENSE[rid][0], "label": DENSE[rid][1], "frac": 100})
        elif m:
            tag, score, sched, frac = m.group(1), m.group(2), m.group(3), int(m.group(4))
            fam = "mbv2" if tag == "MP" else "r18"
            pts.append({**base, "kind": "pruned", "family": fam, "score": score, "sched": sched, "frac": frac,
                        "label": f"{'MobileNetV2' if fam == 'mbv2' else 'ResNet-18'}, {frac}%, {SC_LABEL[score]}, {sched}"})
    trade[ds] = sorted(pts, key=lambda p: p["macs"])
data["tradeoff"] = trade

# 13 bands against the three visible ones (the five partitions that have both)
a13, a3 = acc("eurosat13", "dense_ce"), acc("eurosat3", "dense_ce")
seeds = sorted(set(a13) & set(a3))
d = np.array([a13[s] - a3[s] for s in seeds])
t = stats.t.ppf(0.975, len(d) - 1)
se = d.std(ddof=1) / np.sqrt(len(d))
data["bands"] = {
    "seeds": seeds, "thirteen": [round(a13[s], 2) for s in seeds], "rgb": [round(a3[s], 2) for s in seeds],
    "m13": float(np.mean([a13[s] for s in seeds])), "m3": float(np.mean([a3[s] for s in seeds])),
    "diff": r1(d.mean()), "lo": r1(d.mean() - t * se), "hi": r1(d.mean() + t * se), "ahead": int((d > 0).sum()),
}

# Table 1: all eight scores at 30%, both schedules, four test settings, arrows against random
t1 = []
cols = [("resisc45", "novel_5", "RESISC45"), ("eurosat13", "novel_5", "EuroSAT-MS"), ("plantvillage", "novel_5", "PlantVillage"), ("plantvillage", "plantdoc_5", "PlantDoc")]
for sc in SC:
    row = {"id": sc, "label": SC_LABEL[sc], "cells": {}}
    for ds, key, nm in cols:
        for sched, pre in (("P", "A2"), ("PMP", "A")):
            rid = f"{pre}_{sc}_{sched if sched == 'PMP' else 'P'}_30"
            a = acc(ds, rid, key)
            if not a:
                row["cells"][f"{nm}|{sched}"] = None
                continue
            v = np.array(list(a.values()))
            cell = {"m": float(v.mean()), "sd": float(v.std(ddof=1)), "n": len(v), "sig": 0}
            if sc != "random":
                p = paired(ds, rid, f"{pre}_random_{'PMP' if sched == 'PMP' else 'P'}_30", key)
                cell["sig"] = 1 if p["lo"] > 0 else (-1 if p["hi"] < 0 else 0)
            row["cells"][f"{nm}|{sched}"] = cell
    t1.append(row)
denserow = {"label": "Dense ResNet-18 (unpruned)", "cells": {nm: ms(ds, "dense_ce", key) for ds, key, nm in cols}}
data["table1"] = {"rows": t1, "dense": denserow, "columns": [c[2] for c in cols]}

# Kendall-style headline counts used in the text
mb60 = paired("resisc45", "MP_taylor_P_60", "MP_random_P_60")
data["counts"] = {"taylor_vs_random_r18_30": race["r18_30"]["ahead"], "taylor_vs_random_mb40": race["mb_40"]["ahead"],
                  "taylor_vs_random_mb25": race["mb_25"]["ahead"], "taylor_vs_random_mb60": mb60["ahead"]}
assert mb60["ahead"] == 8 and mb60["n"] == 10, "the page says Taylor is ahead in 8 of 10 partitions at 60% of the MobileNetV2"


# ---------------------------------------------------------------------------------------------------------------- self-check
def chk(label, got, want):
    ok = abs(got - want) < 0.051
    print(("ok  " if ok else "FAIL"), f"{label:55s} got {got:+.2f} paper {want:+.1f}")
    if not ok:
        raise SystemExit(f"number does not match the paper: {label}")


f = {x["id"]: x for x in forest}
chk("Taylor, ResNet-18 30% PMP, RESISC45", f["r18_30"]["ds"]["resisc45"]["taylor"]["m"], 1.4)
chk("DACIS, ResNet-18 30% PMP, RESISC45", f["r18_30"]["ds"]["resisc45"]["dacis"]["m"], 0.0)
chk("Taylor, MobileNetV2 40%, RESISC45", f["mb_40"]["ds"]["resisc45"]["taylor"]["m"], 4.2)
chk("Taylor, MobileNetV2 25%, RESISC45", f["mb_25"]["ds"]["resisc45"]["taylor"]["m"], 6.5)
chk("BatchNorm scale, MobileNetV2 25%, RESISC45", f["mb_25"]["ds"]["resisc45"]["bn_gamma"]["m"], -8.3)
chk("Taylor, MobileNetV2 25%, EuroSAT-MS", f["mb_25"]["ds"]["eurosat13"]["taylor"]["m"], 3.3)
chk("Taylor at 10% of ResNet-18", lead["resisc45"]["r18"]["scores"]["taylor"][3]["m"], 6.0)
chk("Taylor at 20% of ResNet-18", lead["resisc45"]["r18"]["scores"]["taylor"][2]["m"], 2.5)
chk("Taylor at 50% of ResNet-18", lead["resisc45"]["r18"]["scores"]["taylor"][0]["m"], 0.6)
chk("Taylor at 60% of MobileNetV2", lead["resisc45"]["mbv2"]["scores"]["taylor"][0]["m"], 0.8)
chk("PMP against P, RESISC45", sch["resisc45"][0]["m"], 0.9)
chk("probe (no weight update) against P", sch["resisc45"][2]["m"], 1.2)
chk("prune then meta-train against P", sch["resisc45"][3]["m"], 0.0)
chk("episodic stage, dense, 5-shot, RESISC45", data["episodic"]["resisc45"]["k5"]["m"], -0.3)
chk("episodic stage, dense, 1-shot, RESISC45", data["episodic"]["resisc45"]["k1"]["m"], 0.6)
chk("13 bands against RGB, dense, 5-shot", data["bands"]["diff"], -6.6)
cr = {r["key"]: r for r in cost["rows"]}
assert round(cr["r18_30"]["macs_red"]) == 58, "the paper says 58% fewer multiply-accumulates at 30%"
print("MAC reduction at 30%%: %.1f%%" % cr["r18_30"]["macs_red"])
print("latency reduction at 30%% on the laptop CPU: %.1f%%" % cr["r18_30"]["cpu_red"])
assert round(cr["r18_30"]["cpu_red"]) == 39
print("Taylor ahead of random in", race["r18_30"]["ahead"], "of 10 (ResNet-18 30%),", race["mb_40"]["ahead"], "of 10 (MobileNetV2 40%),", race["mb_25"]["ahead"], "of 10 (MobileNetV2 25%)")
print("13 bands vs RGB:", data["bands"]["m13"], data["bands"]["m3"], data["bands"]["diff"], data["bands"]["lo"], data["bands"]["hi"], "ahead in", data["bands"]["ahead"], "of", len(data["bands"]["seeds"]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("/* Generated by tools/make_data.py from the saved runs. Do not edit by hand. */\n")
    fh.write("window.SITE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
print("wrote", OUT, os.path.getsize(OUT), "bytes")
