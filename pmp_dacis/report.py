"""Aggregation, paired statistics, LaTeX tables, figures and the numbers file used by the paper.

Every number that appears in the paper is read from the saved runs by this module. Seeds (class partitions) are the
replicates: tables give mean and sample standard deviation over seeds, paired differences give a t-interval over the
per-seed differences (episodes are shared between methods inside a seed).
"""
import glob
import json
import os

import numpy as np
from scipy import stats as sps

CRIT_LABEL = {
    "random": "Random", "l1": r"Filter $\ell_1$", "bn_gamma": r"BN scale $|\gamma|$", "taylor": "Taylor",
    "G": r"Gradient $\mathcal{G}$", "V": r"Variance $\mathcal{V}$", "D": r"Fisher $\mathcal{D}$",
    "GV": r"$\mathcal{G}{+}\mathcal{V}$", "GD": r"$\mathcal{G}{+}\mathcal{D}$", "VD": r"$\mathcal{V}{+}\mathcal{D}$",
    "dacis": "DACIS",
}
CRIT_PLAIN = {"random": "random", "l1": "filter L1", "bn_gamma": "BN scale", "taylor": "Taylor", "G": "gradient",
              "V": "variance", "D": "Fisher", "dacis": "DACIS"}
SCORES = ["random", "l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"]
RS, ES13, ES3, PV = "resisc45", "eurosat13", "eurosat3", "plantvillage"
VARIANTS = (RS, ES13, ES3, PV)
VTAG = {RS: "rs", ES13: "es", ES3: "e3", PV: "pv"}
# (variant, metric, column header): the four test settings of the paper, two of them Earth observation
COLS = [(RS, "novel_5", "RESISC45"), (ES13, "novel_5", "EuroSAT-MS"), (PV, "novel_5", "PlantVillage"), (PV, "plantdoc_5", "PlantDoc")]
TITLE = {RS: "NWPU-RESISC45 (aerial scenes)", ES13: "EuroSAT-MS (Sentinel-2, 13 bands)", ES3: "EuroSAT-MS (RGB bands only)",
         PV: "PlantVillage (leaf photographs) and PlantDoc"}


def ctag(v, k):
    return VTAG[v] if k.startswith("novel") else "pd"


class Results:
    def __init__(self, root):
        self.root = root
        self.runs = {}
        for p in glob.glob(os.path.join(root, "runs", "*", "seed*", "*.json")):
            r = json.load(open(p))
            self.runs[(r["variant"], r["seed"], r["spec"]["id"])] = r
        self._npz = {}
        self.host = json.load(open(os.path.join(root, "host.json"))) if os.path.exists(os.path.join(root, "host.json")) else {}
        tp = os.path.join(root, "tuning_m_lr.json")
        self.tuning = json.load(open(tp)) if os.path.exists(tp) else None
        tp = os.path.join(root, "tuning_m_lr_resisc45.json")
        self.tuning_rs = json.load(open(tp)) if os.path.exists(tp) else None
        tp = os.path.join(root, "tuning_m_lr_resisc45_low.json")
        self.tuning_low = json.load(open(tp)) if os.path.exists(tp) else None

    # ---- access ---------------------------------------------------------------------------
    def seeds(self, variant):
        return sorted({k[1] for k in self.runs if k[0] == variant})

    def has(self, variant, id_):
        ss = [s for s in self.seeds(variant) if s < 5]            # partitions 5-9 exist only for some runs
        return bool(ss) and all((variant, s, id_) in self.runs for s in ss)

    def seeds_with(self, variant, *ids):
        return [s for s in self.seeds(variant) if all((variant, s, i) in self.runs for i in ids)]

    def pid(self, variant, c):
        """Run id of score c under the plain prune-and-recover schedule (two naming generations)."""
        for cand in (f"A2_{c}_P_30", f"B_{c}_P_30"):
            if self.has(variant, cand):
                return cand
        return None

    def vals(self, variant, id_, key, getter=None):
        out = []
        for s in self.seeds_with(variant, id_):
            r = self.runs[(variant, s, id_)]
            try:
                out.append(getter(r) if getter else r["acc"][key]["mean"])
            except KeyError:
                continue
        return np.array(out, dtype=float)

    def ms(self, variant, id_, key, getter=None):
        if id_ is None:
            return None
        v = self.vals(variant, id_, key, getter)
        if len(v) == 0:
            return None
        return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

    def eff(self, variant, id_, field, sub=None):
        if id_ is None:
            return None
        out = []
        for s in self.seeds_with(variant, id_):
            v = self.runs[(variant, s, id_)]["eff"].get(field)
            if v is None:
                continue
            out.append(v[sub] if sub else v)
        return float(np.mean(out)) if out else None

    def arr(self, variant, seed, id_, key):
        p = os.path.join(self.root, "runs", variant, f"seed{seed}", f"{id_}.npz")
        if p not in self._npz:
            self._npz[p] = np.load(p)
        return self._npz[p][key].astype(np.float64)

    def paired(self, variant, a, b, key):
        """a - b in accuracy points. Replicates are seeds: mean of the per-seed differences with a 95% t-interval.
        `wins` counts seeds in which a beats b."""
        if a is None or b is None:
            return None
        seeds = self.seeds_with(variant, a, b)
        if len(seeds) < 2:
            return None
        try:
            d = np.array([self.runs[(variant, s, a)]["acc"][key]["mean"] - self.runs[(variant, s, b)]["acc"][key]["mean"]
                          for s in seeds])
        except KeyError:
            return None
        n = len(d)
        sd = float(d.std(ddof=1))
        half = float(sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        pval = float(2 * sps.t.sf(abs(d.mean()) / (sd / np.sqrt(n)), n - 1)) if sd > 0 else 1.0
        return {"mean": float(d.mean()), "lo": float(d.mean() - half), "hi": float(d.mean() + half), "sd": sd, "p": pval,
                "per_seed": d.tolist(), "wins": int((d > 0).sum()), "n": n}

    def cost_variant(self):
        """Dataset whose measured cost is quoted (RGB, 128 x 128 input, the same for RESISC45 and PlantVillage)."""
        for v in (RS, PV):
            if self.has(v, "dense_ce"):
                return v
        return None


# ---- formatting ---------------------------------------------------------------------------------
def pm(ms, d=1, bold=False):
    if ms is None:
        return "--"
    m, s = ms
    txt = f"{m:.{d}f}"
    if bold:
        txt = r"\mathbf{" + txt + "}"
    return f"${txt}" + r"_{\pm" + f"{s:.{d}f}" + "}$"


def sgn(x, d=1):
    return f"{x:+.{d}f}".replace("-", r"\ensuremath{-}")


def pd_cell(p, d=1):
    if p is None:
        return "--"
    return f"${sgn(p['mean'], d)}$ " + r"{\scriptsize[" + f"{sgn(p['lo'], d)},{sgn(p['hi'], d)}" + "]}"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def mparams(R, variant, id_):
    p = R.eff(variant, id_, "params")
    return None if p is None else p / 1e6


# ---- tables -------------------------------------------------------------------------------------
def table_scores(R):
    """Eight channel scores at 30% of the parameters under two schedules (P: prune then recover; PMP)."""
    n = len(COLS)
    L = [r"\begin{tabular}{@{}l" + "cc" * n + r"@{}}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{2}{c}{%s}" % h for _, _, h in COLS) + r" \\",
         "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(n)),
         r"Score at 30\% of the parameters & " + " & ".join("P & PMP" for _ in COLS) + r" \\", r"\midrule"]
    for lab, id_ in (("Dense ResNet-18, S0 only", "dense_ce"), ("Dense ResNet-18, S0+R+M+R", "dense_rmr")):
        L.append(lab + " & " + " & ".join(r"\multicolumn{2}{c}{%s}" % pm(R.ms(v, id_, k)) for v, k, _ in COLS) + r" \\")
    L.append(r"\midrule")
    def mark(v, k, a, b):
        p = R.paired(v, a, b, k) if c != "random" else None
        return "" if p is None else (r"$^{\uparrow}$" if p["lo"] > 0 else (r"$^{\downarrow}$" if p["hi"] < 0 else ""))

    for c in SCORES:
        cells = []
        for v, k, _ in COLS:
            cells.append(pm(R.ms(v, R.pid(v, c), k)) + mark(v, k, R.pid(v, c), R.pid(v, "random")))
            cells.append(pm(R.ms(v, f"A_{c}_PMP_30", k)) + mark(v, k, f"A_{c}_PMP_30", "A_random_PMP_30"))
        L.append(CRIT_LABEL[c] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_pairs(R):
    """Paired differences (points) between choices, mean over seeds with 95% t-interval over seeds."""
    def row(label, a, b):
        cells = []
        for v, k, _ in COLS:
            ia, ib = (a(v) if callable(a) else a), (b(v) if callable(b) else b)
            cells.append(pd_cell(R.paired(v, ia, ib, k)))
        return label + " & " + " & ".join(cells) + r" \\"

    P = lambda c: (lambda v: R.pid(v, c))
    PMP = lambda c: f"A_{c}_PMP_30"
    n = len(COLS) + 1
    L = [r"\begin{tabular}{@{}l" + "c" * len(COLS) + r"@{}}", r"\toprule",
         r"Difference (a $-$ b), points & " + " & ".join(h for _, _, h in COLS) + r" \\", r"\midrule",
         r"\multicolumn{%d}{@{}l}{\emph{Scores, schedule P}} \\" % n]
    for c in ("random", "l1", "taylor"):
        L.append(row(f"DACIS $-$ {CRIT_LABEL[c]}", P("dacis"), P(c)))
    L.append(r"\multicolumn{%d}{@{}l}{\emph{Scores, schedule PMP}} \\" % n)
    for c in ("random", "l1", "taylor"):
        L.append(row(f"DACIS $-$ {CRIT_LABEL[c]}", PMP("dacis"), PMP(c)))
    L.append(r"\multicolumn{%d}{@{}l}{\emph{Schedules, DACIS at 30\%%}} \\" % n)
    L.append(row(r"PMP $-$ P", PMP("dacis"), P("dacis")))
    L.append(row(r"P$\to$M $-$ P", "B_dacis_PM_30", P("dacis")))
    L.append(row(r"M$\to$P $-$ P", "B_dacis_MP_30", P("dacis")))
    L.append(row(r"P$\to$G$\to$P (no weight update) $-$ P", "B_dacis_PGP_30", P("dacis")))
    L.append(row(r"P$\to$M$\to$P$\to$M $-$ PMP", "B_dacis_PMPM_30", PMP("dacis")))
    L.append(row(r"P$\to$M ($2\times$ episodes) $-$ P$\to$M", "B_dacis_PM2x_30", "B_dacis_PM_30"))
    L.append(r"\multicolumn{%d}{@{}l}{\emph{Episodic stage on the dense network}} \\" % n)
    L.append(row("Dense+M $-$ Dense", "dense_m", "dense_ce"))
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_sched(R):
    rows = [("P", lambda v: R.pid(v, "dacis")), (r"P$\to$M", lambda v: "B_dacis_PM_30"),
            (r"P$\to$M ($2\times$)", lambda v: "B_dacis_PM2x_30"), (r"M$\to$P", lambda v: "B_dacis_MP_30"),
            (r"P$\to$G$\to$P", lambda v: "B_dacis_PGP_30"), (r"P$\to$M$\to$P (PMP)", lambda v: "A_dacis_PMP_30"),
            (r"P$\to$M$\to$P$\to$M", lambda v: "B_dacis_PMPM_30")]
    L = [r"\begin{tabular}{@{}l" + "r" * len(COLS) + r"@{}}", r"\toprule",
         r"Schedule (DACIS, 30\%), 5-shot & " + " & ".join(h for _, _, h in COLS) + r" \\", r"\midrule"]
    for lab, f in rows:
        L.append(f"{lab} & " + " & ".join(pm(R.ms(v, f(v), k)) for v, k, _ in COLS) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_budget(R):
    """Cost next to accuracy by budget. The pruned architecture is the same for every score, so cost is per budget."""
    cv = R.cost_variant()
    L = [r"\begin{tabular}{@{}lrrrr" + "r" * 3 + r"@{}}", r"\toprule",
         r"Model & Params (M) & MACs (G) & CPU ms & GPU img/s & RESISC45 & EuroSAT-MS & PlantVillage \\", r"\midrule"]
    if cv is None:
        return "% no cost measurements available\n"

    def acc(label_id, v):
        return pm(R.ms(v, label_id, "novel_5")) if label_id and R.has(v, label_id) else "--"

    def line(label, acc_id, cost_id=None):
        cost_id = cost_id or acc_id
        if not R.has(cv, cost_id):
            return None
        mp = mparams(R, cv, cost_id)
        macs = R.eff(cv, cost_id, "macs_128")
        cpu = R.eff(cv, cost_id, "cpu_1thread_b1_ms", "median")
        gpu = R.eff(cv, cost_id, "gpu_b128_img_per_s")
        f = lambda x, fmt: "--" if x is None else fmt.format(x)
        return (f"{label} & {f(mp, '{:.2f}')} & {f(None if macs is None else macs / 1e9, '{:.3f}')} & {f(cpu, '{:.1f}')} & "
                f"{f(gpu, '{:.0f}')} & " + " & ".join(acc(acc_id, v) for v in (RS, ES13, PV)) + r" \\")

    items = [line("ResNet-18, dense", "dense_ce")]
    for frac in (50, 30, 20, 10):
        cid = "A_l1_PMP_30" if frac == 30 else f"C_l1_PMP_{frac}"
        aid = "A_dacis_PMP_30" if frac == 30 else f"C_dacis_PMP_{frac}"
        items.append(line(f"ResNet-18, {frac}\\% (DACIS)", aid, cid))
    items.append(r"\midrule")
    for arch, lab in (("mobilenet_v3_small", "MobileNetV3-Small"), ("mobilenet_v2", "MobileNetV2"), ("efficientnet_b0", "EfficientNet-B0")):
        items.append(line(lab + ", dense", f"K_{arch}"))
    L += [i for i in items if i]
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_budget_scores(R):
    """5-shot accuracy per score at each budget (PMP schedule), one block per test setting."""
    crits = ("random", "l1", "taylor", "dacis")
    L = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Params kept & Random & Filter $\ell_1$ & Taylor & DACIS \\"]
    for v, k, h in COLS:
        L.append(r"\midrule")
        L.append(r"\multicolumn{5}{@{}l}{\emph{%s}} \\" % h)
        for frac in (50, 30, 20, 10):
            cells = []
            for c in crits:
                iid = f"A_{c}_PMP_30" if frac == 30 else f"C_{c}_PMP_{frac}"
                cells.append(pm(R.ms(v, iid, k)))
            L.append(f"{frac}\\% & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_recovery(R):
    L = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"PlantVillage 5-shot, schedule P, 30\% & 1 epoch & 3 epochs & 6 epochs \\", r"\midrule"]
    for c in ("random", "l1", "taylor", "dacis"):
        ids = [R.pid(PV, c), f"R_{c}_P_30_r3", f"R_{c}_P_30_r6"]
        L.append(f"{CRIT_LABEL[c]} & " + " & ".join(pm(R.ms(PV, i, "novel_5")) for i in ids) + r" \\")
    L.append(r"\midrule")
    L.append(r"Dense, no pruning & \multicolumn{3}{c}{" + pm(R.ms(PV, "dense_ce", "novel_5")) + r"} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_ablation(R):
    items = [("DACIS (full)", "A_dacis_PMP_30"),
             (r"$\mathcal{G}$ only", "A_G_PMP_30"), (r"$\mathcal{V}$ only", "A_V_PMP_30"), (r"$\mathcal{D}$ only", "A_D_PMP_30"),
             (r"$\mathcal{G}{+}\mathcal{V}$", "A_GV_PMP_30"), (r"$\mathcal{G}{+}\mathcal{D}$", "A_GD_PMP_30"),
             (r"$\mathcal{V}{+}\mathcal{D}$", "A_VD_PMP_30"),
             ("no meta-gradient term", "E_dacis_PMP_30_norefine"), ("same width at every depth", "E_dacis_PMP_30_uniform"),
             ("min-max instead of ranks", "E_dacis_PMP_30_minmax")]
    L = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"DACIS variant (PMP, 30\%) & PV 5-shot & $\Delta$ vs. full \\", r"\midrule"]
    for lab, id_ in items:
        if not R.has(PV, id_):
            continue
        d = R.paired(PV, id_, "A_dacis_PMP_30", "novel_5") if id_ != "A_dacis_PMP_30" else None
        L.append(f"{lab} & {pm(R.ms(PV, id_, 'novel_5'))} & {pd_cell(d) if d else '--'}" + r" \\")
        if id_ in ("A_VD_PMP_30",):
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_eurosat(R):
    rows = [("13 bands, dense (S0)", ES13, "dense_ce"), ("RGB bands, dense (S0)", ES3, "dense_ce"),
            ("13 bands, dense (S0+R+M+R)", ES13, "dense_rmr"), ("RGB bands, dense (S0+R+M+R)", ES3, "dense_rmr"),
            (r"13 bands, filter $\ell_1$, PMP 30\%", ES13, "A_l1_PMP_30"), (r"RGB bands, filter $\ell_1$, PMP 30\%", ES3, "A_l1_PMP_30"),
            (r"13 bands, DACIS, PMP 30\%", ES13, "A_dacis_PMP_30"), (r"RGB bands, DACIS, PMP 30\%", ES3, "A_dacis_PMP_30"),
            ("13 bands, MobileNetV3-Small", ES13, "K_mobilenet_v3_small"), ("RGB bands, MobileNetV3-Small", ES3, "K_mobilenet_v3_small")]
    L = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"EuroSAT-MS, novel classes & 1-shot & 5-shot & 10-shot \\", r"\midrule"]
    for lab, v, i in rows:
        if R.has(v, i):
            L.append(f"{lab} & " + " & ".join(pm(R.ms(v, i, k)) for k in ("novel_1", "novel_5", "novel_10")) + r" \\")
    L.append(r"\midrule")
    for i, lab in (("dense_ce", r"13 bands $-$ RGB, dense"), ("A_dacis_PMP_30", r"13 bands $-$ RGB, DACIS PMP")):
        if R.has(ES13, i) and R.has(ES3, i):
            diffs = [R.runs[(ES13, s, i)]["acc"]["novel_5"]["mean"] - R.runs[(ES3, s, i)]["acc"]["novel_5"]["mean"]
                     for s in R.seeds_with(ES13, i) if (ES3, s, i) in R.runs]
            d = np.array(diffs)
            if len(d) > 1:
                half = sps.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                L.append(f"{lab} (5-shot) & \\multicolumn{{3}}{{c}}{{${sgn(d.mean())}$ {{\\scriptsize[{sgn(d.mean() - half)},{sgn(d.mean() + half)}]}}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_shots(R):
    """1-, 5- and 10-shot accuracy of the main models on the three novel-class sets."""
    rows = [("Dense ResNet-18", "dense_ce"), ("Random, PMP 30\\%", "A_random_PMP_30"), (r"Filter $\ell_1$, PMP 30\%", "A_l1_PMP_30"),
            ("Taylor, PMP 30\\%", "A_taylor_PMP_30"), ("DACIS, PMP 30\\%", "A_dacis_PMP_30"),
            ("MobileNetV3-Small", "K_mobilenet_v3_small")]
    sets = [(RS, "RESISC45"), (ES13, "EuroSAT-MS"), (PV, "PlantVillage")]
    L = [r"\begin{tabular}{@{}l" + "rrr" * len(sets) + r"@{}}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{3}{c}{%s}" % h for _, h in sets) + r" \\",
         "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(sets))),
         "Model & " + " & ".join("1-shot & 5-shot & 10-shot" for _ in sets) + r" \\", r"\midrule"]
    for lab, id_ in rows:
        cells = []
        for v, _ in sets:
            cells += [pm(R.ms(v, id_, k)) if R.has(v, id_) else "--" for k in ("novel_1", "novel_5", "novel_10")]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_mbv2(R):
    """MobileNetV2 pruned with the same scores (schedule P): 5-shot accuracy by budget."""
    fr = (60, 40, 25)
    sets = [(RS, "RESISC45"), (ES13, "EuroSAT-MS")]
    L = [r"\begin{tabular}{@{}l" + "r" * (len(fr) * len(sets)) + r"@{}}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{%d}{c}{%s}" % (len(fr), h) for _, h in sets) + r" \\",
         "".join(r"\cmidrule(lr){%d-%d}" % (2 + len(fr) * i, 1 + len(fr) * (i + 1)) for i in range(len(sets))),
         "Parameters kept & " + " & ".join(f"{f}\\%" for _ in sets for f in fr) + r" \\", r"\midrule"]
    L.append("MobileNetV2, dense & " + " & ".join(r"\multicolumn{%d}{c}{%s}" % (len(fr), pm(R.ms(v, "MK_mobilenet_v2_dense", "novel_5")))
                                              for v, _ in sets) + r" \\")
    L.append(r"\midrule")
    for c in ("random", "l1", "bn_gamma", "taylor", "dacis"):
        cells = [pm(R.ms(v, f"MP_{c}_P_{f}", "novel_5")) for v, _ in sets for f in fr]
        L.append(CRIT_LABEL[c] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


LAMBDAS = [(0.3, 0.2, 0.5), (1 / 3, 1 / 3, 1 / 3), (0.5, 0.2, 0.3), (0.2, 0.5, 0.3), (0.2, 0.2, 0.6), (0.1, 0.1, 0.8),
           (0.0, 0.0, 1.0), (0.5, 0.5, 0.0)]


def table_lambda(R):
    L = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"$(\lambda_G,\lambda_V,\lambda_D)$ & Val 1-shot & Val 5-shot & Novel 1-shot & Novel 5-shot \\", r"\midrule"]
    for i, lam in enumerate(LAMBDAS):
        id_ = f"S_dacis_PMP_30_lam{i}"
        if not R.has(PV, id_):
            continue
        txt = "(" + ",".join(f"{x:.2f}".rstrip("0").rstrip(".") if x not in (0.0, 1.0) else str(int(x)) for x in lam) + ")"
        L.append(f"{txt} & " + " & ".join(pm(R.ms(PV, id_, k)) for k in ("val_1", "val_5", "novel_1", "novel_5")) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def table_mlr(R):
    if not R.tuning:
        return ""
    t2 = R.tuning_rs["rows"] if R.tuning_rs else {}
    L = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"Episodic learning rate & \multicolumn{2}{c}{PlantVillage val.} & \multicolumn{2}{c}{RESISC45 val.} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & 1-shot & 5-shot & 1-shot & 5-shot \\", r"\midrule"]
    for k, v in R.tuning["rows"].items():
        name = "none (S0 only)" if k.startswith("S0") else k
        w = t2.get(k)
        extra = f"{w['1']:.1f} & {w['5']:.1f}" if w else "-- & --"
        L.append(f"{name} & {v['1']:.1f} & {v['5']:.1f} & {extra}" + r" \\")
    if R.tuning_low:
        L.append(r"\midrule")
        for k, v in R.tuning_low["rows"].items():
            if k.startswith("S0") or k in R.tuning["rows"]:
                continue
            L.append(f"{k} (extra check) & -- & -- & {v['1']:.1f} & {v['5']:.1f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ---- analysis helpers -------------------------------------------------------------------------
def jaccard_matrix(R, variant=PV, crits=("random", "l1", "bn_gamma", "taylor", "G", "V", "D", "dacis")):
    """Mean Jaccard overlap of the channels kept by the first prune stage of the PMP runs (budget 60%)."""
    M = np.full((len(crits), len(crits)), np.nan)
    for i, a in enumerate(crits):
        for j, b in enumerate(crits):
            vals = []
            for s in R.seeds(variant):
                ra, rb = R.runs.get((variant, s, f"A_{a}_PMP_30")), R.runs.get((variant, s, f"A_{b}_PMP_30"))
                if not ra or not rb:
                    continue
                ka = {(bi, c) for bi, ch in enumerate(ra["stages"][0]["keep"]) for c in ch}
                kb = {(bi, c) for bi, ch in enumerate(rb["stages"][0]["keep"]) for c in ch}
                vals.append(len(ka & kb) / len(ka | kb))
            if vals:
                M[i, j] = np.mean(vals)
    return M, list(crits)


def score_matrix(R, variant, key, sched):
    """Partitions x scores accuracy matrix (only partitions where every score finished), for the rank statistics."""
    ids = [R.pid(variant, c) if sched == "P" else f"A_{c}_PMP_30" for c in SCORES]
    ids_ok = [(c, i) for c, i in zip(SCORES, ids) if i is not None and R.has(variant, i)]
    if len(ids_ok) < 3:
        return None, []
    seeds = R.seeds_with(variant, *[i for _, i in ids_ok])
    if len(seeds) < 2:
        return None, []
    M = np.array([[R.runs[(variant, s, i)]["acc"][key]["mean"] for _, i in ids_ok] for s in seeds])
    return M, [c for c, _ in ids_ok]


def kendall_w(M):
    """Kendall's coefficient of concordance of the score ranking across partitions (rows) and its chi-square p value."""
    m, n = M.shape
    ranks = np.vstack([sps.rankdata(row) for row in M])
    rsum = ranks.sum(0)
    S = ((rsum - rsum.mean()) ** 2).sum()
    W = 12 * S / (m ** 2 * (n ** 3 - n))
    chi2 = m * (n - 1) * W
    return float(W), float(sps.chi2.sf(chi2, n - 1))


# ---- figures ----------------------------------------------------------------------------------
def make_figures(R, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    os.makedirs(outdir, exist_ok=True)
    seed_cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300"]
    seed_marks = ["o", "s", "^", "D", "v", "P"]
    panels = [(v, k, h) for v, k, h in COLS if R.seeds(v)]
    if not panels:
        return
    # (1) per-seed accuracy of each score: the ranking changes with the class partition
    fig, axs = plt.subplots(1, len(panels), figsize=(6.9, 2.0), squeeze=False)
    axs = axs[0]
    order = ["random", "l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"]
    short = {"random": "Rnd", "l1": "L1", "bn_gamma": "BN", "taylor": "Tay", "G": "G", "V": "V", "D": "D", "dacis": "DAC"}
    for ax, (var, key, title) in zip(axs, panels):
        for si, s in enumerate(R.seeds(var)):
            ys = np.array([R.runs[(var, s, f"A_{c}_PMP_30")]["acc"][key]["mean"] if (var, s, f"A_{c}_PMP_30") in R.runs else np.nan
                           for c in order])
            if np.all(np.isnan(ys)):
                continue
            ax.plot(np.arange(len(order)) + (si - 2) * 0.1, ys - np.nanmean(ys), ls="none", marker=seed_marks[si % len(seed_marks)],
                    ms=3.2, color=seed_cols[si % len(seed_cols)], label=f"partition {s}")
        ax.axhline(0, color="#999", lw=0.6)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([short[c] for c in order], rotation=45)
        ax.set_title(title)
    axs[0].set_ylabel("Accuracy minus score mean (pts)")
    hd, lb = axs[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="upper center", ncol=len(lb), frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(outdir, "fig_seeds.pdf"))
    plt.close(fig)
    # (2) accuracy versus parameter budget
    col = {"dacis": "#eb6834", "l1": "#2a78d6", "random": "#7f7f7f", "taylor": "#1baf7a"}
    mk = {"dacis": "o", "l1": "s", "random": "x", "taylor": "^"}
    lab = {"dacis": "DACIS", "l1": r"Filter $\ell_1$", "random": "Random", "taylor": "Taylor"}
    fig, axs = plt.subplots(1, len(panels), figsize=(6.9, 1.95), squeeze=False)
    axs = axs[0]
    for ax, (var, key, title) in zip(axs, panels):
        for c in ("random", "l1", "taylor", "dacis"):
            xs, ys, es = [], [], []
            for frac in (10, 20, 30, 50):
                iid = f"A_{c}_PMP_30" if frac == 30 else f"C_{c}_PMP_{frac}"
                if R.has(var, iid):
                    m, s = R.ms(var, iid, key)
                    xs.append(frac); ys.append(m); es.append(s / np.sqrt(len(R.vals(var, iid, key))))
            if xs:                                           # small horizontal offsets keep the four error bars apart
                off = {"random": -1.0, "l1": -0.35, "taylor": 0.35, "dacis": 1.0}[c]
                ax.errorbar([x + off for x in xs], ys, yerr=es, marker=mk[c], ms=3.5, lw=1.0, capsize=1.5, color=col[c], label=lab[c])
        if R.has(var, "dense_ce"):
            ax.axhline(R.ms(var, "dense_ce", key)[0], ls="--", lw=0.8, color="#333", label="Dense")
        ax.set_title(title)
        ax.set_xlabel("Parameters kept (%)")
        ax.set_xticks([10, 20, 30, 50])
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].legend(loc="lower right", frameon=False, fontsize=5.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(outdir, "fig_budget.pdf"))
    plt.close(fig)
    # (2b) accuracy against cost: the compact networks sit up and to the left of every pruned ResNet-18
    sets = [(v, k, h) for v, k, h in panels if k == "novel_5"]
    fig, axs = plt.subplots(1, len(sets), figsize=(6.9, 2.0), squeeze=False)
    for ax, (var, key, title) in zip(axs[0], sets):
        def pt(id_):
            if not R.has(var, id_):
                return None
            m = R.eff(var, id_, "macs_res")
            return None if m is None else (m / 1e9, R.ms(var, id_, key)[0])
        dense = pt("dense_ce")
        if dense:
            ax.plot(*dense, marker="*", ms=8, ls="none", color="#333", label="ResNet-18, dense")
        pr = [pt("A_dacis_PMP_30" if f == 30 else f"C_dacis_PMP_{f}") for f in (50, 30, 20, 10)]
        pr = [q for q in pr if q]
        if pr:
            ax.plot([q[0] for q in pr], [q[1] for q in pr], marker="o", ms=3.5, lw=1.0, color="#eb6834", label="ResNet-18 pruned (DACIS, 50/30/20/10%)")
        pm2 = [pt(i) for i in ("MP_dacis_P_60", "MP_dacis_P_40", "MP_dacis_P_25")]
        pm2 = [q for q in pm2 if q]
        if pm2:
            ax.plot([q[0] for q in pm2], [q[1] for q in pm2], marker="^", ms=3.5, lw=1.0, ls="--", color="#1baf7a",
                    label="MobileNetV2 pruned (DACIS, 60/40/25%)")
        for arch, lab, mk2, c in (("mobilenet_v3_small", "MobileNetV3-S", "s", "#2a78d6"), ("mobilenet_v2", "MobileNetV2", "v", "#8a8a8a"),
                                  ("efficientnet_b0", "EfficientNet-B0", "D", "#6250d6")):
            q = pt(f"K_{arch}")
            if q:
                ax.plot(*q, marker=mk2, ms=4.5, ls="none", color=c, label=lab)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Multiply-accumulates (G)")
        ax.grid(alpha=0.25, lw=0.4)
    axs[0][0].set_ylabel("5-shot accuracy (%)")
    hd, lb = axs[0][0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="upper center", ncol=len(lb), frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(outdir, "fig_tradeoff.pdf"))
    plt.close(fig)
    # (3) resolution sweep (trained at one resolution, tested at others)
    rv = next((v for v in (RS, PV) if R.has(v, "A_dacis_PMP_30") and
               any("res_sweep_5shot" in R.runs[(v, s, "A_dacis_PMP_30")]["acc"] for s in R.seeds(v))), None)
    if rv:
        fig, ax = plt.subplots(figsize=(3.2, 1.9))
        for id_, lb, c in (("dense_ce", "Dense", "#333"), ("A_l1_PMP_30", r"Filter $\ell_1$, 30%", "#2a78d6"), ("A_dacis_PMP_30", "DACIS, 30%", "#eb6834")):
            if not R.has(rv, id_):
                continue
            res = sorted({int(k) for s in R.seeds(rv) for k in R.runs[(rv, s, id_)]["acc"].get("res_sweep_5shot", {})})
            if not res:
                continue
            ys = [np.mean([R.runs[(rv, s, id_)]["acc"]["res_sweep_5shot"][str(r)] for s in R.seeds(rv)]) for r in res]
            ax.plot(res, ys, marker="o", ms=2.5, lw=1.0, color=c, label=lb)
        ax.set_xlabel("Test resolution (px; trained at 128)")
        ax.set_ylabel("5-shot accuracy (%)")
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(frameon=False)
        fig.tight_layout(pad=0.4)
        fig.savefig(os.path.join(outdir, "fig_res.pdf"))
        plt.close(fig)
    # (4) overlap of kept channels
    ov = next((v for v in (RS, PV) if R.seeds(v)), None)
    if ov:
        M, names = jaccard_matrix(R, ov)
        if not np.all(np.isnan(M)):
            fig, ax = plt.subplots(figsize=(3.2, 2.7))
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
            lb = [short[n] for n in names]
            ax.set_xticks(range(len(lb)))
            ax.set_xticklabels(lb, rotation=45, ha="right")
            ax.set_yticks(range(len(lb)))
            ax.set_yticklabels(lb)
            for i in range(len(lb)):
                for j in range(len(lb)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if M[i, j] < 0.6 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            fig.tight_layout(pad=0.4)
            fig.savefig(os.path.join(outdir, "fig_overlap.pdf"))
            plt.close(fig)


# ---- numbers used in the text -------------------------------------------------------------------
def numbers(R):
    """Every number the prose quotes, keyed by name. \\num{key} in the LaTeX reads them."""
    n = {}
    for var in VARIANTS:
        ids = sorted({k[2] for k in R.runs if k[0] == var})
        n[f"{var}:seeds"] = str(len(R.seeds(var)))
        for id_ in ids:
            if not R.has(var, id_):
                continue
            some = R.runs[(var, R.seeds(var)[0], id_)]["acc"]
            for key in some:
                if isinstance(some[key], dict) and "mean" in some[key]:
                    ms = R.ms(var, id_, key)
                    n[f"{var}:{id_}:{key}"] = f"{ms[0]:.1f}"
                    n[f"{var}:{id_}:{key}:sd"] = f"{ms[1]:.1f}"
                    v = R.vals(var, id_, key)
                    n[f"{var}:{id_}:{key}:min"] = f"{v.min():.1f}"
                    n[f"{var}:{id_}:{key}:max"] = f"{v.max():.1f}"
            pf = R.vals(var, id_, None, getter=lambda r: r["param_frac"] * 100).mean()
            n[f"{var}:{id_}:pct"] = f"{pf:.1f}"
            for fld in ("params", "macs_128", "macs_224"):
                v = R.eff(var, id_, fld)
                if v is not None:
                    n[f"{var}:{id_}:{fld}"] = f"{v / 1e6:.2f}" if fld == "params" else f"{v / 1e9:.3f}"
            for fld, sub in (("cpu_1thread_b1_ms", "median"), ("cpu_all_threads_b1_ms", "median"), ("gpu_b1_ms", "median")):
                v = R.eff(var, id_, fld, sub)
                if v is not None:
                    n[f"{var}:{id_}:{fld}"] = f"{v:.1f}"
            v = R.eff(var, id_, "gpu_b128_img_per_s")
            if v is not None:
                n[f"{var}:{id_}:gpu_ips"] = f"{v:.0f}"
    # aliases for the plain prune-and-recover runs (two naming generations): <var>:P_<score>:<field>
    for var in VARIANTS:
        for c in SCORES:
            pid = R.pid(var, c)
            if pid:
                for k in [k for k in n if k.startswith(f"{var}:{pid}:")]:
                    n[f"{var}:P_{c}:" + k[len(f"{var}:{pid}:"):]] = n[k]

    # paired differences, keyed d:<setting>:<metric>:<a>-<b>:<schedule>
    def addp(tag, var, a, b, key):
        p = R.paired(var, a, b, key)
        if p:
            n[f"d:{tag}"] = sgn(p["mean"])
            n[f"d:{tag}:lo"] = sgn(p["lo"])
            n[f"d:{tag}:hi"] = sgn(p["hi"])
            n[f"d:{tag}:abs"] = f"{abs(p['mean']):.1f}"
            n[f"d:{tag}:wins"] = str(p["wins"])
            n[f"d:{tag}:n"] = str(p["n"])
            n[f"d:{tag}:worst"] = f"{max(abs(p['lo']), abs(p['hi'])):.1f}"

    for var, key, _ in COLS + [(PV, "novel_1", ""), (ES13, "novel_1", ""), (RS, "novel_1", ""), (ES3, "novel_5", ""),
                           (PV, "novel_10", ""), (ES13, "novel_10", ""), (RS, "novel_10", "")]:
        t = f"{ctag(var, key)}:{key}"
        for c in ("random", "l1", "taylor", "bn_gamma", "D", "G", "V"):
            addp(f"{t}:dacis-{c}:PMP", var, "A_dacis_PMP_30", f"A_{c}_PMP_30", key)
            addp(f"{t}:dacis-{c}:P", var, R.pid(var, "dacis"), R.pid(var, c), key)
        for c in ("l1", "taylor", "bn_gamma", "D", "G", "V", "dacis"):
            addp(f"{t}:{c}-random:PMP", var, f"A_{c}_PMP_30", "A_random_PMP_30", key)
            addp(f"{t}:{c}-random:P", var, R.pid(var, c), R.pid(var, "random"), key)
        for s in ("PM", "MP", "PGP", "PMPM", "PM2x"):
            addp(f"{t}:{s}-P", var, f"B_dacis_{s}_30", R.pid(var, "dacis"), key)
        addp(f"{t}:PMP-P", var, "A_dacis_PMP_30", R.pid(var, "dacis"), key)
        addp(f"{t}:densem-dense", var, "dense_m", "dense_ce", key)
        addp(f"{t}:densermr-dense", var, "dense_rmr", "dense_ce", key)
        addp(f"{t}:dense-dacisP", var, "dense_ce", R.pid(var, "dacis"), key)
        addp(f"{t}:dense-dacisPMP", var, "dense_ce", "A_dacis_PMP_30", key)
        addp(f"{t}:dense-randomP", var, "dense_ce", R.pid(var, "random"), key)
        addp(f"{t}:dense-randomPMP", var, "dense_ce", "A_random_PMP_30", key)
        addp(f"{t}:norefine-full", var, "E_dacis_PMP_30_norefine", "A_dacis_PMP_30", key)
        addp(f"{t}:mnv3-dacisPMP", var, "K_mobilenet_v3_small", "A_dacis_PMP_30", key)
        addp(f"{t}:effb0-dacisPMP", var, "K_efficientnet_b0", "A_dacis_PMP_30", key)
        addp(f"{t}:mnv2-dacisPMP", var, "K_mobilenet_v2", "A_dacis_PMP_30", key)
        for fr in (50, 20, 10):
            addp(f"{t}:dacis-l1:PMP{fr}", var, f"C_dacis_PMP_{fr}", f"C_l1_PMP_{fr}", key)
            addp(f"{t}:dacis-random:PMP{fr}", var, f"C_dacis_PMP_{fr}", f"C_random_PMP_{fr}", key)
            addp(f"{t}:dense-dacisPMP{fr}", var, "dense_ce", f"C_dacis_PMP_{fr}", key)
    # spread among the eight scores and agreement of their ranking across partitions
    for var, key, _ in COLS:
        t = f"{ctag(var, key)}:{key}"
        for sched in ("P", "PMP"):
            M, names = score_matrix(R, var, key, sched)
            if M is None:
                continue
            mu = M.mean(0)
            n[f"spread:{t}:{sched}"] = f"{mu.max() - mu.min():.1f}"
            n[f"best:{t}:{sched}"] = CRIT_PLAIN[names[int(mu.argmax())]]
            n[f"worst:{t}:{sched}"] = CRIT_PLAIN[names[int(mu.argmin())]]
            n[f"bestv:{t}:{sched}"] = f"{mu.max():.1f}"
            n[f"worstv:{t}:{sched}"] = f"{mu.min():.1f}"
            n[f"nscores:{t}:{sched}"] = str(len(names))
            n[f"partsd:{t}:{sched}"] = f"{M.mean(1).std(ddof=1):.1f}"       # sd over partitions of the partition-mean accuracy
            w, p = kendall_w(M)
            n[f"kw:{t}:{sched}"] = f"{w:.2f}"
            n[f"kwp:{t}:{sched}"] = f"{p:.2f}"
            if "random" in names:
                ir = names.index("random")
                n[f"randrank:{t}:{sched}"] = str(int((mu > mu[ir]).sum()) + 1)      # 1 = best; position of random among the scores
    # ablations of DACIS (PlantVillage, PMP at 30%): the largest mean change
    abl_names = {"A_G_PMP_30": "the gradient norm alone", "A_V_PMP_30": "the variance alone", "A_D_PMP_30": "the Fisher ratio alone",
                 "A_GV_PMP_30": "gradient norm and variance", "A_GD_PMP_30": "gradient norm and Fisher ratio",
                 "A_VD_PMP_30": "variance and Fisher ratio", "E_dacis_PMP_30_norefine": "dropping the meta-gradient term",
                 "E_dacis_PMP_30_uniform": "a uniform depth profile", "E_dacis_PMP_30_minmax": "min-max instead of rank scaling"}
    diffs = {k: R.paired(PV, k, "A_dacis_PMP_30", "novel_5") for k in abl_names if R.has(PV, k)}
    diffs = {k: v for k, v in diffs.items() if v}
    if diffs:
        kmax = max(diffs, key=lambda k: abs(diffs[k]["mean"]))
        n["abl:max"] = f"{abs(diffs[kmax]['mean']):.1f}"
        n["abl:maxname"] = abl_names[kmax]
        n["abl:maxsign"] = "below" if diffs[kmax]["mean"] < 0 else "above"
        n["abl:count"] = str(len(diffs))
        n["abl:nsig"] = str(sum(1 for v in diffs.values() if v["lo"] > 0 or v["hi"] < 0))
    # the DACIS weights, judged on validation classes only
    lam = [R.ms(PV, f"S_dacis_PMP_30_lam{i}", "val_5") for i in range(len(LAMBDAS)) if R.has(PV, f"S_dacis_PMP_30_lam{i}")]
    if len(lam) == len(LAMBDAS):
        v = [m for m, _ in lam]
        n["lam:val5:default"] = f"{v[0]:.1f}"
        n["lam:val5:best"] = f"{max(v):.1f}"
        n["lam:val5:worst"] = f"{min(v):.1f}"
        n["lam:val5:count"] = str(len(v))
    # standard error of an accuracy estimated from 2000 test episodes (all runs of a variant, 5-shot)
    for var in VARIANTS:
        ci = [r["acc"]["novel_5"]["ci95"] / 1.96 for (v, _, _), r in R.runs.items() if v == var and "novel_5" in r["acc"]]
        if ci:
            n[f"epse:{var}"] = f"{np.mean(ci):.1f}"
    # pruned MobileNetV2: scores against random selection and against the dense network
    mt = ma = mb = 0
    mlist_a, mlist_b = [], []
    for var, key, h in COLS[:2]:
        t = f"{ctag(var, key)}:{key}"
        for f in (60, 40, 25):
            for c in ("l1", "bn_gamma", "taylor", "dacis"):
                addp(f"{t}:mp-{c}-random:{f}", var, f"MP_{c}_P_{f}", f"MP_random_P_{f}", key)
                pp = R.paired(var, f"MP_{c}_P_{f}", f"MP_random_P_{f}", key)
                if pp:
                    mt += 1
                    if pp["lo"] > 0:
                        ma += 1
                        mlist_a.append(f"{CRIT_PLAIN[c]} at {f}\\% on {h}")
                    elif pp["hi"] < 0:
                        mb += 1
                        mlist_b.append(f"{CRIT_PLAIN[c]} at {f}\\% on {h}")
            addp(f"{t}:mp-dense-dacis:{f}", var, "MK_mobilenet_v2_dense", f"MP_dacis_P_{f}", key)
            addp(f"{t}:mp-dacis-resnet:{f}", var, f"MP_dacis_P_{f}", "A_dacis_PMP_30", key)
    if mt:
        n["cmpm:total"], n["cmpm:above"], n["cmpm:below"] = str(mt), str(ma), str(mb)
        n["cmpm:abovelist"] = "; ".join(mlist_a) if mlist_a else "none"
        n["cmpm:belowlist"] = "; ".join(mlist_b) if mlist_b else "none"
    # all paired comparisons "computed score minus random selection" at 30%: how many favour either side
    for sched in ("PMP", "P"):
        tot = above = below = 0
        blist, alist, gains, pvals, halfw, pabove = [], [], [], [], [], []
        for var, key, h in COLS:
            best = None
            for c in ("l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"):
                a = f"A_{c}_PMP_30" if sched == "PMP" else R.pid(var, c)
                b = "A_random_PMP_30" if sched == "PMP" else R.pid(var, "random")
                pp = R.paired(var, a, b, key)
                if pp is None:
                    continue
                tot += 1
                gains.append(pp["mean"])
                pvals.append(pp["p"])
                if pp["mean"] > 0:
                    pabove.append(pp["p"])
                halfw.append((pp["hi"] - pp["lo"]) / 2)
                if best is None or pp["mean"] > best[1]["mean"]:
                    best = (c, pp)
                if pp["lo"] > 0:
                    above += 1
                    alist.append(f"{CRIT_PLAIN[c]} on {h}")
                elif pp["hi"] < 0:
                    below += 1
                    blist.append(f"{CRIT_PLAIN[c]} on {h}")
            if best is not None:
                t = f"{ctag(var, key)}:{key}"
                n[f"gain:{t}:{sched}"] = f"{best[1]['mean']:.1f}"
                n[f"gainhi:{t}:{sched}"] = f"{best[1]['hi']:.1f}"
                n[f"gainname:{t}:{sched}"] = CRIT_PLAIN[best[0]]
        if tot:
            n[f"cmp:total:{sched}"] = str(tot)
            n[f"cmp:above:{sched}"] = str(above)
            n[f"cmp:below:{sched}"] = str(below)
            n[f"cmp:abovelist:{sched}"] = "; ".join(alist) if alist else "none"
            n[f"cmp:belowlist:{sched}"] = "; ".join(blist) if blist else "none"
            n[f"cmp:chance:{sched}"] = f"{tot * 0.05:.1f}"                # intervals expected to exclude zero if nothing differed
            n[f"cmp:maxgain:{sched}"] = f"{max(gains):.1f}"
            n[f"cmp:bonf:{sched}"] = f"{min(1.0, min(pvals) * tot):.2f}"      # smallest p value after Bonferroni correction
            n[f"cmp:bonfabove:{sched}"] = f"{min(1.0, min(pabove) * tot):.2f}" if pabove else "1.00"
            n[f"cmp:halfw:{sched}"] = f"{np.mean(halfw):.1f}"
            n[f"cmp:halfwmax:{sched}"] = f"{max(halfw):.1f}"
    ks = [float(n[k]) for k in n if k.startswith("kw:") and k.endswith(":PMP")]
    if ks:
        n["kw:max"] = f"{max(ks):.2f}"
        n["kw:min"] = f"{min(ks):.2f}"
    # cost: how much of the MAC saving survives as latency
    cv = R.cost_variant()
    if cv:
        n["cost:variant"] = cv
        for frac, iid in ((30, "A_l1_PMP_30"), (50, "C_l1_PMP_50"), (20, "C_l1_PMP_20"), (10, "C_l1_PMP_10")):
            if not R.has(cv, iid):
                continue
            for fld, sub, name in (("macs_128", None, "macs"), ("cpu_1thread_b1_ms", "median", "cpu"), ("gpu_b128_img_per_s", None, "gpu")):
                a, b = R.eff(cv, "dense_ce", fld, sub), R.eff(cv, iid, fld, sub)
                if a and b:
                    n[f"cost:{name}-red:{frac}"] = f"{(1 - b / a) * 100:.0f}" if name != "gpu" else f"{(b / a - 1) * 100:.0f}"
        for arch in ("mobilenet_v3_small", "mobilenet_v2", "efficientnet_b0"):
            for fld, sub, name in (("macs_128", None, "macs"), ("cpu_1thread_b1_ms", "median", "cpu"), ("params", None, "params")):
                b = R.eff(cv, f"K_{arch}", fld, sub)
                if b is not None:
                    n[f"cost:{arch}:{name}"] = f"{b / 1e9:.3f}" if name == "macs" else (f"{b / 1e6:.2f}" if name == "params" else f"{b:.1f}")
    if R.tuning:
        n["tune:best"] = f"{R.tuning['best_lr']:g}"
        for k, v in R.tuning["rows"].items():
            k = "S0" if k.startswith("S0") else k
            n[f"tune:{k}:1"] = f"{v['1']:.1f}"
            n[f"tune:{k}:5"] = f"{v['5']:.1f}"
    if R.tuning_rs:
        n["tune-rs:best"] = f"{R.tuning_rs['best_lr']:g}"
        for k, v in R.tuning_rs["rows"].items():
            k = "S0" if k.startswith("S0") else k
            n[f"tune-rs:{k}:1"] = f"{v['1']:.1f}"
            n[f"tune-rs:{k}:5"] = f"{v['5']:.1f}"
    h = R.host.get("host", {})
    if R.tuning_low:
        rows = R.tuning_low["rows"]
        s0 = rows.get("S0 (no episodic stage)", {}).get("5")
        best = max((v["5"] for k, v in rows.items() if not k.startswith("S0")), default=None)
        if s0 is not None and best is not None:
            n["tune-low:s0"], n["tune-low:best"] = f"{s0:.1f}", f"{best:.1f}"
            n["tune-low:lr"] = min(float(k) for k in rows if not k.startswith("S0")).__format__("g")
    n["hw-gpu"] = h.get("gpu", "an unnamed GPU").replace("_", " ")
    cpu = h.get("cpu", "")
    n["hw-cpu"] = cpu if cpu and cpu.lower() != "unknown" else f"a {h.get('cpu_count', '?')}-thread server CPU (model not reported by the host)"
    n["hw-gpumem"] = str(h.get("gpu_mem_gb", ""))
    return n


def write_numbers(R, path, extra=None):
    lines = ["% generated by pmp_dacis.report.write_numbers; do not edit"]
    all_numbers = dict(numbers(R))
    all_numbers.update(extra or {})
    for k, v in all_numbers.items():
        lines.append(r"\expandafter\gdef\csname num@%s\endcsname{%s}" % (k, v))
    _write(path, "\n".join(lines) + "\n")


# ---- claims check -------------------------------------------------------------------------------
def psgn(x, d=1):
    """Signed number as plain text (sgn() writes LaTeX)."""
    return f"{x:+.{d}f}"


def _ci_has_zero(p):
    return p is not None and p["lo"] <= 0 <= p["hi"]


def claims_check(R):
    """Each qualitative sentence of the paper has a test here. A FAIL means the sentence must be reworded."""
    out, fails = [], 0

    def claim(name, ok, detail):
        nonlocal fails
        if ok is None:
            out.append(f"SKIP  {name}: not enough runs")
            return
        out.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        fails += 0 if ok else 1

    def ci(p):
        return f"{psgn(p['mean'])} [{psgn(p['lo'])},{psgn(p['hi'])}]"

    for var, key, h in COLS:
        if not R.seeds(var):
            continue
        # C1: no computed score is clearly better than random selection (PMP and P)
        for sched in ("PMP", "P"):
            ps = {}
            for c in ("l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"):
                a = f"A_{c}_PMP_30" if sched == "PMP" else R.pid(var, c)
                b = "A_random_PMP_30" if sched == "PMP" else R.pid(var, "random")
                p = R.paired(var, a, b, key)
                if p is not None:
                    ps[c] = p
            if len(ps) >= 3:
                wins = [c for c, v in ps.items() if v["lo"] > 0]
                claim(f"C1 {h} [{sched}]: no computed score is above random selection with a 95% interval clear of zero",
                      not wins, "above random: " + (", ".join(f"{c} {ci(ps[c])}" for c in wins) if wins else "none")
                      + "; DACIS " + (ci(ps["dacis"]) if "dacis" in ps else "n/a"))
            else:
                claim(f"C1 {h} [{sched}]", None, "")
        # C2: the spread among the score means is below the spread between partitions
        M, names = score_matrix(R, var, key, "PMP")
        if M is not None:
            mu = M.mean(0)
            out.append(f"INFO  {h}: spread of the {len(names)} score means (PMP) {mu.max() - mu.min():.1f} points, "
                       f"sd over partitions {M.mean(1).std(ddof=1):.1f} (not a claim of the paper)")
            w, p = kendall_w(M)
            claim(f"C3 {h}: the score ranking is at best weakly concordant across partitions (Kendall W < 0.5)", w < 0.5, f"W={w:.2f}, p={p:.2f}")
        else:
            out.append(f"SKIP  {h}: spread of the score means")
        # C4: the episodic stage does not add accuracy
        dm = R.paired(var, "dense_m", "dense_ce", key)
        pm_ = R.paired(var, "A_dacis_PMP_30", R.pid(var, "dacis"), key)
        if dm and pm_:
            claim(f"C4 {h}: dense+M and PMP are not above their M-free counterparts (upper bound of interval <= +1)",
                  dm["hi"] <= 1.0 and pm_["hi"] <= 1.0, f"dense+M - dense {ci(dm)}, PMP - P {ci(pm_)}")
        else:
            claim(f"C4 {h}", None, "")
        sch = {s_: R.paired(var, f"B_dacis_{s_}_30", R.pid(var, "dacis"), key) for s_ in ("PM", "MP", "PGP", "PMPM", "PM2x")}
        sch = {k: v for k, v in sch.items() if v}
        if sch:
            wins = [k for k, v in sch.items() if v["lo"] > 0]
            claim(f"C4b {h}: no schedule with an episodic or probe stage is above P with an interval clear of zero", not wins,
                  "; ".join(f"{k} {ci(v)}" for k, v in sch.items()))
        # C5: pruning to 30% costs accuracy relative to the dense network
        dd = R.paired(var, "dense_ce", "A_dacis_PMP_30", key)
        if dd:
            claim(f"C5 {h}: the dense network beats the 30% network (interval above 0)", dd["lo"] > 0, f"dense - DACIS PMP {ci(dd)}")
        # C9: the budget moves accuracy more than the choice of score does
        if R.has(var, "C_dacis_PMP_50") and R.has(var, "C_dacis_PMP_10") and M is not None:
            a50, a10 = R.ms(var, "C_dacis_PMP_50", key)[0], R.ms(var, "C_dacis_PMP_10", key)[0]
            mu = M.mean(0)
            claim(f"C9 {h}: going from 50% to 10% of the parameters costs more than the spread among scores at 30%",
                  (a50 - a10) > (mu.max() - mu.min()), f"50%->10%: {a50 - a10:.1f} points; score spread {mu.max() - mu.min():.1f}")
    # C1b: the same at 1 and 10 shots
    for var, h in ((RS, "RESISC45"), (ES13, "EuroSAT-MS"), (PV, "PlantVillage")):
        for key in ("novel_1", "novel_10"):
            ps = {}
            for c in ("l1", "bn_gamma", "taylor", "G", "V", "D", "dacis"):
                p = R.paired(var, f"A_{c}_PMP_30", "A_random_PMP_30", key)
                if p is not None:
                    ps[c] = p
            if len(ps) >= 3:
                wins = [c for c, v in ps.items() if v["lo"] > 0]
                claim(f"C1b {h} [{key}]: no computed score is above random selection with a 95% interval clear of zero (PMP)",
                      not wins, "above random: " + (", ".join(f"{c} {ci(ps[c])}" for c in wins) if wins else "none"))
    # C10: longer recovery helps more than the score (PlantVillage)
    if all(R.has(PV, i) for i in ("R_random_P_30_r6", "R_l1_P_30_r6", "R_taylor_P_30_r6", "R_dacis_P_30_r6")) and R.pid(PV, "random"):
        gains = [R.ms(PV, f"R_{c}_P_30_r6", "novel_5")[0] - R.ms(PV, R.pid(PV, c), "novel_5")[0] for c in ("random", "l1", "taylor", "dacis")]
        one = [R.ms(PV, R.pid(PV, c), "novel_5")[0] for c in ("random", "l1", "taylor", "dacis")]
        claim("C10 PlantVillage: six epochs of recovery instead of one raise accuracy more than the spread among four scores at one epoch",
              float(np.mean(gains)) > (max(one) - min(one)), f"mean gain {np.mean(gains):.1f}; spread at one epoch {max(one) - min(one):.1f}")
    cv = R.cost_variant()
    if cv and R.has(cv, "A_l1_PMP_30") and R.has(cv, "dense_ce"):
        a, b = R.eff(cv, "dense_ce", "macs_128"), R.eff(cv, "A_l1_PMP_30", "macs_128")
        c, d = R.eff(cv, "dense_ce", "cpu_1thread_b1_ms", "median"), R.eff(cv, "A_l1_PMP_30", "cpu_1thread_b1_ms", "median")
        if a and b and c and d:
            claim("C6 cost: the CPU latency saving at 30% is smaller than the MAC saving", (1 - d / c) < (1 - b / a),
                  f"MACs -{(1 - b / a) * 100:.0f}%, CPU latency -{(1 - d / c) * 100:.0f}%")
    if R.has(ES13, "dense_ce") and R.has(ES3, "dense_ce"):
        diffs = np.array([R.runs[(ES13, s, "dense_ce")]["acc"]["novel_5"]["mean"] - R.runs[(ES3, s, "dense_ce")]["acc"]["novel_5"]["mean"]
                          for s in R.seeds_with(ES13, "dense_ce") if (ES3, s, "dense_ce") in R.runs])
        if len(diffs) > 1:
            claim("C7 EuroSAT-MS: thirteen bands do not beat the three visible bands for the dense network", diffs.mean() <= 0.5,
                  f"13 bands - RGB = {psgn(diffs.mean())} points over {len(diffs)} partitions")
    if all(R.has(PV, f"S_dacis_PMP_30_lam{i}") for i in range(len(LAMBDAS))):
        v = [R.ms(PV, f"S_dacis_PMP_30_lam{i}", "val_5")[0] for i in range(len(LAMBDAS))]
        claim("C12 DACIS weights: no weighting beats the default by more than 1 point on validation classes (PlantVillage, 5-shot)",
              max(v) - v[0] <= 1.0, f"default {v[0]:.1f}, best {max(v):.1f}, worst {min(v):.1f}")
    if R.tuning:
        rows = R.tuning["rows"]
        s0 = rows.get("S0 (no episodic stage)", {}).get("5")
        best = max((v["5"] for k, v in rows.items() if not k.startswith("S0")), default=None)
        if s0 is not None and best is not None:
            claim("C8 validation sweep: no episodic learning rate beats stopping after S0 (5-shot, PlantVillage)", best <= s0 + 0.2,
                  f"S0 {s0:.1f}, best episodic {best:.1f}")
    if R.tuning_rs:
        rows = R.tuning_rs["rows"]
        s0 = rows.get("S0 (no episodic stage)", {}).get("5")
        best = max((v["5"] for k, v in rows.items() if not k.startswith("S0")), default=None)
        if s0 is not None and best is not None:
            claim("C8b validation sweep: no episodic learning rate beats stopping after S0 (5-shot, RESISC45)", best <= s0 + 0.2,
                  f"S0 {s0:.1f}, best episodic {best:.1f}")
    head = f"claims check: {len([o for o in out if o.startswith('PASS')])} pass, {fails} fail, {len([o for o in out if o.startswith('SKIP')])} skipped"
    return head + "\n" + "\n".join(out) + "\n"


def markdown_summary(R):
    """Markdown tables of every run (mean ± sd over seeds), straight from the saved results."""
    def cell(var, id_, key):
        ms = R.ms(var, id_, key) if R.has(var, id_) else None
        return "--" if ms is None else f"{ms[0]:.1f} ± {ms[1]:.1f}"

    out = []
    for var, title, keys, heads in (
        (RS, "NWPU-RESISC45 novel classes (5-way)", ("novel_1", "novel_5", "novel_10"), ("1-shot", "5-shot", "10-shot")),
        (ES13, "EuroSAT-MS (13 bands) novel classes (5-way)", ("novel_1", "novel_5", "novel_10"), ("1-shot", "5-shot", "10-shot")),
        (ES3, "EuroSAT-MS (RGB bands only) novel classes (5-way)", ("novel_1", "novel_5", "novel_10"), ("1-shot", "5-shot", "10-shot")),
        (PV, "PlantVillage novel classes and PlantDoc (5-way)", ("novel_1", "novel_5", "novel_10", "plantdoc_5"),
         ("1-shot", "5-shot", "10-shot", "PlantDoc 5-shot")),
    ):
        ids = sorted({k[2] for k in R.runs if k[0] == var})
        if not ids:
            continue
        out.append(f"### {title}\n")
        out.append("| run | params kept | " + " | ".join(heads) + " |")
        out.append("|---|---|" + "---|" * len(heads))
        for id_ in ids:
            if not R.has(var, id_):
                continue
            pf = R.vals(var, id_, None, getter=lambda r: r["param_frac"] * 100).mean()
            out.append(f"| {id_} | {pf:.1f}% | " + " | ".join(cell(var, id_, k) for k in keys) + " |")
        out.append("")
    return "\n".join(out)


def build_all(out_root, gen_dir):
    """Write every generated table / figure / number used by the paper into gen_dir."""
    R = Results(out_root)
    os.makedirs(gen_dir, exist_ok=True)
    for name, fn in (("tab_scores", table_scores), ("tab_pairs", table_pairs), ("tab_sched", table_sched),
                     ("tab_budget", table_budget), ("tab_budget_scores", table_budget_scores), ("tab_recovery", table_recovery),
                     ("tab_ablation", table_ablation), ("tab_eurosat", table_eurosat), ("tab_lambda", table_lambda),
                     ("tab_mlr", table_mlr), ("tab_shots", table_shots), ("tab_mbv2", table_mbv2)):
        try:
            _write(os.path.join(gen_dir, name + ".tex"), fn(R))
        except Exception as e:  # a missing run must not stop the other outputs
            _write(os.path.join(gen_dir, name + ".tex"), f"% {name} could not be built: {e!r}\n")
            print("WARNING:", name, repr(e))
    try:
        make_figures(R, gen_dir)
    except Exception as e:
        print("WARNING: figures", repr(e))
    _write(os.path.join(gen_dir, "summary.md"), markdown_summary(R))
    return R


def readme_block(R):
    """Markdown for the README: the headline rows of the reported run, mean +/- sd over class partitions."""
    def c(var, id_, key):
        ms = R.ms(var, id_, key) if id_ and R.has(var, id_) else None
        return "--" if ms is None else f"{ms[0]:.1f} ± {ms[1]:.1f}"

    h = R.host.get("host", {})
    out = [f"Run on {h.get('gpu', 'unknown GPU')} ({h.get('cpu', 'unknown CPU')}); {len(R.seeds(RS) or R.seeds(PV))} random class partitions; "
           f"episodic learning rate {R.tuning['best_lr'] if R.tuning else 'n/a'}; mean ± standard deviation over partitions, "
           "5-way 5-shot test episodes, identical for every row.\n",
           "| model | params kept | RESISC45 | EuroSAT-MS | PlantVillage | PlantDoc |",
           "|---|---|---|---|---|---|"]

    def row(label, id_pair):
        ids = {v: id_pair(v) for v in (RS, ES13, PV)}
        v0 = next((v for v in (RS, PV, ES13) if ids[v] and R.has(v, ids[v])), None)
        if v0 is None:
            return
        pf = R.vals(v0, ids[v0], None, getter=lambda r: r["param_frac"] * 100).mean()
        out.append(f"| {label} | {pf:.0f}% | {c(RS, ids[RS], 'novel_5')} | {c(ES13, ids[ES13], 'novel_5')} | "
                   f"{c(PV, ids[PV], 'novel_5')} | {c(PV, ids[PV], 'plantdoc_5')} |")

    row("ResNet-18 dense (S0)", lambda v: "dense_ce")
    row("ResNet-18 dense (S0+R+M+R)", lambda v: "dense_rmr")
    for arch, lab in (("mobilenet_v3_small", "MobileNetV3-Small"), ("mobilenet_v2", "MobileNetV2"), ("efficientnet_b0", "EfficientNet-B0")):
        row(lab, lambda v, a=arch: f"K_{a}")
    for cc in SCORES:
        row(f"30%, {cc}, prune+recover (P)", lambda v, cc=cc: R.pid(v, cc))
    for cc in SCORES:
        row(f"30%, {cc}, PMP", lambda v, cc=cc: f"A_{cc}_PMP_30")
    for frac in (50, 20, 10):
        for cc in ("l1", "dacis"):
            row(f"{frac}%, {cc}, PMP", lambda v, cc=cc, frac=frac: f"C_{cc}_PMP_{frac}")
    return "\n".join(out) + "\n"


def replication_numbers(R, R2):
    """Compare the same (variant, seed, run) between two machines. Keys: repl:*"""
    diffs = []
    for key, r in R.runs.items():
        r2 = R2.runs.get(key)
        if r2 is None:
            continue
        for k in ("novel_1", "novel_5", "novel_10", "plantdoc_5"):
            a, b = r["acc"].get(k), r2["acc"].get(k)
            if a and b:
                diffs.append(a["mean"] - b["mean"])
    n = {}
    if diffs:
        d = np.array(diffs)
        n["repl:n"] = str(len(d))
        n["repl:mad"] = f"{np.abs(d).mean():.1f}"
        n["repl:max"] = f"{np.abs(d).max():.1f}"
        n["repl:bias"] = sgn(d.mean())
        n["repl:seeds"] = str(len({k[1] for k in R.runs if k in R2.runs}))
        h2 = R2.host.get("host", {})
        n["repl:gpu"] = h2.get("gpu", "a second GPU").replace("_", " ")
    return n
