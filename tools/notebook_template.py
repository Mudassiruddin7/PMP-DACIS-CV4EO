# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.9",
#     "torch",
#     "torchvision",
#     "numpy",
#     "pandas",
#     "scipy",
#     "pyarrow",
#     "pillow",
#     "matplotlib",
#     "huggingface_hub",
#     "psutil",
#     "tqdm",
# ]
# ///
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="PMP-DACIS: run everything")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Channel pruning for few-shot Earth observation: one-click reproduction

    **What to do:** press *Run all*. Nothing else needs to be selected or typed.
    When the last cell finishes, click the **Results** button at the bottom; it downloads one zip with every
    number, table, figure, log and the compiled paper PDFs.

    **What this notebook does, in order**

    1. Detects the GPU and picks the `paper` profile (a CPU-only machine gets a tiny `smoke` profile, which only checks that the code runs).
    2. Downloads NWPU-RESISC45 (aerial scenes), EuroSAT-MS (Sentinel-2, 13 bands), PlantVillage and PlantDoc from the Hugging Face Hub (about 5 GB).
    3. Trains and evaluates every model of the study: ResNet-18 pruned by eight channel scores under several schedules and budgets,
       three off-the-shelf compact networks, 5-way 1/5/10-shot episodes on held-out classes, five class partitions (seeds);
       then the second part: partitions 5 to 9 for the RESISC45 and EuroSAT-MS comparisons and MobileNetV2 pruned with the same
       scores. On an RTX 6000-class GPU this takes about 3.5 hours (2.3 h and 1.3 h). Finished runs are cached on disk, so *Run all*
       can be repeated after an interruption and resumes where it stopped.
    4. Builds all tables and figures from the saved per-episode results. No number in the paper is typed by hand.
    5. Compiles the anonymous review PDF, the author-version PDF and the supplement with Tectonic.
    6. Packs everything into `results.zip`.

    Every accuracy in the paper comes from the files this notebook writes. If a statement in the text no longer
    holds for your run, the *claims check* cell says so.
    """)
    return


@app.cell
def _():
    import base64
    import io
    import json
    import os
    import platform
    import subprocess
    import sys
    import tarfile
    import time
    from pathlib import Path
    return Path, base64, io, json, os, platform, subprocess, sys, tarfile, time


@app.cell
def _(Path, base64, io, os, sys, tarfile):
    # The experiment library and the paper sources are embedded below (gzip + base64) so that this single
    # file is all that is needed. The readable sources live in the repository (pmp_dacis/ and paper/).
    WORK = Path(os.environ.get("PMP_WORK", "pmp_dacis_run")).resolve()
    SRC = WORK / "src"
    PAPER = WORK / "paper"
    PKG_B64 = """@@PKG_B64@@"""
    PAPER_B64 = """@@PAPER_B64@@"""

    def _unpack(b64, dest):
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64)), mode="r:gz") as tf:
            try:
                tf.extractall(dest, filter="data")
            except TypeError:
                tf.extractall(dest)

    _unpack(PKG_B64, SRC)
    _unpack(PAPER_B64, PAPER)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from pmp_dacis import data as D
    from pmp_dacis import evaluate as EV
    from pmp_dacis import experiments as E
    from pmp_dacis import paper as PP
    from pmp_dacis import report as RP
    return D, E, EV, PAPER, PP, RP, WORK


@app.cell
def _(E, EV, WORK, json, os):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    PROFILE = os.environ.get("PMP_PROFILE") or ("paper" if device.type == "cuda" else "smoke")
    CFG = dict(E.PROFILES[PROFILE])
    HOST = EV.host_info(device)
    OUT = WORK / "results"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "host.json").write_text(json.dumps({"profile": PROFILE, "host": HOST, "cfg": {k: v for k, v in CFG.items()}}, indent=1, default=str))
    print("profile:", PROFILE)
    print("host   :", HOST)
    if device.type != "cuda":
        print("\n*** No GPU found: running a tiny profile. Its numbers are NOT paper numbers. ***")
    return CFG, HOST, OUT, PROFILE, device


@app.cell
def _(CFG, D, PROFILE, WORK, os):
    DATA = WORK / "data"
    CACHE = WORK / "cache"
    # raw parquet files are deleted once decoded (disk space); PMP_KEEP_PARQUET=1 keeps them
    D.ensure_data(str(DATA), str(CACHE), CFG, keep_parquet=os.environ.get("PMP_KEEP_PARQUET") == "1")
    print("datasets decoded into", CACHE, "| profile", PROFILE)
    return CACHE, DATA


@app.cell
def _(CACHE, CFG, DATA, E, OUT, PROFILE, device, time):
    VARIANTS = ("resisc45", "eurosat13", "eurosat3", "plantvillage")
    TIERS = (1, 2, 3) if PROFILE in ("paper", "micro") else (1,)
    _fh = open(OUT / "log.txt", "a")
    _t0 = time.time()
    E.log(f"starting runs: tiers {TIERS}, variants {VARIANTS}", _fh)
    E.run_all(CFG, device, str(DATA), str(CACHE), str(OUT), variants=VARIANTS, tiers=TIERS, fh=_fh)
    E.log(f"all runs finished in {(time.time() - _t0) / 60:.1f} min (this session)", _fh)
    _fh.close()
    RUNS_DONE = True
    return RUNS_DONE, VARIANTS


@app.cell
def _(CACHE, CFG, DATA, E, OUT, PROFILE, RUNS_DONE, device, os, time):
    # Second part of the study: more class partitions for the Earth-observation comparisons, MobileNetV2 pruned with the same
    # scores, and a check of smaller episodic learning rates. Without a GPU (smoke profile) it is skipped.
    assert RUNS_DONE
    EXT_DONE = False
    if PROFILE in ("paper", "micro"):
        _eo = ("resisc45", "eurosat13")
        _jobs = E.ext_jobs(_eo) if PROFILE == "paper" else E.ext_jobs(_eo, new_seeds=(2,), all_seeds=(0, 1))
        _fh = open(OUT / "log.txt", "a")
        _t0 = time.time()
        E.log(f"starting extension runs: {[(v, len(g), s) for v, g, s in _jobs]}", _fh)
        _ds = E.run_all(CFG, device, str(DATA), str(CACHE), str(OUT), variants=_eo, fh=_fh, jobs=_jobs)
        _low = dict(CFG, m_lr_grid=[1e-5, 2.5e-5, 5e-5, 1e-4] if PROFILE == "paper" else [1e-4, 2.5e-4])
        E.tune_m_lr(_low, device, _ds, str(OUT), _fh, variant="resisc45", tag="resisc45_low")
        E.log(f"extension runs finished in {(time.time() - _t0) / 60:.1f} min (this session)", _fh)
        _fh.close()
        EXT_DONE = True
    return (EXT_DONE,)


@app.cell
def _(EXT_DONE, OUT, PAPER, RP, RUNS_DONE):
    assert RUNS_DONE
    GEN = PAPER / "generated"
    R = RP.build_all(str(OUT), str(GEN))
    RP.write_numbers(R, str(GEN / "numbers.tex"))
    print("tables, figures and numbers written to", GEN)
    return GEN, R


@app.cell
def _(GEN, OUT, PAPER, PP, R, RP, WORK):
    BUILD_LOG = []
    claims = RP.claims_check(R)
    (OUT / "claims_check.txt").write_text(claims)
    print(claims)
    _tect = PP.ensure_tectonic(str(WORK / "bin"))
    PDFS = {}
    for _name in ("paper_review", "paper_final", "paper_supp"):
        try:
            _pdf, _log = PP.compile_pdf(_tect, str(PAPER), _name + ".tex")
        except Exception as _e:  # keep going: the zip must always be produced
            _pdf, _log = None, repr(_e)
        PDFS[_name] = _pdf
        BUILD_LOG.append(f"== {_name}: {'OK' if _pdf else 'FAILED'}\n{_log[-1500:]}")
    (OUT / "paper_build_log.txt").write_text("\n".join(BUILD_LOG))
    print({k: bool(v) for k, v in PDFS.items()})
    return BUILD_LOG, PDFS, claims


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Headline results (computed from this run)")
    return


@app.cell
def _(R, mo):
    import pandas as pd
    _rows = []
    _cols = (("resisc45", "novel_5", "RESISC45 5-shot"), ("eurosat13", "novel_5", "EuroSAT-MS 5-shot"),
             ("plantvillage", "novel_5", "PlantVillage 5-shot"), ("plantvillage", "plantdoc_5", "PlantDoc 5-shot"))
    for _vid, _lab in (("dense_ce", "Dense ResNet-18"), ("dense_m", "Dense + episodic"), ("A_random_PMP_30", "Random, PMP, 30% params"),
                       ("A_l1_PMP_30", "Filter L1, PMP, 30% params"), ("A_taylor_PMP_30", "Taylor, PMP, 30% params"),
                       ("A_dacis_PMP_30", "DACIS, PMP, 30% params"), ("K_mobilenet_v3_small", "MobileNetV3-Small")):
        _cells = {}
        for _v, _k, _h in _cols:
            _ms = R.ms(_v, _vid, _k) if R.has(_v, _vid) else None
            _cells[_h] = "--" if _ms is None else "%.1f ± %.1f" % _ms
        _rows.append({"model": _lab, **_cells})
    mo.ui.table(pd.DataFrame(_rows), selection=None) if _rows else mo.md("no results yet")
    return (pd,)


@app.cell
def _(OUT, PAPER, PDFS, PP, WORK, mo):
    import os as _os
    _extra = []
    for _base, _dirs, _files in _os.walk(PAPER):
        for _f in _files:
            if _f.endswith((".aux", ".log", ".out", ".bbl", ".blg")):
                continue
            _full = _os.path.join(_base, _f)
            _rel = _os.path.relpath(_full, PAPER).replace("\\", "/")
            _extra.append((_full, _rel if _rel.endswith(".pdf") and "/" not in _rel else "paper/" + _rel))
    ZIP = PP.zip_results(str(OUT), extra_files=_extra)
    ZIP_PATH = WORK / "results.zip"
    ZIP_PATH.write_bytes(ZIP)
    mo.vstack([
        mo.md(f"### Done. `results.zip` is {len(ZIP) / 1e6:.1f} MB (also saved as `{ZIP_PATH}`). Click **Results** to download it."),
        mo.download(data=lambda: ZIP_PATH.read_bytes(), filename="results.zip", mimetype="application/zip", label="Results"),
    ])
    return (ZIP, ZIP_PATH)


if __name__ == "__main__":
    app.run()
