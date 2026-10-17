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
app = marimo.App(width="medium", app_title="Channel pruning: extension run")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Extension run: more class partitions and a pruned MobileNetV2

    **What to do:** press *Run all*, wait, then click **Results** at the bottom. Nothing else needs to be selected or typed.

    This notebook adds two things to the main study (`pmp_dacis_cv4eo.py`), with the same code and settings:

    1. class partitions 5 to 9 for the RESISC45 and EuroSAT-MS comparisons (the main run used partitions 0 to 4), so that
       the intervals over partitions become narrower;
    2. MobileNetV2 pruned with the same channel scores (random, filter L1, BatchNorm scale, Taylor, DACIS) at 60, 40 and
       25 % of its parameters, on partitions 0 to 9, to see whether the findings are specific to ResNet-18.

    It downloads RESISC45 and EuroSAT-MS (about 2 GB). On an RTX 6000-class GPU it takes roughly 1.5 hours. Finished runs are
    cached, so *Run all* can be repeated after an interruption.
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
    WORK = Path(os.environ.get("PMP_WORK", "pmp_dacis_ext_run")).resolve()
    SRC = WORK / "src"
    PKG_B64 = """@@PKG_B64@@"""

    def _unpack(b64, dest):
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64)), mode="r:gz") as tf:
            try:
                tf.extractall(dest, filter="data")
            except TypeError:
                tf.extractall(dest)

    _unpack(PKG_B64, SRC)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from pmp_dacis import data as D
    from pmp_dacis import evaluate as EV
    from pmp_dacis import experiments as E
    from pmp_dacis import paper as PP
    return D, E, EV, PP, WORK


@app.cell
def _(E, EV, WORK, json, os):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    PROFILE = os.environ.get("PMP_PROFILE") or ("paper" if device.type == "cuda" else "smoke")
    CFG = dict(E.PROFILES[PROFILE])
    # the main run chose 1e-4 on PlantVillage validation classes; every dataset uses that value
    CFG["m_lr"] = 0.0001 if PROFILE == "paper" else 0.001
    HOST = EV.host_info(device)
    OUT = WORK / "results"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "host.json").write_text(json.dumps({"profile": PROFILE, "host": HOST, "cfg": {k: v for k, v in CFG.items()}}, indent=1, default=str))
    print("profile:", PROFILE)
    print("host   :", HOST)
    if PROFILE not in ("paper", "micro"):
        print("\n*** No GPU found: tiny smoke profile. Its numbers are NOT results. ***")
    return CFG, HOST, OUT, PROFILE, device


@app.cell
def _(CFG, D, PROFILE, WORK, os):
    DATA = WORK / "data"
    CACHE = WORK / "cache"
    D.ensure_data(str(DATA), str(CACHE), CFG, keep_parquet=os.environ.get("PMP_KEEP_PARQUET") == "1", only=("resisc45", "eurosat"))
    print("datasets decoded into", CACHE, "| profile", PROFILE)
    return CACHE, DATA


@app.cell
def _(CACHE, CFG, DATA, E, OUT, PROFILE, device, os, time):
    VARIANTS = ("resisc45", "eurosat13")
    JOBS = E.ext_jobs(VARIANTS) if PROFILE == "paper" else E.ext_jobs(VARIANTS, new_seeds=(2,), all_seeds=(0, 1))
    _fh = open(OUT / "log.txt", "a")
    _t0 = time.time()
    E.log(f"starting extension runs: {[(v, len(g), s) for v, g, s in JOBS]}", _fh)
    _ds = E.run_all(CFG, device, str(DATA), str(CACHE), str(OUT), variants=VARIANTS, fh=_fh, jobs=JOBS)
    # the main run picked the smallest learning rate of its grid for the episodic stage; check smaller ones on validation classes
    _low = dict(CFG, m_lr_grid=[1e-5, 2.5e-5, 5e-5, 1e-4] if PROFILE == "paper" else [1e-4, 2.5e-4])
    E.tune_m_lr(_low, device, _ds, str(OUT), _fh, variant="resisc45")
    os.replace(OUT / "tuning_m_lr_resisc45.json", OUT / "tuning_m_lr_resisc45_low.json")
    E.log(f"all runs finished in {(time.time() - _t0) / 60:.1f} min (this session)", _fh)
    _fh.close()
    RUNS_DONE = True
    return JOBS, RUNS_DONE


@app.cell
def _(JOBS, OUT, PP, RUNS_DONE, WORK, mo):
    assert RUNS_DONE
    ZIP = PP.zip_results(str(OUT))
    ZIP_PATH = WORK / "results_extension.zip"
    ZIP_PATH.write_bytes(ZIP)
    mo.vstack([
        mo.md(f"### Done. `results_extension.zip` is {len(ZIP) / 1e6:.1f} MB (also saved as `{ZIP_PATH}`). Click **Results** to download it."),
        mo.download(data=lambda: ZIP_PATH.read_bytes(), filename="results_extension.zip", mimetype="application/zip", label="Results"),
    ])
    return (ZIP, ZIP_PATH)


if __name__ == "__main__":
    app.run()
