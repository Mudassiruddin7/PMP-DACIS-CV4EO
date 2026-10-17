"""Latency of the architectures of the study on a CPU whose model is known (the GPU host of the main run did not report its CPU).

    python tools/latency_named_cpu.py [results_dir] [out.json]

Only the architecture matters for latency, so the networks are rebuilt from the widths saved in the run files (random weights).
Single thread, batch size 1, 128 x 128 RGB input, fp32, median over repeats of the median of 100 forward passes.
"""
import glob
import json
import os
import platform
import statistics
import subprocess
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from pmp_dacis.evaluate import latency_ms  # noqa: E402
from pmp_dacis.model import PResNet18, count_macs, count_params, make_compact, prune_mbv2  # noqa: E402


def cpu_name():
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20)
            return out.stdout.strip()
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def main(res_dir, out_path):
    run = lambda v, s, i: json.load(open(os.path.join(res_dir, "runs", v, f"seed{s}", f"{i}.json")))
    items = [("ResNet-18, dense", "dense", PResNet18(3))]
    for frac, rid in ((50, "C_l1_PMP_50"), (30, "A_l1_PMP_30"), (20, "C_l1_PMP_20"), (10, "C_l1_PMP_10")):
        items.append((f"ResNet-18, {frac}%", f"r18_{frac}", PResNet18(3, mids=run("resisc45", 0, rid)["mids"])))
    for arch, lab in (("mobilenet_v3_small", "MobileNetV3-Small"), ("mobilenet_v2", "MobileNetV2"), ("efficientnet_b0", "EfficientNet-B0")):
        items.append((lab, arch, make_compact(arch, 3)))
    base = make_compact("mobilenet_v2", 3)
    for frac in (60, 40, 25):
        p = os.path.join(res_dir, "runs", "resisc45", "seed0", f"MP_l1_P_{frac}.json")
        if os.path.exists(p):
            keep = [torch.tensor(k) for k in json.load(open(p))["stages"][0]["keep"]]
            items.append((f"MobileNetV2, {frac}%", f"mbv2_{frac}", prune_mbv2(base, keep)))
    torch.set_num_threads(1)
    rows = []
    for name, key, m in items:
        m = m.eval().float()
        meds = []
        for _ in range(5):
            meds.append(latency_ms(m, 3, 128, torch.device("cpu"), batch=1, threads=1, warm=20, iters=100)["median"])
        rows.append({"name": name, "key": key, "params_m": count_params(m) / 1e6, "macs_g": count_macs(m, 128, 3) / 1e9,
                     "cpu1_ms": statistics.median(meds), "cpu1_spread": max(meds) - min(meds)})
        print(f"{name:24s} {rows[-1]['params_m']:6.2f} M {rows[-1]['macs_g']:6.3f} G {rows[-1]['cpu1_ms']:6.2f} ms (spread {rows[-1]['cpu1_spread']:.2f})", flush=True)
    json.dump({"cpu": cpu_name(), "torch": torch.__version__, "platform": platform.platform(), "rows": rows}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results_replication", "latency_named_cpu.json"))
