"""Download a small candidate pool of real sample images from the public Hugging Face datasets the paper uses.

    python -I fetch_images.py <out_dir>

RESISC45: 8 scenes for each of the 15 classes held out in partition 0. PlantVillage: a few leaf photographs.
"""
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

BASE = "https://datasets-server.huggingface.co"
OUT = sys.argv[1]

NOVEL0 = [0, 1, 7, 18, 20, 23, 24, 25, 27, 28, 30, 33, 40, 41, 42]
CLASSES = ["airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral", "church",
           "circular_farmland", "cloud", "commercial_area", "dense_residential", "desert", "forest", "freeway", "golf_course",
           "ground_track_field", "harbor", "industrial_area", "intersection", "island", "lake", "meadow", "medium_residential",
           "mobile_home_park", "mountain", "overpass", "palace", "parking_lot", "railway", "railway_station", "rectangular_farmland",
           "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential", "stadium", "storage_tank",
           "tennis_court", "terrace", "thermal_power_station", "wetland"]


def rows(ds, offset, length, split="train"):
    for attempt in range(4):
        try:
            r = requests.get(BASE + "/rows", params={"dataset": ds, "config": "default", "split": split, "offset": offset,
                                                     "length": length}, timeout=90)
            r.raise_for_status()
            return r.json()["rows"]
        except Exception as e:  # noqa: BLE001
            err = e
    raise err


def get_img(url):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            err = e
    raise err


def resisc():
    d = os.path.join(OUT, "resisc45")
    os.makedirs(d, exist_ok=True)
    meta = []

    def one(c):
        got = rows("jonathan-roberts1/NWPU-RESISC45", 700 * c, 100)
        assert all(r["row"]["label"] == c for r in got), f"class {c} offset mismatch"
        for k in (4, 15, 28, 41, 53, 66, 80, 93):
            im = get_img(got[k]["row"]["image"]["src"])
            name = f"{CLASSES[c]}_{k:02d}.jpg"
            im.save(os.path.join(d, name), quality=92)
            meta.append({"class": c, "name": CLASSES[c], "row": 700 * c + k, "file": name, "size": im.size})

    with ThreadPoolExecutor(5) as ex:
        list(ex.map(one, NOVEL0))
    json.dump(sorted(meta, key=lambda m: m["row"]), open(os.path.join(d, "meta.json"), "w"), indent=1)
    print("resisc45", len(meta))


def plant():
    d = os.path.join(OUT, "plantvillage")
    os.makedirs(d, exist_ok=True)
    meta = []
    for off in range(0, 43456, 2200):
        got = rows("BrandonFors/Plant-Diseases-PlantVillage-Dataset", off, 1)[0]["row"]
        im = get_img(got["image"]["src"])
        name = f"pv_{off:05d}.jpg"
        im.save(os.path.join(d, name), quality=92)
        meta.append({"row": off, "label": got["label"], "file": name, "size": im.size})
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print("plantvillage", len(meta))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    resisc()
    plant()
