"""Cut one full-resolution Sentinel-2 scene into the pictures and spectra of the bands section.

    python tools/make_scene.py <site root>

Needs rasterio and a network connection. Reads the public Level-2A COG files of one scene through the Earth Search catalogue
(no account needed) and writes to <site root>/assets/img/scene/ and <site root>/assets/js/scene.js.

Scene: S2B_32TNQ_20240711_1_L2A (11 July 2024), a 900 x 900 pixel window at 10 m, so 9 km on a side.
Contains modified Copernicus Sentinel data (2024).

Level-2A has no cirrus band (B10), so twelve of the thirteen bands are shown. The 20 m and 60 m bands are resampled (cubic) to the
10 m grid, so the blur of B01 and B09 is real: those bands have 60 m pixels.
"""
import json
import os
import sys

import numpy as np
import requests
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as win_bounds, from_bounds

ROOT = sys.argv[1]
OUT = os.path.join(ROOT, "assets", "img", "scene")
os.makedirs(OUT, exist_ok=True)

ITEM = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2B_32TNQ_20240711_1_L2A"
COL, ROW, N = 5130, 815, 900
BANDS = [("B01", "coastal", 443), ("B02", "blue", 490), ("B03", "green", 560), ("B04", "red", 665), ("B05", "rededge1", 705),
         ("B06", "rededge2", 740), ("B07", "rededge3", 783), ("B08", "nir", 833), ("B8A", "nir08", 865), ("B09", "nir09", 945),
         ("B11", "swir16", 1610), ("B12", "swir22", 2190)]

item = requests.get(ITEM, timeout=60).json()
base = item["properties"].get("s2:processing_baseline")
print("scene", item["id"], "baseline", base, "cloud", item["properties"]["eo:cloud_cover"])

env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES")
refl = {}
with env:
    with rasterio.open(item["assets"]["blue"]["href"]) as ref:
        geo = win_bounds(Window(COL, ROW, N, N), ref.transform)
    for bid, key, wl in BANDS:
        with rasterio.open(item["assets"][key]["href"]) as ds:
            w = from_bounds(*geo, transform=ds.transform)
            a = ds.read(1, window=w, out_shape=(N, N), resampling=Resampling.cubic).astype(np.float32)
        # These COG files hold reflectance x 10000. The catalogue metadata lists a baseline-4 offset of -0.1, but the stored values
        # carry none: the 1st percentile of B08 is DN 298, which would be negative reflectance if the offset were applied.
        r = a / 10000.0
        refl[bid] = np.clip(r, 0, 1.2)
        print(bid, "res", round(ds.res[0]), "reflectance 2-98%", np.percentile(r[r > 0], [2, 98]).round(3))
    with rasterio.open(item["assets"]["visual"]["href"]) as ds:
        tci = np.moveaxis(ds.read(window=Window(COL, ROW, N, N)), 0, -1)

valid = (tci.sum(-1) > 0)


def stretch(x, lo=2, hi=98):
    v = x[valid]
    l, h = np.percentile(v, [lo, hi])
    return np.clip((x - l) / max(h - l, 1e-6), 0, 1)


def save(arr01, name, size, q=90, mode=None):
    im = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8), mode)
    if size != N:
        im = im.resize((size, size), Image.LANCZOS)
    im.save(os.path.join(OUT, name), quality=q, optimize=True)


# the main picture: ESA's own true-colour rendering at full resolution
Image.fromarray(tci).save(os.path.join(OUT, "true_900.jpg"), quality=92, optimize=True)

# one grey picture per band, contrast-stretched on its own
for bid, key, wl in BANDS:
    save(stretch(refl[bid]), bid + ".jpg", 480, 88, "L")

rgb = lambda r, g, b, f=stretch: np.stack([f(refl[r]), f(refl[g]), f(refl[b])], -1)
save(rgb("B04", "B03", "B02", lambda x: np.clip(x * 3.0, 0, 1) ** (1 / 1.15)), "comp_true.jpg", 640)
save(rgb("B08", "B04", "B03"), "comp_nir.jpg", 640)
save(rgb("B12", "B8A", "B04"), "comp_swir.jpg", 640)

# NDVI on a ramp that runs from water, through bare ground, to dense crops
ndvi = (refl["B08"] - refl["B04"]) / np.maximum(refl["B08"] + refl["B04"], 1e-6)
stops = [(-0.3, (109, 167, 236)), (0.05, (236, 230, 211)), (0.45, (159, 209, 159)), (0.85, (0, 99, 0))]
xs = [s[0] for s in stops]
img = np.stack([np.interp(ndvi, xs, [s[1][c] for s in stops]) for c in range(3)], -1) / 255.0
save(img, "comp_ndvi.jpg", 640)

# six points on the ground, chosen near a guess by what the pixels are (darkest in NIR for water, and so on)
yy, xx = np.mgrid[0:N, 0:N]


def near(cx, cy, rad):
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= rad ** 2


def pick(cx, cy, rad, score):
    m = near(cx, cy, rad) & valid
    s = np.where(m, score, np.nan)
    i = np.nanargmax(s)
    return int(i % N), int(i // N)


nir, red, grn, blu, swir = refl["B08"], refl["B04"], refl["B03"], refl["B02"], refl["B11"]
pins = [
    ("Po river", "water", "#2a78d6", pick(430, 205, 14, -nir)),
    ("sandbar", "bare sand in the river", "#cdb26b", pick(497, 210, 9, red)),
    ("dark crop", "dense green crop", "#2e8b3c", pick(385, 595, 16, ndvi)),
    ("stubble", "harvested field", "#b7794a", pick(440, 870, 14, red - nir * 0.0 - grn * 0.0)),
    ("old town", "red clay roofs", "#c0603a", pick(330, 285, 14, red - blu)),
    ("warehouse", "white roof", "#8a8a85", pick(828, 459, 7, red + grn + blu)),
]
out_pins = []
for name, what, color, (px, py) in pins:
    m = near(px, py, 4) & valid          # a patch of about 9 x 9 pixels, 90 m across
    spec = [round(float(refl[b][m].mean()), 4) for b, _, _ in BANDS]
    out_pins.append({"name": name, "what": what, "color": color, "x": round(px / N, 4), "y": round(py / N, 4), "spectrum": spec})
    print(f"{name:10s} at ({px},{py})  B04 {spec[3]:.3f}  B08 {spec[7]:.3f}  B12 {spec[11]:.3f}")

data = {"scene": item["id"], "date": item["properties"]["datetime"][:10], "size_px": N, "pixel_m": 10,
        "bands": [{"id": b, "wl": wl} for b, _, wl in BANDS], "pins": out_pins, "patch_px": 64,
        "ndvi_stops": [s[0] for s in stops]}
with open(os.path.join(ROOT, "assets", "js", "scene.js"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write("/* Generated by tools/make_scene.py from one Sentinel-2 Level-2A scene. Contains modified Copernicus Sentinel data (2024). */\n")
    fh.write("window.SCENE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
print("wrote", len(os.listdir(OUT)), "images,", round(total / 1024), "KB")
