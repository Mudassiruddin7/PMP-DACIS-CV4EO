"""Cut the candidate pool from fetch_images.py down to the aerial and leaf pictures the page uses.

    python -I tools/make_images.py <pool_dir> <assets/img dir>

All pictures are samples from the public datasets the study uses (NWPU-RESISC45, PlantVillage); the page credits them.
"""
import os
import sys

from PIL import Image

POOL, OUT = sys.argv[1], sys.argv[2]

# --- aerial scenes: one per held-out class of partition 0, for the opening contact sheet (kept at the source 256 px)
HERO = ["airplane_53", "airport_15", "church_15", "industrial_area_80", "island_41", "medium_residential_15",
        "mobile_home_park_15", "mountain_41", "palace_41", "parking_lot_28", "railway_station_04", "roundabout_41",
        "storage_tank_15", "tennis_court_15", "terrace_80"]
os.makedirs(f"{OUT}/aerial", exist_ok=True)
for n in HERO:
    Image.open(f"{POOL}/resisc45/{n}.jpg").convert("RGB").save(f"{OUT}/aerial/{n}.jpg", quality=88, optimize=True)

# --- one 5-way episode: five support scenes and one query for each of five held-out classes (192 px)
EPISODE = {
    "airplane": ["04", "28", "53", "80", "93", "15"],
    "industrial_area": ["04", "15", "28", "41", "80", "93"],
    "island": ["04", "28", "41", "53", "66", "93"],
    "roundabout": ["15", "28", "41", "53", "66", "93"],
    "storage_tank": ["04", "15", "28", "53", "66", "93"],
}
os.makedirs(f"{OUT}/episode", exist_ok=True)
for cls, ks in EPISODE.items():
    for i, k in enumerate(ks):
        im = Image.open(f"{POOL}/resisc45/{cls}_{k}.jpg").convert("RGB").resize((192, 192), Image.LANCZOS)
        im.save(f"{OUT}/episode/{cls}_{'q' if i == 5 else 's' + str(i + 1)}.jpg", quality=84, optimize=True)

# (the Sentinel-2 band pictures of the bands section now come from tools/make_scene.py, at full resolution)

# --- three leaves for the ground-level control (PlantVillage, 256 px)
os.makedirs(f"{OUT}/leaf", exist_ok=True)
for row, name in ((8700, "grape_black_rot"), (17400, "peach_bacterial_spot"), (31900, "tomato_late_blight")):
    Image.open(f"{POOL}/plantvillage/pv_{row:05d}.jpg").convert("RGB").save(f"{OUT}/leaf/{name}.jpg", quality=88, optimize=True)

total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs)
print("images written,", round(total / 1024), "KB in total")
