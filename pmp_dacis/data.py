"""Dataset loading (Hugging Face parquet mirrors), caching, class splits, GPU augmentation."""
import glob
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)
HF_SOURCES = {
    "plantvillage": "BrandonFors/Plant-Diseases-PlantVillage-Dataset",
    "plantdoc": "Project-AgML/plant_doc_classification",
    "eurosat_msi": "blanchon/EuroSAT_MSI",
    "resisc45": "jonathan-roberts1/NWPU-RESISC45",
}
# Band order of the Hugging Face parquet (checked from per-band means: index 8 is the water-vapour band B09,
# index 9 the cirrus band B10 with near-zero reflectance, index 12 the narrow NIR band B8A).
ES_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]
ES_RGB = [3, 2, 1]   # B04 (red), B03 (green), B02 (blue)


def download(root, names=("plantvillage", "plantdoc", "eurosat_msi", "resisc45")):
    from huggingface_hub import snapshot_download
    for n in names:
        d = os.path.join(root, n)
        if glob.glob(os.path.join(d, "data", "*.parquet")):
            continue
        snapshot_download(repo_id=HF_SOURCES[n], repo_type="dataset", local_dir=d, max_workers=8)


def ensure_data(root, cache_dir, cfg, keep_parquet=False, only=None):
    """Make sure the decoded arrays exist in cache_dir. Downloads a dataset only if its cache is missing and
    removes the raw parquet files afterwards (set keep_parquet=True to keep them)."""
    import shutil
    jobs = (("plantvillage", "plantvillage", load_plantvillage), ("plantdoc", "plantdoc", load_plantdoc),
            ("eurosat", "eurosat_msi", None), ("resisc45", "resisc45", None))
    for name, hf, loader in jobs:
        if only is not None and name not in only:
            continue
        tag = cache_tag(name, cfg)
        if all(os.path.exists(os.path.join(cache_dir, f"{tag}_{x}")) for x in ("X.npy", "y.npy", "names.json")):
            continue
        download(root, (hf,))
        if name == "plantvillage":
            load_plantvillage(root, cache_dir, cfg["pv_res"], cfg.get("limit"))
        elif name == "plantdoc":
            load_plantdoc(root, cache_dir, cfg["pv_res"], cfg.get("limit"))
        elif name == "resisc45":
            load_resisc45(root, cache_dir, cfg["rs_res"], cfg.get("limit"))
        else:
            load_eurosat(root, cache_dir, cfg.get("limit"))
        if not keep_parquet:
            shutil.rmtree(os.path.join(root, hf), ignore_errors=True)


def _hf_names(parquet_path, col="label"):
    import pyarrow.parquet as pq
    md = pq.read_schema(parquet_path).metadata
    info = json.loads(md[b"huggingface"])["info"]["features"][col]
    nm = info["names"]
    return [nm[str(i)] if isinstance(nm, dict) else nm[i] for i in range(len(nm))]


def _resize(im, res, crop):
    from PIL import Image
    if crop:  # resize the shorter side to res, then centre-crop
        w, h = im.size
        s = res / min(w, h)
        im = im.resize((max(res, round(w * s)), max(res, round(h * s))), Image.BILINEAR)
        w, h = im.size
        left, top = (w - res) // 2, (h - res) // 2
        return im.crop((left, top, left + res, top + res))
    return im.resize((res, res), Image.BILINEAR)


def _decode(b, res, crop):
    from PIL import Image
    im = Image.open(io.BytesIO(b)).convert("RGB")
    return np.asarray(_resize(im, res, crop))


def _stride(files, limit):
    """Keep every stride-th row so that a `limit`-sized subset still covers every class."""
    import pyarrow.parquet as pq
    if not limit:
        return 1
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    return max(1, total // limit)


def _load_rgb_parquet(files, res, crop, limit=None):
    import pyarrow.parquet as pq
    Xs, ys = [], []
    n_threads = min(16, os.cpu_count() or 4)
    stride, row0 = _stride(files, limit), 0
    with ThreadPoolExecutor(n_threads) as ex:
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=512, columns=["image", "label"]):
                keep = (np.arange(row0, row0 + len(batch)) % stride) == 0
                row0 += len(batch)
                if not keep.any():
                    continue
                imgs = [d for d, k in zip(batch.column("image").to_pylist(), keep) if k]
                labs = batch.column("label").to_numpy()[keep]
                arr = list(ex.map(lambda d: _decode(d["bytes"], res, crop), imgs))
                Xs.append(np.stack(arr))
                ys.append(labs.astype(np.int64))
    X = np.concatenate(Xs).transpose(0, 3, 1, 2)  # N,3,H,W uint8
    return np.ascontiguousarray(X), np.concatenate(ys)


def _cached(cache_dir, tag, builder):
    os.makedirs(cache_dir, exist_ok=True)
    fx, fy, fn = (os.path.join(cache_dir, f"{tag}_{s}") for s in ("X.npy", "y.npy", "names.json"))
    if all(os.path.exists(p) for p in (fx, fy, fn)):
        return np.load(fx), np.load(fy), json.load(open(fn))
    X, y, names = builder()
    np.save(fx, X)
    np.save(fy, y)
    json.dump(names, open(fn, "w"))
    return X, y, names


def cache_tag(name, cfg):
    lim = f"_lim{cfg['limit']}" if cfg.get("limit") else ""
    return {"plantvillage": f"pv_{cfg['pv_res']}{lim}", "plantdoc": f"pd_{cfg['pv_res']}{lim}", "eurosat": f"es{lim}",
            "resisc45": f"rs_{cfg.get('rs_res', 128)}{lim}"}[name]


def load_plantvillage(root, cache_dir, res=128, limit=None):
    def build():
        files = sorted(glob.glob(os.path.join(root, "plantvillage", "data", "*.parquet")))
        names = _hf_names(files[0])
        X, y = _load_rgb_parquet(files, res, crop=False, limit=limit)
        return X, y, names

    X, y, names = _cached(cache_dir, f"pv_{res}" + (f"_lim{limit}" if limit else ""), build)
    return DS("plantvillage", torch.from_numpy(X), torch.from_numpy(y), names, "rgb8", res)


def load_plantdoc(root, cache_dir, res=128, limit=None):
    def build():
        files = sorted(glob.glob(os.path.join(root, "plantdoc", "data", "*.parquet")))
        names = _hf_names(files[0])
        X, y = _load_rgb_parquet(files, res, crop=True, limit=limit)
        return X, y, names

    X, y, names = _cached(cache_dir, f"pd_{res}" + (f"_lim{limit}" if limit else ""), build)
    return DS("plantdoc", torch.from_numpy(X), torch.from_numpy(y), names, "rgb8", res)


# The parquet file carries no class names. Label i is the i-th name below; checked against the image file names
# inside the parquet (airplane_001.jpg ... wetland_700.jpg, 700 per class, in this order).
RESISC45_CLASSES = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral", "church",
    "circular_farmland", "cloud", "commercial_area", "dense_residential", "desert", "forest", "freeway", "golf_course",
    "ground_track_field", "harbor", "industrial_area", "intersection", "island", "lake", "meadow", "medium_residential",
    "mobile_home_park", "mountain", "overpass", "palace", "parking_lot", "railway", "railway_station", "rectangular_farmland",
    "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "sparse_residential", "stadium", "storage_tank",
    "tennis_court", "terrace", "thermal_power_station", "wetland",
]


def load_resisc45(root, cache_dir, res=128, limit=None):
    """NWPU-RESISC45: 31,500 aerial scenes, 45 classes, 700 per class, 256x256 RGB, resized to res x res."""
    def build():
        files = sorted(glob.glob(os.path.join(root, "resisc45", "data", "*.parquet")))
        X, y = _load_rgb_parquet(files, res, crop=False, limit=limit)
        assert int(y.max()) < len(RESISC45_CLASSES), "unexpected label in RESISC45"
        return X, y, list(RESISC45_CLASSES)

    X, y, names = _cached(cache_dir, f"rs_{res}" + (f"_lim{limit}" if limit else ""), build)
    return DS("resisc45", torch.from_numpy(X), torch.from_numpy(y), names, "rgb8", res)


def load_eurosat(root, cache_dir, limit=None):
    import pyarrow.parquet as pq

    def build():
        files = sorted(glob.glob(os.path.join(root, "eurosat_msi", "data", "*.parquet")))
        names = _hf_names(files[0])
        Xs, ys = [], []
        stride, row0 = _stride(files, limit), 0
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
                n = len(batch)
                keep = (np.arange(row0, row0 + n) % stride) == 0
                row0 += n
                if not keep.any():
                    continue
                flat = batch.column("image")
                for _ in range(3):  # array3d is stored as nested lists
                    flat = flat.flatten()
                a = flat.to_numpy(zero_copy_only=False).astype(np.uint16).reshape(n, 64, 64, 13)[keep]
                Xs.append(a.transpose(0, 3, 1, 2))
                ys.append(batch.column("label").to_numpy().astype(np.int64)[keep])
        return np.ascontiguousarray(np.concatenate(Xs)), np.concatenate(ys), names

    X, y, names = _cached(cache_dir, "es" + (f"_lim{limit}" if limit else ""), build)
    return DS("eurosat", torch.from_numpy(X), torch.from_numpy(y), names, "ms16", 64)


class DS:
    """In-memory image dataset. X is uint8 [N,3,H,W] (rgb8) or uint16 [N,13,H,W] (ms16)."""

    def __init__(self, name, X, y, class_names, kind, res, bands=None):
        self.name, self.X, self.y, self.class_names, self.kind, self.res = name, X, y, class_names, kind, res
        self.bands = bands  # channel subset for ms16 (None = all)
        self.mean = self.std = None
        self._by = None

    @property
    def in_ch(self):
        if self.kind == "rgb8":
            return 3
        return len(self.bands) if self.bands is not None else self.X.shape[1]

    def with_bands(self, bands):
        d = DS(self.name, self.X, self.y, self.class_names, self.kind, self.res, bands)
        d._by = self._by
        return d

    def idx_by_class(self):
        if self._by is None:
            y = self.y.numpy()
            self._by = {int(c): torch.from_numpy(np.where(y == c)[0]) for c in np.unique(y)}
        return self._by

    def set_norm(self, idx, n=3000):
        """Per-channel mean/std for ms16 data, from (at most n) images of `idx`."""
        if self.kind != "ms16":
            return
        sel = idx[torch.randperm(len(idx))[:n]]
        x = self.X[sel].float()
        if self.bands is not None:
            x = x[:, self.bands]
        self.mean = x.mean((0, 2, 3)).view(1, -1, 1, 1)
        self.std = x.std((0, 2, 3)).view(1, -1, 1, 1) + 1e-6

    def fetch(self, idx, device, train=False, res=None):
        """Float, normalised batch on `device`. train=True applies augmentation."""
        x = self.X[idx].to(device, non_blocking=True)
        res = res or self.res
        if self.kind == "rgb8":
            x = x.float() / 255.0
            if train:
                B = x.shape[0]
                b = torch.empty(B, 1, 1, 1, device=device).uniform_(0.85, 1.15)
                c = torch.empty(B, 1, 1, 1, device=device).uniform_(0.85, 1.15)
                m = x.mean((1, 2, 3), keepdim=True)
                x = ((x - m) * c + m) * b
            mean = torch.tensor(IMNET_MEAN, device=device).view(1, 3, 1, 1)
            std = torch.tensor(IMNET_STD, device=device).view(1, 3, 1, 1)
            x = (x - mean) / std
        else:
            x = x.float()
            if self.bands is not None:
                x = x[:, self.bands]
            x = (x - self.mean.to(device)) / self.std.to(device)
        if train:
            x = gpu_augment(x, res)
        elif x.shape[-1] != res:
            x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=x.shape[-1] > res)
        return x.contiguous()


def gpu_augment(x, out_res, scale=(0.6, 1.0)):
    """Random zoom-in crop + horizontal/vertical flips, done with one affine grid sample."""
    B = x.shape[0]
    dev = x.device
    s = torch.empty(B, device=dev).uniform_(*scale).sqrt()
    tx = (1 - s) * (torch.rand(B, device=dev) * 2 - 1)
    ty = (1 - s) * (torch.rand(B, device=dev) * 2 - 1)
    fx = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    fy = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = s * fx
    theta[:, 1, 1] = s * fy
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B, x.shape[1], out_res, out_res), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def subsample(ds, n):
    """About n images, spread evenly over the classes (at least 40 per class), so that a tiny test run can still sample
    episodes from every class. Used by the micro profile only."""
    y = ds.y.numpy()
    classes = np.unique(y)
    k = max(40, n // len(classes))
    sel = []
    for c in classes:
        idx = np.where(y == c)[0]
        sel.append(idx[:: max(1, len(idx) // k)][:k])
    sel = torch.from_numpy(np.sort(np.concatenate(sel)))
    return DS(ds.name, ds.X[sel], ds.y[sel], ds.class_names, ds.kind, ds.res, ds.bands)


def class_split(n_classes, seed, n_base, n_val):
    """Random disjoint base / val / novel class lists for a seed."""
    rng = np.random.RandomState(1000 + seed)
    perm = rng.permutation(n_classes).tolist()
    return sorted(perm[:n_base]), sorted(perm[n_base:n_base + n_val]), sorted(perm[n_base + n_val:])
