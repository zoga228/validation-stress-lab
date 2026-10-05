"""Original synthetic benchmarks; no real people, records, or competition data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def grouped(seed=42, groups=400, repeats=30):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(groups), repeats)
    latent = rng.integers(0, 2, size=groups)
    flip = rng.random(len(ids)) < .08
    return pd.DataFrame({"group_id": ids.astype(str), "sensor_noise": rng.normal(size=len(ids)), "target": np.logical_xor(latent[ids], flip).astype(int)})


def missingness(seed=42, n=6000, shifted=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    score = 1.6*x[:, 0] - .8*x[:, 1] + .5*x[:, 2]*x[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    # An apparent signal reverses at deployment. This is a controlled stress test.
    correlated = (y + rng.normal(0, .4, n)) if not shifted else ((1-y) + rng.normal(0, .4, n))
    missing = rng.random(n) < np.where(y == 1, .3 if not shifted else .7, .1 if not shifted else .4)
    x[missing, 0] = np.nan
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(6)])
    df["spurious"] = correlated
    df["target"] = y
    return df


def regression(seed=42, n=4000, shifted=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(1.8 if shifted else 0, 1, n)
    sigma = .3 + .4*np.abs(x)
    y = np.sin(2*x) + .5*x + sigma*rng.normal(size=n)
    return pd.DataFrame({"x": x, "target": y})


def export(root):
    root = Path(root)
    specs = {
        "grouped-validation": {"all.csv": grouped()},
        "missingness-shift": {"train.csv": missingness(42), "iid-test.csv": missingness(43, 3000), "shift-test.csv": missingness(44, 3000, True)},
        "conformal-shift": {"train.csv": regression(42), "calibration.csv": regression(43, 2000), "iid-test.csv": regression(44, 3000), "shift-test.csv": regression(45, 3000, True)},
    }
    for name, files in specs.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename, frame in files.items():
            frame.to_csv(folder / filename, index=False)
        info = {"synthetic": True, "license": "CC0-1.0", "generator": "make_data.py", "version": "1.0", "files": {k: {"rows": len(v), "columns": list(v)} for k, v in files.items()}, "purpose": "Controlled experiments about validation assumptions; not evidence about a real population"}
        (folder / "data-card.json").write_text(json.dumps(info, indent=2))
        print(name, info["files"])


if __name__ == "__main__":
    export(Path(__file__).resolve().parent / "datasets")
