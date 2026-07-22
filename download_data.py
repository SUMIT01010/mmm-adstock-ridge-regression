"""
Regression project — data download.  # --- scaffolding ---

Robyn Sample Data (Meta OSS), 208 weeks, 5 media channels, weekly revenue.

The R package only ships this table as a binary `.RData` file
(`facebookexperimental/Robyn` `R/data/dt_simulated_weekly.RData`) — not directly usable
without R/pyreadr. The Robyn *Python* port (same repo, `python/src/robyn/tutorials/
resources/dt_simulated_weekly.csv`) ships the identical simulated panel as a plain CSV.
Verified live before building anything else (2026-07-17): 208 rows, no auth wall, public
raw.githubusercontent.com URL, 200 OK. This is the same dataset Robyn's own R and Python
tutorials use — not a substitute dataset, just a differently-serialized copy of it.

Run:  .venv/bin/python download_data.py
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import requests

from config import N_WEEKS_EXPECTED, PROC, RAW, ROBYN_CSV_URL

MANIFEST = PROC / "data_manifest.json"


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    dest = RAW / "dt_simulated_weekly.csv"
    if not dest.exists():
        r = requests.get(ROBYN_CSV_URL, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"downloaded -> {dest} ({len(r.content):,} bytes)")
    else:
        print(f"already present -> {dest}")

    df = pd.read_csv(dest)
    assert len(df) == N_WEEKS_EXPECTED, (
        f"expected {N_WEEKS_EXPECTED} weekly rows, got {len(df)}")
    assert "DATE" in df.columns and "revenue" in df.columns

    manifest = {
        "dt_simulated_weekly": {
            "file": str(dest.relative_to(RAW.parent.parent)),
            "sha256": _sha256(dest),
            "source": ROBYN_CSV_URL,
            "shape": list(df.shape),
            "date_min": str(df["DATE"].min()),
            "date_max": str(df["DATE"].max()),
        }
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Robyn simulated weekly panel: {df.shape[0]} weeks x {df.shape[1]} cols "
          f"({df['DATE'].min()} -> {df['DATE'].max()}) -> manifest {MANIFEST}")


if __name__ == "__main__":
    main()
