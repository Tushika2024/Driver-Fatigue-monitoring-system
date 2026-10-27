"""Helper utility to inspect, validate, and prepare Kaggle UTA-RLDD datasets.

Usage:
    python prepare_kaggle.py --input /path/to/downloaded_kaggle_data
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def analyze_dataset(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        print(f"[ERROR] Directory '{root}' does not exist.")
        return

    print(f"\n==========================================")
    print(f" Scanning Kaggle Dataset: {root.resolve()}")
    print(f"==========================================")

    all_vids = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    print(f"Found {len(all_vids)} video files in total.")

    # Group by potential subjects and labels
    # Standard format: .../<subject>/<label>.<ext> where label is '0', '5', or '10'
    standard_vids = []
    other_vids = []

    for v in all_vids:
        stem = v.stem.lower()
        parent = v.parent.name
        if stem in {"0", "10", "5"}:
            standard_vids.append((parent, stem, v))
        else:
            # Check if name contains alert/drowsy
            if "alert" in stem or stem.startswith("0_"):
                other_vids.append((parent, "0", v))
            elif "drowsy" in stem or stem.startswith("10_"):
                other_vids.append((parent, "10", v))
            elif "low" in stem or stem.startswith("5_"):
                other_vids.append((parent, "5", v))
            else:
                other_vids.append((parent, stem, v))

    print(f"  - Clean standard naming (<subject>/<label>): {len(standard_vids)} videos")
    if other_vids:
        print(f"  - Non-standard naming: {len(other_vids)} videos")

    subjects = {}
    for sub, lbl, path in standard_vids:
        if lbl in {"0", "10"}:
            subjects.setdefault(sub, {})[lbl] = path

    print(f"\nUsable Subjects for Binary Classification (0=alert, 10=drowsy): {len(subjects)}")
    complete_pairs = sum(1 for s, lbls in subjects.items() if "0" in lbls and "10" in lbls)
    print(f"  - Subjects with BOTH alert (0) and drowsy (10): {complete_pairs}")

    # Ensure project data and models directories exist
    backend_dir = Path(__file__).resolve().parent
    data_dir = backend_dir / "data"
    models_dir = backend_dir / "models"
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    print(f"\n[NEXT STEPS]")
    print(f"1. Extract per-frame features:")
    print(f"   fatigue extract --videos \"{root.resolve()}\" --out data/features.csv --jobs 4")
    print(f"2. Generate metadata template (for glasses/facial hair fairness audit):")
    print(f"   fatigue subjects-template --features data/features.csv --out data/subjects.csv")
    print(f"3. Train production model:")
    print(f"   fatigue train --features data/features.csv --out models/bundle.joblib --jobs 4")
    print(f"4. (Optional) Run offline LOSO evaluation & benchmarks:")
    print(f"   fatigue evaluate --features data/features.csv --subjects data/subjects.csv --outdir results --jobs 4")
    print(f"\nOnce 'models/bundle.joblib' is created, start the API (`python -m fatigue_api`) and it will automatically load it!")


def main():
    parser = argparse.ArgumentParser(description="Inspect and prepare Kaggle dataset for Night Watch")
    parser.add_argument("--input", "-i", required=True, help="Path to unzipped Kaggle dataset")
    args = parser.parse_args()
    analyze_dataset(args.input)


if __name__ == "__main__":
    main()
