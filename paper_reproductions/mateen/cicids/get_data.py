"""Extract CICIDS2017/clean_data.csv from the downloaded Datasets.zip.

Mateen authors' Drive folder ships everything in one zip
(`drive_listing/Datasets.zip`). We only need the CICIDS2017 file for
this subproject; the Kitsune subproject pulls its own files from the
same zip if needed.
"""
from __future__ import annotations
import os
import sys
import zipfile

ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "drive_listing", "Datasets.zip")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main() -> None:
    if not os.path.exists(ZIP):
        # Try the unrenamed .part file in case extraction is being run on
        # a partial download.
        cand = os.path.join(os.path.dirname(ZIP),
                             "Datasets.zipreesotew.part")
        if os.path.exists(cand):
            print(f"Found partial: {cand}", file=sys.stderr)
        raise FileNotFoundError(f"Expected {ZIP}; gdown may not be done.")

    os.makedirs(OUT, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        print(f"Datasets.zip contains {len(names)} entries")
        for n in names:
            print(f"  {n}")
        # Extract anything under CICIDS2017/
        wanted = [n for n in names if "CICIDS2017" in n
                  and n.endswith(".csv")]
        if not wanted:
            print("No CICIDS2017 CSV found; full listing above.",
                  file=sys.stderr)
            return
        for n in wanted:
            print(f"extracting {n}")
            z.extract(n, OUT)
    # Normalize to the canonical layout the data_loader expects.
    import shutil
    src_csv = None
    for root, _, files in os.walk(OUT):
        for fn in files:
            if fn == "clean_data.csv":
                src_csv = os.path.join(root, fn)
                break
        if src_csv:
            break
    if src_csv is None:
        print("clean_data.csv not present after extraction.", file=sys.stderr)
        return
    canonical = os.path.join(OUT, "CICIDS2017", "clean_data.csv")
    if os.path.abspath(src_csv) != os.path.abspath(canonical):
        os.makedirs(os.path.dirname(canonical), exist_ok=True)
        shutil.move(src_csv, canonical)
    print(f"OK: {canonical} ({os.path.getsize(canonical):,} bytes)")


if __name__ == "__main__":
    main()
