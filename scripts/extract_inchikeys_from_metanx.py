"""Extract InChIKeys for human1 entries using MetaNetX mapping.

Creates data/resources/generated/human1_inchikeys_from_mnx.txt with one InChIKey per line.

Usage: run from repo root (the script finds paths relative to its location).
"""
from pathlib import Path
import json
import re
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
METANETX_DIR = REPO_ROOT / "data" / "resources" / "generated" / "metanetx"
MERGED_JSON = METANETX_DIR / "merged_metanetx.json"
HUMAN_CSV = REPO_ROOT / "data" / "resources" / "human1_identifiers.csv"
OUT_DIR = REPO_ROOT / "data" / "resources" / "generated"
OUT_FILE = OUT_DIR / "human1_inchikeys_from_mnx.txt"

mnx_re = re.compile(r"(MNXM\d+)", flags=re.I)


def load_merged():
    if MERGED_JSON.exists():
        with MERGED_JSON.open() as f:
            return json.load(f)
    # fallback: merge all json files in the metanetx dir
    merged = {}
    if not METANETX_DIR.exists():
        return merged
    for p in sorted(METANETX_DIR.glob("*.json")):
        try:
            with p.open() as f:
                d = json.load(f)
            if isinstance(d, dict):
                merged.update(d)
        except Exception:
            continue
    return merged


def build_mnx_to_inchikey_map(merged: dict) -> dict:
    """Return mapping MNXM -> set(inchikey_prefix/full)
    merged is expected to map inchikey -> payload but we search payload for MNXM ids."""
    mnx_map: dict[str, set] = {}
    for ik, payload in merged.items():
        try:
            text = json.dumps(payload)
        except Exception:
            text = str(payload)
        found = set(m.group(1).upper() for m in mnx_re.finditer(text))
        for mnx in found:
            mnx_map.setdefault(mnx, set()).add(ik)
    return mnx_map


def normalize_mnx(value: str) -> str | None:
    if not value or not isinstance(value, str):
        return None
    # try to find MNXM token anywhere
    m = mnx_re.search(value)
    if m:
        return m.group(1).upper()
    # tolerate last-segment formats like 'metanetx.chemical:MNXM123' or '...:mnxm123'
    if ":" in value:
        last = value.split(":")[-1].strip()
        if mnx_re.fullmatch(last) or last.upper().startswith("MNXM"):
            return last.upper()
    v = value.strip().upper()
    if v.startswith("MNXM"):
        return v
    return None


def main():
    if not HUMAN_CSV.exists():
        print(f"Human CSV not found at {HUMAN_CSV}")
        sys.exit(1)

    df = pd.read_csv(HUMAN_CSV, dtype=str)
    if "metanetx" not in df.columns:
        print("No 'metanetx' column in human CSV")
        sys.exit(1)

    merged = load_merged()
    if not merged:
        print("No merged MetaNetX data found (merged_metanetx.json or json files).")

    mnx_map = build_mnx_to_inchikey_map(merged)

    found_iks = set()
    for raw in df["metanetx"].dropna().unique():
        mnx = normalize_mnx(str(raw))
        if not mnx:
            continue
        iks = mnx_map.get(mnx)
        if iks:
            found_iks.update(iks)

    # Save full InChIKeys (as they appear as keys in merged). If you prefer prefixes, adjust here.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        for ik in sorted(found_iks):
            f.write(f"{ik}\n")

    print(f"Found {len(found_iks)} unique InChIKeys from MetaNetX mappings. Wrote to {OUT_FILE}")


if __name__ == "__main__":
    main()
