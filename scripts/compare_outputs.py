"""Compare pipeline outputs between two folders.

Usage (from repo root):
    python scripts/compare_outputs.py reference .        # old snapshot vs fresh main.py run
    python scripts/compare_outputs.py <dirA> <dirB>      # e.g. notebook run vs main.py run

Reports, per file: (a) shape / key changes, (b) NaN-pattern changes, (c) value diffs (6 dp).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    "weekly_board_enriched.csv": ["player_id"],
    "predicted_fixtures_GW.csv": ["home_team_name", "away_team_name"],
}
DECIMALS = 6


def compare(old_path, new_path, keys):
    print(f"\n=== {old_path.name} ===")
    old = pd.read_csv(old_path, encoding="utf-8-sig")
    new = pd.read_csv(new_path, encoding="utf-8-sig")
    clean = True

    # (a) shape / columns / keys
    print(f"rows: old={len(old)} new={len(new)} | cols: old={old.shape[1]} new={new.shape[1]}")
    only_old_cols = sorted(set(old.columns) - set(new.columns))
    only_new_cols = sorted(set(new.columns) - set(old.columns))
    if only_old_cols or only_new_cols:
        clean = False
        print(f"  [a] columns only in old: {only_old_cols}")
        print(f"  [a] columns only in new: {only_new_cols}")
    if list(old.columns) != list(new.columns) and not (only_old_cols or only_new_cols):
        clean = False
        print("  [a] same columns, different order")

    old_i = old.set_index(keys)
    new_i = new.set_index(keys)
    if old_i.index.duplicated().any() or new_i.index.duplicated().any():
        print("  [a] WARNING: duplicate keys present; comparing first occurrence")
        old_i = old_i[~old_i.index.duplicated()]
        new_i = new_i[~new_i.index.duplicated()]
    only_old = old_i.index.difference(new_i.index)
    only_new = new_i.index.difference(old_i.index)
    if len(only_old) or len(only_new):
        clean = False
        print(f"  [a] keys only in old ({len(only_old)}): {list(only_old)[:15]}")
        print(f"  [a] keys only in new ({len(only_new)}): {list(only_new)[:15]}")

    common = old_i.index.intersection(new_i.index)
    cols = [c for c in old_i.columns if c in new_i.columns]
    o, n = old_i.loc[common, cols], new_i.loc[common, cols]

    nan_changes, value_changes = {}, {}
    for c in cols:
        on, nn = o[c].isna(), n[c].isna()
        nan_diff = on != nn
        if nan_diff.any():
            nan_changes[c] = int(nan_diff.sum())
        both = ~on & ~nn
        if pd.api.types.is_numeric_dtype(o[c]) and pd.api.types.is_numeric_dtype(n[c]):
            diff = ~np.isclose(o[c][both].round(DECIMALS), n[c][both].round(DECIMALS),
                               rtol=0, atol=10 ** -DECIMALS)
            mask = pd.Series(diff, index=o[c][both].index)
        else:
            mask = o[c][both].astype(str) != n[c][both].astype(str)
        if mask.any():
            ex = mask[mask].index[:3]
            value_changes[c] = (int(mask.sum()),
                                [(k, o.at[k, c], n.at[k, c]) for k in ex])

    if nan_changes:
        clean = False
        print(f"  [b] NaN-pattern changes (col: rows): {nan_changes}")
    if value_changes:
        clean = False
        print(f"  [c] value diffs on {len(value_changes)} columns:")
        for c, (cnt, ex) in sorted(value_changes.items(), key=lambda kv: -kv[1][0]):
            print(f"      {c}: {cnt} rows, e.g. {ex}")
    if clean:
        print("  IDENTICAL (to 6 dp)")
    return clean


def main(old_dir, new_dir):
    ok = True
    for fname, keys in FILES.items():
        ok &= compare(Path(old_dir) / fname, Path(new_dir) / fname, keys)
    print("\nALL MATCH" if ok else "\nDIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    a, b = (sys.argv[1:3] if len(sys.argv) >= 3 else ("reference", "."))
    sys.exit(main(a, b))
