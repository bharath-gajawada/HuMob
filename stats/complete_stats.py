# stats_extractor.py
# Chunked scanner that produces per-city JSON summaries and optional per-user CSVs.
# Safe for large files; set COMPUTE_DISTINCT=True only if you have memory.

import os
import json
import math
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ----------------- CONFIG -----------------
DATA_DIR = "data"   # change to your path
CITIES = ["A","B","C","D"]                      # list of cities to scan
COLUMNS = ["uid","d","t","x","y"]
DTYPES  = {"uid":"int32","d":"int16","t":"int16","x":"int16","y":"int16"}
CHUNK_SIZE = 500_000
MASK_VALUE = 999

# Target UID ranges (inclusive)
TARGET_RANGES = {
    "A": (147001, 150000),
    "B": (27001, 30000),
    "C": (22001, 25000),
    "D": (17001, 20000),
}

# Controls
COMPUTE_DISTINCT_CELLS = False   # set True to compute per-user distinct-cell counts (may use lots of memory)
SAVE_PER_USER_CSV = True        # set False to skip writing per-user CSV (smaller output)
OUTPUT_DIR = "./stats_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------- AUX helpers -----------------
def in_target(uid, city):
    lo, hi = TARGET_RANGES[city]
    return lo <= uid <= hi

def describe_arr(a):
    a = np.array(a, dtype=float) if len(a)>0 else np.array([])
    if a.size == 0:
        return {"count":0, "min":None, "max":None, "mean":None, "median":None, "p10":None, "p25":None, "p75":None, "p90":None}
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a,10)),
        "p25": float(np.percentile(a,25)),
        "p75": float(np.percentile(a,75)),
        "p90": float(np.percentile(a,90))
    }


# ----------------- MAIN per-city scanner -----------------
def scan_city(city):
    path = os.path.join(DATA_DIR, f"city_{city}_challengedata.csv")
    print(f"\nScanning city {city} from {path}")

    # global counters
    total_rows = 0
    unique_uids = set()
    rows_by_day = Counter()
    masked_by_day = Counter()
    unmasked_by_day = Counter()
    rows_by_slot = Counter()  # optional: counts by t
    # per-uid aggregates
    # For memory-efficiency keep integer counters and optional sets for distinct cells
    uid_agg = defaultdict(lambda: {
        "total":0, "train":0, "test_total":0, "test_unmasked":0, "test_masked":0
    })
    if COMPUTE_DISTINCT_CELLS:
        uid_cells = defaultdict(set)

    # also track target / non-target sets
    target_users_present = set()
    non_target_users_present = set()

    # chunked read
    reader = pd.read_csv(path, usecols=COLUMNS, dtype=DTYPES, chunksize=CHUNK_SIZE, compression="gzip")
    for chunk in tqdm(reader, desc=f"Reading {city}"):
        # prepare flags
        chunk["is_masked"] = ((chunk["x"] == MASK_VALUE) & (chunk["y"] == MASK_VALUE))
        total_rows += len(chunk)
        unique_uids.update(chunk["uid"].unique().tolist())

        # per-day counters
        for d, g in chunk.groupby("d"):
            rows_by_day[int(d)] += len(g)
            masked_by_day[int(d)] += int(g["is_masked"].sum())
            unmasked_by_day[int(d)] += int((~g["is_masked"]).sum())

        # per-slot
        for t, cnt in chunk["t"].value_counts().items():
            rows_by_slot[int(t)] += int(cnt)

        # per-uid updates (vectorized)
        # We'll group by uid in chunk and update uid_agg
        for uid, g in chunk.groupby("uid"):
            uid = int(uid)
            total = len(g)
            train_mask = (g["d"] <= 60)
            test_mask = (g["d"] >= 61) & (g["d"] <= 75)
            train_cnt = int(train_mask.sum())
            test_total = int(test_mask.sum())
            test_masked = int(((g["is_masked"]) & test_mask).sum())
            test_unmasked = int(((~g["is_masked"]) & test_mask).sum())

            st = uid_agg[uid]
            st["total"] += total
            st["train"] += train_cnt
            st["test_total"] += test_total
            st["test_masked"] += test_masked
            st["test_unmasked"] += test_unmasked

            # distinct cells per user (optional)
            if COMPUTE_DISTINCT_CELLS:
                for x,y in zip(g["x"], g["y"]):
                    if x == MASK_VALUE and y == MASK_VALUE: 
                        continue
                    uid_cells[uid].add((int(x), int(y)))

            # track presence
            if in_target(uid, city):
                target_users_present.add(uid)
            else:
                non_target_users_present.add(uid)

    # derive summary stats
    summary = {}
    summary["city"] = city
    summary["total_rows"] = int(total_rows)
    summary["num_unique_uids"] = int(len(unique_uids))
    summary["rows_by_day_sample"] = {int(k): int(v) for k,v in list(rows_by_day.items())[:10]}
    summary["days_total"] = len(rows_by_day)

    # mask summary overall and for days 61-75
    total_masked = sum(masked_by_day.values())
    total_unmasked = sum(unmasked_by_day.values())
    summary["total_masked"] = int(total_masked)
    summary["total_unmasked"] = int(total_unmasked)

    summary["days61_75"] = {
        "total_rows": int(sum(rows_by_day[d] for d in range(61,76) if d in rows_by_day)),
        "masked": int(sum(masked_by_day[d] for d in range(61,76) if d in masked_by_day)),
        "unmasked": int(sum(unmasked_by_day[d] for d in range(61,76) if d in unmasked_by_day)),
    }

    # target / non-target user summaries
    target_lo, target_hi = TARGET_RANGES[city]
    tgt_uids = [uid for uid in uid_agg.keys() if target_lo <= uid <= target_hi]
    non_tgt_uids = [uid for uid in uid_agg.keys() if not (target_lo <= uid <= target_hi)]

    def summarize_uid_list(uid_list):
        if not uid_list:
            return {}
        train_rows = np.array([uid_agg[u]["train"] for u in uid_list], dtype=int)
        test_total = np.array([uid_agg[u]["test_total"] for u in uid_list], dtype=int)
        test_masked = np.array([uid_agg[u]["test_masked"] for u in uid_list], dtype=int)
        test_unmasked = np.array([uid_agg[u]["test_unmasked"] for u in uid_list], dtype=int)
        total_rows_arr = np.array([uid_agg[u]["total"] for u in uid_list], dtype=int)

        # mask density per user (fraction masked among test_total). If test_total==0, set nan and ignore in stats.
        mask_density = np.array([ (uid_agg[u]["test_masked"] / uid_agg[u]["test_total"]) if uid_agg[u]["test_total"]>0 else np.nan for u in uid_list ])
        # count users with at least one unmasked test row
        have_unmasked = np.array([1 if uid_agg[u]["test_unmasked"]>0 else 0 for u in uid_list], dtype=int)

        return {
            "num_users": int(len(uid_list)),
            "train_rows": describe_arr(train_rows),
            "test_total": describe_arr(test_total),
            "test_masked": describe_arr(test_masked),
            "test_unmasked": describe_arr(test_unmasked),
            "total_rows": describe_arr(total_rows_arr),
            "mask_density_stats": describe_arr(mask_density[~np.isnan(mask_density)]) if np.any(~np.isnan(mask_density)) else {"count":0},
            "users_with_unmasked_test_rows": int(have_unmasked.sum())
        }

    summary["target_range"] = {
        "range": [int(target_lo), int(target_hi)],
        "users_present_in_data": int(len(tgt_uids)),
        "stats": summarize_uid_list(tgt_uids)
    }
    summary["non_target"] = {
        "users_present_in_data": int(len(non_tgt_uids)),
        "stats": summarize_uid_list(non_tgt_uids)
    }

    # optionally distinct cells per user
    if COMPUTE_DISTINCT_CELLS:
        distinct_counts = {uid: len(s) for uid,s in uid_cells.items()}
        arr = np.array(list(distinct_counts.values()), dtype=int) if distinct_counts else np.array([])
        summary["distinct_cells_per_user"] = describe_arr(arr)
    else:
        summary["distinct_cells_per_user"] = "COMPUTE_DISTINCT_CELLS=False (skipped)"

    # Save summary JSON
    out_json = os.path.join(OUTPUT_DIR, f"city_stats_{city}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_json}")

    # Save per-user CSV (if requested)
    if SAVE_PER_USER_CSV:
        rows_out = []
        for uid, vals in uid_agg.items():
            row = {
                "uid": int(uid),
                "is_target": bool(in_target(uid, city)),
                "total_rows": int(vals["total"]),
                "train_rows": int(vals["train"]),
                "test_total": int(vals["test_total"]),
                "test_masked": int(vals["test_masked"]),
                "test_unmasked": int(vals["test_unmasked"]),
            }
            if COMPUTE_DISTINCT_CELLS:
                row["distinct_cells"] = int(len(uid_cells.get(uid, set())))
            rows_out.append(row)
        per_user_df = pd.DataFrame(rows_out)
        out_csv = os.path.join(OUTPUT_DIR, f"per_user_stats_{city}.csv.gz")
        per_user_df.to_csv(out_csv, index=False, compression="gzip")
        print(f"Saved per-user CSV to {out_csv}")

    # print short summary for quick visibility
    print("Quick summary (printed):")
    print(json.dumps({
        "city": city,
        "total_rows": summary["total_rows"],
        "unique_uids": summary["num_unique_uids"],
        "days61_75_total": summary["days61_75"]["total_rows"],
        "days61_75_unmasked": summary["days61_75"]["unmasked"],
        "target_users_present": summary["target_range"]["users_present_in_data"],
        "target_users_with_unmasked_test_rows": summary["target_range"]["stats"]["users_with_unmasked_test_rows"],
    }, indent=2))
    return summary

# ----------------- run for all CITIES -----------------
if __name__ == "__main__":
    all_summaries = {}
    for city in CITIES:
        s = scan_city(city)
        all_summaries[city] = s
    all_out = os.path.join(OUTPUT_DIR, "all_city_summaries.json")
    with open(all_out, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"All summaries saved to {all_out}")
