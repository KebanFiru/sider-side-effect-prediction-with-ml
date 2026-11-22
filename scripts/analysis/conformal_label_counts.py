"""Positive / negative counts per endpoint for the calibration and test sets used in the
conformal study (needs outputs/conformal/rf_<split>.npz). Writes outputs/conformal/label_counts.csv.
Class-conditional (Mondrian) conformal at level alpha needs at least ceil(1/alpha)-1 examples
of the class: 9 for alpha=0.10, 4 for alpha=0.20."""
import csv
import gzip
import math

import numpy as np

OUT = "outputs/conformal"
SETS = [("rf_oof", "oof"), ("stopping_slice", "stop"), ("test", "test")]
with gzip.open("data/raw/sider.csv.gz", "rt") as f:
    names = next(csv.reader(f))[1:]
rows = []
for split in ("scaffold", "ogb_scaffold"):
    d = np.load(f"{OUT}/rf_{split}.npz", allow_pickle=True)
    for label, key in SETS:
        y = d[f"{key}_y"]
        for e, nm in enumerate(names):
            pos = int(y[:, e].sum()); n = int(len(y))
            rows.append([split, label, e, nm, n, pos, n - pos,
                         pos >= math.ceil(1 / 0.10) - 1, pos >= math.ceil(1 / 0.20) - 1,
                         (n - pos) >= math.ceil(1 / 0.10) - 1, (n - pos) >= math.ceil(1 / 0.20) - 1])
with open(f"{OUT}/label_counts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "set", "endpoint_idx", "endpoint", "n", "n_pos", "n_neg",
                "pos_ok_alpha0.10", "pos_ok_alpha0.20", "neg_ok_alpha0.10", "neg_ok_alpha0.20"])
    w.writerows(rows)
for split in ("scaffold", "ogb_scaffold"):
    for label, _ in SETS:
        r = [x for x in rows if x[0] == split and x[1] == label]
        print(split, label, "n =", r[0][4], "| min pos", min(x[5] for x in r), "min neg", min(x[6] for x in r),
              "| endpoints with <9 pos:", [x[3][:20] for x in r if x[5] < 9], "| <4 pos:", [x[3][:20] for x in r if x[5] < 4])
