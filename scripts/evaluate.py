"""Evaluation: confusion matrix, false-alarm rate, 95% Wilson CI.

Usage:
    python scripts/evaluate.py                         (dummy test)
    python scripts/evaluate.py data/predictions/a.csv [b.csv ...]

CSV columns: event_id, dataset, true, pred, prob_une
true and pred hold EQ or UNE (earthquake/explosion are also accepted).
"""
import sys

import numpy as np
import pandas as pd

LABELS = ["EQ", "UNE"]


def norm(label):
    s = str(label).strip().lower()
    if s in ("eq", "earthquake"):
        return "EQ"
    if s in ("une", "explosion"):
        return "UNE"
    raise ValueError(f"Unknown label: {label!r}")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return center - half, center + half


def evaluate(y_true, y_pred, title=""):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[LABELS.index(t), LABELS.index(p)] += 1
    n = int(cm.sum())
    if n == 0:
        return
    correct = int(np.trace(cm))
    n_eq = int(cm[0].sum())
    n_une = int(cm[1].sum())

    print("\n=== " + (title or "All events") + f" (n = {n}) ===")
    print("Confusion matrix (rows = true, columns = predicted)")
    print(pd.DataFrame(cm, index=["true EQ", "true UNE"],
                       columns=["pred EQ", "pred UNE"]).to_string())

    lo, hi = wilson(correct, n)
    print(f"\nAccuracy: {correct}/{n} = {correct / n:.1%}  "
          f"(95% CI {lo:.1%} to {hi:.1%})")
    print("PASS (>90%)" if correct / n > 0.90 else "FAIL (<=90%)")

    if n_eq:
        fa = int(cm[0, 1])
        lo, hi = wilson(fa, n_eq)
        print(f"False-alarm rate (EQ called UNE): {fa}/{n_eq} = "
              f"{fa / n_eq:.1%}  (95% CI {lo:.1%} to {hi:.1%})")
    if n_une:
        hit = int(cm[1, 1])
        lo, hi = wilson(hit, n_une)
        print(f"Explosion detection rate (UNE called UNE): {hit}/{n_une} = "
              f"{hit / n_une:.1%}  (95% CI {lo:.1%} to {hi:.1%})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        df = pd.concat([pd.read_csv(p) for p in sys.argv[1:]], ignore_index=True)
        for col in ("true", "pred"):
            if col not in df.columns:
                raise SystemExit(f"Missing column '{col}'. Found: {list(df.columns)}")
        y_true = [norm(v) for v in df["true"]]
        y_pred = [norm(v) for v in df["pred"]]
        evaluate(y_true, y_pred)
        if "dataset" in df.columns and df["dataset"].nunique() > 1:
            for name, g in df.groupby("dataset"):
                evaluate([norm(v) for v in g["true"]],
                         [norm(v) for v in g["pred"]], title=str(name))
    else:
        print("DUMMY PREDICTIONS (31 EQ, 2 UNE), for testing only")
        evaluate(["EQ"] * 33, ["EQ"] * 31 + ["UNE"] * 2)