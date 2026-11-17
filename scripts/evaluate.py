"""Evaluation: confusion matrix, false-alarm rate, 95% Wilson CI."""
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


def evaluate(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[LABELS.index(t), LABELS.index(p)] += 1
    n = int(cm.sum())
    correct = int(np.trace(cm))
    n_eq = int(cm[0].sum())
    false_alarms = int(cm[0, 1])

    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(pd.DataFrame(cm, index=["true EQ", "true UNE"],
                       columns=["pred EQ", "pred UNE"]).to_string())

    lo, hi = wilson(correct, n)
    print(f"\nAccuracy: {correct}/{n} = {correct / n:.1%}  "
          f"(95% CI {lo:.1%} to {hi:.1%})")
    print("PASS (>90%)" if correct / n > 0.90 else "FAIL (<=90%)")

    if n_eq:
        lo, hi = wilson(false_alarms, n_eq)
        print(f"False-alarm rate: {false_alarms}/{n_eq} = "
              f"{false_alarms / n_eq:.1%}  (95% CI {lo:.1%} to {hi:.1%})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CSV with a column named "pred" holding EQ/UNE (or earthquake/explosion),
        # one row per event
        pred = [norm(p) for p in pd.read_csv(sys.argv[1])["pred"]]
        truth = ["EQ"] * len(pred)  # all 33 DAS events are earthquakes
    else:
        print("DUMMY PREDICTIONS (31 EQ, 2 UNE), for testing only")
        truth = ["EQ"] * 33
        pred = ["EQ"] * 31 + ["UNE"] * 2
    evaluate(truth, pred)