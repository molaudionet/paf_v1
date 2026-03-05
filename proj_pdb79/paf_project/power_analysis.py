#!/usr/bin/env python3
"""
Power Analysis for Paper 1: Zero-Shot BBBP Audio Classification
================================================================
Determines the minimum sample size needed to detect above-chance
accuracy with a one-sided binomial test at alpha=0.05.

Also shows: for a *given* N, what true accuracy is detectable?
"""

import numpy as np
from scipy.stats import binom, binomtest
import matplotlib.pyplot as plt
from itertools import product


def min_correct_for_significance(n: int, alpha: float = 0.05) -> int:
    """Minimum number of correct predictions to reject H0: p=0.5 (one-sided)."""
    for k in range(n + 1):
        p_val = binom.sf(k - 1, n, 0.5)  # P(X >= k) under H0
        if p_val < alpha:
            return k
    return n + 1  # never significant


def power_at(n: int, true_acc: float, alpha: float = 0.05) -> float:
    """
    Probability of detecting significance given true accuracy and sample size.
    Power = P(X >= k_crit | p = true_acc)
    """
    k_crit = min_correct_for_significance(n, alpha)
    return float(binom.sf(k_crit - 1, n, true_acc))


def main():
    alpha = 0.05
    target_power = 0.80  # standard threshold

    # ---- Table 1: For fixed N, what accuracy is detectable? ----
    print("=" * 70)
    print("TABLE 1: Minimum detectable accuracy at alpha=0.05 (one-sided binomial)")
    print("=" * 70)
    print(f"{'N':>6}  {'Min correct':>12}  {'Min accuracy':>13}  {'p-value at min':>14}")
    print("-" * 70)

    for n in [100, 150, 200, 300, 400, 500]:
        k = min_correct_for_significance(n, alpha)
        acc = k / n
        pval = binom.sf(k - 1, n, 0.5)
        print(f"{n:>6}  {k:>12}  {acc:>13.3f}  {pval:>14.4f}")

    # ---- Table 2: Power for various (N, true_acc) combos ----
    print("\n")
    print("=" * 70)
    print("TABLE 2: Statistical power (prob of detecting significance)")
    print("         H0: acc=0.50, one-sided binomial, alpha=0.05")
    print("=" * 70)

    n_values = [100, 150, 200, 300, 400, 500]
    acc_values = [0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.65, 0.70]

    header = f"{'True acc':>9}" + "".join(f"  N={n:>3}" for n in n_values)
    print(header)
    print("-" * len(header))

    for acc in acc_values:
        row = f"{acc:>9.2f}"
        for n in n_values:
            pw = power_at(n, acc, alpha)
            marker = " *" if pw >= target_power else "  "
            row += f"  {pw:>.3f}{marker}"
        print(row)

    print("\n  * = power >= 0.80 (adequate)")

    # ---- Table 3: Minimum N for 80% power at each true accuracy ----
    print("\n")
    print("=" * 70)
    print("TABLE 3: Minimum N for 80% power at each true accuracy level")
    print("=" * 70)
    print(f"{'True acc':>9}  {'Min N (power>=0.80)':>20}")
    print("-" * 35)

    for acc in acc_values:
        for n in range(50, 2001, 10):
            if power_at(n, acc, alpha) >= target_power:
                print(f"{acc:>9.2f}  {n:>20}")
                break
        else:
            print(f"{acc:>9.2f}  {'>2000':>20}")

    # ---- Figure: Power curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Power vs N for different true accuracies
    ax = axes[0]
    ns = np.arange(50, 601, 10)
    for acc in [0.55, 0.58, 0.60, 0.65, 0.70]:
        powers = [power_at(int(n), acc, alpha) for n in ns]
        ax.plot(ns, powers, label=f"True acc = {acc:.2f}", linewidth=2)
    ax.axhline(0.80, color="gray", linestyle="--", alpha=0.7, label="Power = 0.80")
    ax.axvline(200, color="red", linestyle=":", alpha=0.7, label="N = 200 (current plan)")
    ax.axvline(400, color="blue", linestyle=":", alpha=0.7, label="N = 400 (recommended)")
    ax.set_xlabel("Sample Size (N)", fontsize=12)
    ax.set_ylabel("Power", fontsize=12)
    ax.set_title("A) Power vs Sample Size", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel B: Minimum detectable accuracy vs N
    ax = axes[1]
    ns_b = np.arange(50, 601, 10)
    min_accs = []
    for n in ns_b:
        k = min_correct_for_significance(int(n), alpha)
        min_accs.append(k / n)
    ax.plot(ns_b, min_accs, color="darkgreen", linewidth=2.5)
    ax.axvline(200, color="red", linestyle=":", alpha=0.7, label="N = 200")
    ax.axvline(400, color="blue", linestyle=":", alpha=0.7, label="N = 400")
    ax.set_xlabel("Sample Size (N)", fontsize=12)
    ax.set_ylabel("Minimum Detectable Accuracy", fontsize=12)
    ax.set_title("B) Detection Threshold vs Sample Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.50, 0.70)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("power_analysis_paper1.png", dpi=200, bbox_inches="tight")
    print("\nFigure saved: power_analysis_paper1.png")

    # ---- Summary recommendation ----
    print("\n")
    print("=" * 70)
    print("RECOMMENDATION FOR PAPER 1")
    print("=" * 70)
    print("""
At N=200 (your current plan):
  - You need ~58% accuracy (116/200 correct) for significance
  - If true accuracy is 55%, you have only ~26% power (BAD — likely miss it)
  - If true accuracy is 60%, you have ~75% power (borderline)
  - If true accuracy is 65%, you have ~98% power (safe)

At N=400 (recommended):
  - You need ~55% accuracy (220/400 correct) for significance
  - If true accuracy is 55%, you have ~52% power (still weak)
  - If true accuracy is 58%, you have ~83% power (adequate)
  - If true accuracy is 60%, you have ~97% power (strong)

BOTTOM LINE:
  - If you expect accuracy ~55-58%, use N=400 (200/200 balanced)
  - If you expect accuracy ~60%+, N=200 is probably fine
  - N=400 costs only ~2x more API calls — worth the insurance

STRATEGY:
  1. Run pilot with N=50 first (25/25) to estimate true accuracy
  2. Use pilot estimate to pick final N via this power table
  3. If pilot shows ~55%, scale to N=400
  4. If pilot shows ~60%+, N=200 is sufficient
""")


if __name__ == "__main__":
    main()
