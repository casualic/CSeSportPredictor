"""Generate side-by-side comparison of iter 16 vs iter 18 tier accuracy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Data from iteration 16 (before: no best_team_rank, no tier weighting)
iter16 = {
    "name": "Iter 16 (Ens_FSVM_XGB_w0.7)",
    "overall": 66.4,
    "best_tier_labels": ["Top 10", "10-20", "20-35", "35-55", "55+"],
    "best_tier_counts": [118, 96, 195, 391, 1400],
    "best_tier_upsets": [72.9, 68.8, 70.8, 73.7, 29.2],
    "best_tier_acc": [72.0, 66.7, 69.7, 71.4, 64.0],
}

# Data from iteration 18 (after: best_team_rank + tier_boost=1.5 + improved ensembles)
iter18 = {
    "name": "Iter 18 (Ens_FSVM_XGB_w0.65)",
    "overall": 66.1,
    "best_tier_labels": ["Top 10", "10-20", "20-35", "35-55", "55+"],
    "best_tier_counts": [118, 96, 195, 391, 1400],
    "best_tier_upsets": [72.9, 68.8, 70.8, 73.7, 29.2],
    "best_tier_acc": [68.6, 69.8, 68.7, 71.4, 63.7],
}

labels = iter16["best_tier_labels"]
counts = iter16["best_tier_counts"]
x = np.arange(len(labels))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Tier-Aware Changes: Iteration 16 vs 18\n(best_team_rank feature + tier weighting + improved ensembles)",
             fontsize=14, fontweight="bold")

# Left: Accuracy comparison
ax = axes[0]
bars1 = ax.bar(x - w/2, iter16["best_tier_acc"], w, label=iter16["name"],
               color="#3498db", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + w/2, iter18["best_tier_acc"], w, label=iter18["name"],
               color="#2ecc71", alpha=0.85, edgecolor="white")

ax.set_xlabel("Best (Higher-Ranked) Team Tier", fontsize=11)
ax.set_ylabel("Ensemble Accuracy (%)", fontsize=11)
ax.set_title("Accuracy by Higher-Ranked Team Tier", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(labels, counts)], fontsize=9)
ax.set_ylim(55, 80)
ax.legend(fontsize=9, loc="upper right")

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color="#2c3e50")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color="#27ae60")

# Overall lines
ax.axhline(y=iter16["overall"], color="#3498db", linestyle="--", alpha=0.4,
           linewidth=1)
ax.axhline(y=iter18["overall"], color="#2ecc71", linestyle="--", alpha=0.4,
           linewidth=1)

# Right: Delta (change)
ax = axes[1]
deltas = [a18 - a16 for a16, a18 in zip(iter16["best_tier_acc"], iter18["best_tier_acc"])]
colors = ["#2ecc71" if d >= 0 else "#e74c3c" for d in deltas]
bars = ax.bar(x, deltas, 0.6, color=colors, alpha=0.85, edgecolor="white")

ax.set_xlabel("Best (Higher-Ranked) Team Tier", fontsize=11)
ax.set_ylabel("Accuracy Change (pp)", fontsize=11)
ax.set_title("Change: Iter 18 vs Iter 16", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(labels, counts)], fontsize=9)
ax.axhline(y=0, color="gray", linewidth=0.8)
ax.set_ylim(-3, 3.5)

for bar, d in zip(bars, deltas):
    va = "bottom" if d >= 0 else "top"
    offset = 0.1 if d >= 0 else -0.1
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
            f"{d:+.1f}pp", ha="center", va=va, fontsize=10, fontweight="bold")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "tier_comparison_16_vs_18.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
