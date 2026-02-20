# CS Esport Predictor -- Results Summary

All iterations from `experiments.json`, one row per logical iteration.
Walk-forward (WF) results use the best single-model accuracy unless noted.
"Edge" = WF model accuracy minus WF rank baseline.
"95% CI" = Bootstrap 95% confidence interval (10,000 resamples) for the best WF model.

| Iter | Features | Best Model | Test Acc | Test LogLoss | WF Best | WF Edge | 95% CI | Rank Baseline | Notes |
|-----:|---------:|:-----------|:--------:|:------------:|:-------:|:-------:|:------:|:-------------:|:------|
|    1 |       15 | XGBoost              | 59.39% | 0.7051 |    -    |    -    |    -   |  61.21% (static)  | Baseline Elo/rank/form/H2H features |
|    2 |       51 | LogisticRegression   | 58.79% | 0.6847 |    -    |    -    |    -   |  60.91% (static)  | Added player stats, roster, rest days |
|    3 |       51 | XGB_tuned_lean       | 60.30% | 0.6850 |    -    |    -    |    -   |  60.91% (static)  | Lean feature set + XGB grid search |
|    4 |       51 | XGB_tuned_lean       | 60.30% | 0.6850 |    -    |    -    |    -   |  60.91% (static)  | Interaction features, BO format added |
|    5 |       56 | LR_L1_C0.01         | 63.33% | 0.6611 |    -    |    -    |    -   |  60.91% (static)  | Memory-optimized, heavy regularization |
|    6 |       64 | LR_minimal           | 65.83% | 0.6094 |    -    |    -    |    -   |  62.67% (static)  | Expanded to 6K matches, Elo momentum |
|    7 |       64 | LR_minimal           | 66.58% | 0.6056 | 61.02% | +1.68%  |    -   |  59.34% (WF dyn)  | Sample weighting, scaler leak fix |
|    8 |       72 | XGB_lean             | 65.00% | 0.6252 | 60.36% | -0.16%  |    -   |  60.52% (WF dyn)  | Dynamic ranks + map pool features |
|    9 |       75 | Stacking_lean        | 64.25% | 0.6251 | 60.25% | -0.27%  |    -   |  60.52% (WF dyn)  | Team chemistry feature (negligible) |
|   10 |       75 | XGB_tuned_lean       | 64.17% | 0.6275 | 62.55% | +2.60%  |    -   |  59.95% (WF dyn)  | Deeper trees, SVM, multi-model WF |
|   11 |       82 | XGB_tuned_lean       | 64.17% | 0.6262 | 63.23% | +3.27%  |    -   |  59.95% (WF dyn)  | Rank volatility + upset detector |
|   12 |       82 | XGB_lean_d15         | 66.00% | 0.7152 | 63.23% | +3.27%  |    -   |  59.95% (WF dyn)  | Deeper trees (d10, d15, d20 grid) |
|   13 |       83 | XGB_lean_d15         | 65.00% | 0.7208 | 63.50% | +3.55%  |    -   |  59.95% (WF dyn)  | Pistol round win rate feature |
|   14 |       83 | XGB_tuned_lean       | 66.50% | 0.6105 | 64.86% | +4.91%  |    -   |  59.95% (WF dyn)  | Full player data (94% coverage) |
|   15 |       83 | XGB_tuned_lean       | 66.50% | 0.6105 | 65.50% | +5.55%  |    -   |  59.95% (WF dyn)  | Fuzzy SVM (FSVM_time_lean = best WF single) |
|   16 |       83 | XGB_tuned_lean       | 66.50% | 0.6105 | 65.50% | +5.55%  | [63.5, 67.5] |  59.95% (WF dyn)  | Overfit diagnostics added |
|   18 |       84 | XGB_tuned_lean       | 66.58% | 0.6102 | 65.55% | +5.60%  | [63.6, 67.6] |  59.95% (WF dyn)  | best_team_rank + tier_boost=1.5; individuals improved, ensemble dropped |
|   19 |       83 | XGB_tuned_lean       | 66.50% | 0.6105 | 65.50% | +5.55%  | [63.5, 67.5] |  59.95% (WF dyn)  | Reverted to iter 16; tier-specialized 4-model ensemble + agree-boost |

### Walk-Forward Ensemble Highlights

| Iter | Best Ensemble          | WF Acc  | WF Edge | 95% CI |
|-----:|:-----------------------|:-------:|:-------:|:------:|
|   10 | Ens_rank_blend_a0.2    | 63.05%  | +3.10%  |    -   |
|   11 | Ens_rank_blend_a0.1    | 64.36%  | +4.41%  |    -   |
|   12 | Ens_rank_blend_a0.1    | 64.36%  | +4.41%  |    -   |
|   13 | Ens_rank_blend_a0.1    | 63.95%  | +4.00%  |    -   |
|   14 | Ens_rank_blend_a0.1    | 65.36%  | +5.41%  |    -   |
|   15 | Ens_FSVM_XGB_w0.7     | **66.36%** | **+6.41%** | **[64.4, 68.3]** |
|   18 | Ens_FSVM_XGB_w0.65    | 66.05%  | +6.10%  | [64.0, 68.0] |
|   19 | Ens_agree_boost_w0.7  | **66.50%** | **+6.55%** | **[64.5, 68.5]** |

### Tier-Specialized Models (Iteration 19)

| Model | All Samples | Own Tier | N (own tier) |
|:------|:----------:|:--------:|:------------:|
| FSVM_elite (rank<55) | 65.82% | **70.00%** | 800 |
| XGB_elite (rank<55)  | 62.09% | 66.87% | 800 |
| FSVM_lower (rank>=55) | 63.05% | 63.14% | 1400 |
| XGB_lower (rank>=55)  | 62.68% | 60.00% | 1400 |
| Best TierSpec ensemble | 65.59% | — | 2200 |

### Overfit Diagnostics (Iteration 16)

| Model | Train Acc | WF Acc | Train-WF Gap | Window Std | 95% CI | SV Ratio | vs Rank CI |
|:------|:---------:|:------:|:------------:|:----------:|:------:|:--------:|:-----------|
| XGB_tuned_lean   | 63.11% | 64.86% | **-1.75%** | 3.85% | [62.9, 66.9] | — | SEPARATED (+0.86pp) |
| XGB_lean_d15     | 64.51% | 63.32% | +1.19%     | 3.83% | [61.3, 65.3] | — | overlaps |
| LGBM_lean_d6     | 63.02% | 63.73% | -0.71%     | 3.00% | [61.8, 65.8] | — | overlaps |
| FSVM_time_lean   | 67.35% | 65.50% | +1.85%     | **2.14%** | [63.5, 67.5] | 81.4% | SEPARATED (+1.50pp) |
| Ens_FSVM_XGB_w0.7 | — | 66.36% | — | — | [64.4, 68.3] | — | SEPARATED (+2.41pp) |
| Rank baseline    | — | 59.95% | — | — | [57.9, 62.0] | — | — |

**Key findings:**
- **FSVM_time_lean is NOT overfit**: Train-WF gap of +1.85% is small and comparable to XGB_lean_d15 (+1.19%). Its **lowest window std (2.14%)** among all models indicates the most temporally stable predictions.
- **XGB_tuned_lean shows negative gap** (-1.75%): WF accuracy exceeds mean training accuracy, indicating it generalizes well rather than memorizing.
- **Bootstrap CIs confirm significance**: Both FSVM_time_lean and XGB_tuned_lean CIs are fully separated from the rank baseline CI. The best ensemble (Ens_FSVM_XGB_w0.7) CI lower bound (64.4%) is 2.4pp above the rank CI upper bound (62.0%).
- **FSVM SV ratio (81.4%)** is high but expected for RBF kernels on noisy data — the kernel implicitly regularizes via the margin, and time_decay membership down-weights older noisy samples.
- **XGB_lean_d15 and LGBM_lean_d6 CIs overlap with rank**: These models don't show statistically significant improvement over the rank baseline individually.

### Key Observations

- **Best test accuracy**: 66.58% (Iter 7, LR_minimal) and 66.50% (Iter 14-16, XGB_tuned_lean)
- **Best walk-forward single model**: 65.50% (Iter 15-16, FSVM_time_lean, +5.55% edge)
- **Best walk-forward ensemble**: **66.50%** (Iter 19, Ens_agree_boost_w0.7, +6.55% edge)
- **Target**: 70% accuracy (3.50pp remaining)
- Rank-only baseline outperformed ML models in Iters 1-4
- Walk-forward validation consistently shows lower accuracy than test split (overfitting gap)
- Simple models (LR with few features) outperformed complex ones until Iter 10+
- Deeper trees (d15) improve test accuracy (+1.83pp) but NOT walk-forward — overfitting
- Full player data (Iter 14) was the largest single-iteration lift: +1.36pp WF single, +1.41pp WF ensemble
- Fuzzy SVM + XGB blend (Iter 15) leverages complementary strengths: FSVM on expected outcomes, XGB on upsets
- **Overfit check (Iter 16)**: FSVM and XGB_tuned are statistically significant vs rank baseline; the ensemble is strongly significant (CI lower bound 2.4pp above rank CI upper bound)
- **Tier-specialized models (Iter 19)**: FSVM_elite achieves 70% on elite (rank<55) matches — strong betting signal. But tier-specialized ensembles (65.59%) didn't beat global ensembles (66.50%) due to smaller training subsets
- **Agreement-boost ensemble (Iter 19)**: New best at 66.50% — trusts predictions when FSVM & XGB agree, uses 70%FSVM/30%XGB blend on disagreements
