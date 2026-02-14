# CS Esport Predictor -- Results Summary

All iterations from `experiments.json`, one row per logical iteration.
Walk-forward (WF) results use the best single-model accuracy unless noted.
"Edge" = WF model accuracy minus WF rank baseline.

| Iter | Features | Best Model | Test Acc | Test LogLoss | WF Best | WF Edge | Rank Baseline | Notes |
|-----:|---------:|:-----------|:--------:|:------------:|:-------:|:-------:|:-------------:|:------|
|    1 |       15 | XGBoost              | 59.39% | 0.7051 |    -    |    -    |  61.21% (static)  | Baseline Elo/rank/form/H2H features |
|    2 |       51 | LogisticRegression   | 58.79% | 0.6847 |    -    |    -    |  60.91% (static)  | Added player stats, roster, rest days |
|    3 |       51 | XGB_tuned_lean       | 60.30% | 0.6850 |    -    |    -    |  60.91% (static)  | Lean feature set + XGB grid search |
|    4 |       51 | XGB_tuned_lean       | 60.30% | 0.6850 |    -    |    -    |  60.91% (static)  | Interaction features, BO format added |
|    5 |       56 | LR_L1_C0.01         | 63.33% | 0.6611 |    -    |    -    |  60.91% (static)  | Memory-optimized, heavy regularization |
|    6 |       64 | LR_minimal           | 65.83% | 0.6094 |    -    |    -    |  62.67% (static)  | Expanded to 6K matches, Elo momentum |
|    7 |       64 | LR_minimal           | 66.58% | 0.6056 | 61.02% | +1.68%  |  59.34% (WF dyn)  | Sample weighting, scaler leak fix |
|    8 |       72 | XGB_lean             | 65.00% | 0.6252 | 60.36% | -0.16%  |  60.52% (WF dyn)  | Dynamic ranks + map pool features |
|    9 |       75 | Stacking_lean        | 64.25% | 0.6251 | 60.25% | -0.27%  |  60.52% (WF dyn)  | Team chemistry feature (negligible) |
|   10 |       75 | XGB_tuned_lean       | 64.17% | 0.6275 | 62.55% | +2.60%  |  59.95% (WF dyn)  | Deeper trees, SVM, multi-model WF |
|   11 |       82 | XGB_tuned_lean       | 64.17% | 0.6262 | 63.23% | +3.27%  |  59.95% (WF dyn)  | Rank volatility + upset detector |
|   12 |       82 | XGB_lean_d15         | 66.00% | 0.7152 | 63.23% | +3.27%  |  59.95% (WF dyn)  | Deeper trees (d10, d15, d20 grid) |
|   13 |       83 | XGB_lean_d15         | 65.00% | 0.7208 | 63.50% | +3.55%  |  59.95% (WF dyn)  | Pistol round win rate feature |

### Walk-Forward Ensemble Highlights

| Iter | Best Ensemble          | WF Acc  | WF Edge |
|-----:|:-----------------------|:-------:|:-------:|
|   10 | Ens_rank_blend_a0.2    | 63.05%  | +3.10%  |
|   11 | Ens_rank_blend_a0.1    | 64.36%  | +4.41%  |
|   12 | Ens_rank_blend_a0.1    | 64.36%  | +4.41%  |
|   13 | Ens_rank_blend_a0.1    | 63.95%  | +4.00%  |

### Key Observations

- **Best test accuracy**: 66.58% (Iter 7, LR_minimal) and 66.00% (Iter 12, XGB_lean_d15)
- **Best walk-forward single model**: 63.50% (Iter 13, XGB_tuned_lean d=3, +3.55% edge)
- **Best walk-forward ensemble**: 64.36% (Iter 11-12, Ens_rank_blend_a0.1, +4.41% edge)
- **Target**: 70% accuracy (not yet met)
- Rank-only baseline outperformed ML models in Iters 1-4
- Walk-forward validation consistently shows lower accuracy than test split (overfitting gap)
- Simple models (LR with few features) outperformed complex ones until Iter 10+
- Deeper trees (d15) improve test accuracy (+1.83pp) but NOT walk-forward — overfitting
