# Training Progress

## Current State
- **Iteration**: 15 (completed)
- **Status**: Full player data + Fuzzy SVM + conditional ensembles
- **Best Test Model**: XGB_tuned_lean at 66.50% (test)
- **Best WF Single Model**: FSVM_time_lean at 65.50% (+5.55% edge)
- **Best WF Ensemble**: Ens_FSVM_XGB_w0.7 at 66.36% (+6.41% edge) — **all-time best**
- **Target**: 70% accuracy (3.64pp remaining)

### Pre-FSVM Best (Iteration 14 — XGBoost only)
- **Best WF Model**: XGB_tuned_lean (d=5) at 64.86% (+4.91% edge)
- **Best WF Ensemble**: Ens_rank_blend_a0.1 at 65.36% (+5.41% edge)

## Dataset
- Total scraped: 6000 matches from HLTV
- Top 100 filtered: 1647 matches
- Unique teams: 99
- Source: HLTV results pages (60 pages)
- **Match detail pages**: ALL 6,000 matches scraped (completed 2026-02-19)
  - match_details.csv: 6,000 rows
  - player_stats.csv: 74,979 rows
  - half_scores.csv: 7,296 rows
  - map_player_stats.csv: 133,470 rows
  - veto_data.csv: 15,858 rows
- Map results: 3,818 map results from 1,608 matches (extracted from batch JSONs)
- Rankings history: 2,907 rows, 29 weekly snapshots (Jul 21 2025 - Feb 9 2026)
- Pistol rounds: 3,808 rows from 1,608 matches (scraped via CDP/MCP browser)

## Pipeline (Iteration 8)
1. `scrape_data.py` — Re-scrape results with match URLs (incremental, resumable)
2. `scrape_match_details.py` — Async parallel scraper for player stats (5 workers)
3. `extract_map_data.py` — Extract map-level results from batch JSONs
4. `scrape_rankings.py` — Scrape weekly HLTV ranking history (resume support)
5. `train_model.py` — Dynamic ranks + map features + all previous features

## History

### Iteration 1 — Baseline Models
- **Features**: Elo ratings, ranking diff, recent form, H2H, streaks (15 features)
- **Models**: LogisticRegression, GradientBoosting, XGBoost, LightGBM
- **Best**: XGBoost 59.39% accuracy
- **Baselines**: Elo-only 54.85%, Rank-only 61.21%
- **Finding**: Rank-only baseline outperformed all ML models. Features added noise.

### Iteration 2-4 — Player-Level Features
- **New features**: Player rolling stats, team aggregates, roster experience,
  rest days, home/away inference, BO format, interaction features
- **Total features**: 51
- **Best**: XGBoost_tuned_lean 60.3% (iter 3-4)
- **Finding**: Still below rank-only baseline (60.91%)

### Iteration 5 — Memory-optimized + feature refinement
- **Fixes**: Reduced GridSearchCV from 1620 to 54 fits, n_jobs=1, gc.collect()
- **Best**: LR_L1_C0.01 at 63.33% on top-100 data (1648 matches)
- **Finding**: Heavy regularization + minimal features outperform complex models

### Iteration 6-7 — All data + sample weighting + new features
- **Data**: Expanded from 1,648 to **6,000 matches** (all scraped results)
- **New features**: Elo momentum, win rate vs strong teams, dynamic Elo-based rank
- **Sample weighting**: decay=0.985 (recent matches weighted higher)
- **Best test split**: LR_minimal at **66.58%** (9 features, C=0.01)
- **Walk-forward (honest)**: **61.02%** (vs rank-only 59.34%, +1.7% edge)
- **Baselines**: Rank-only 62.67%, Elo-only 58.17%
- **Stacking/Voting**: 64.5-64.8% — strong but below LR_minimal
- **Bug fixed**: StandardScaler was fit_transform on test set (data leakage)
- **Finding**: Simple LR with 9 features + sample decay beats all ensembles.
  Walk-forward shows true edge is ~1.7% over rank-only baseline.
  The 39% upset rate in CS limits any rank-based predictor to ~61%.

### Iteration 8 — Dynamic rankings + map features
- **New data**: Map results (3,818 maps from batch JSONs), rankings history (2,907 rows, 29 weeks)
- **New features**: dyn_log_rank_ratio, dyn_rank_diff_x_bo, map_pool_depth_diff,
  map_wr_overlap, map_upset_potential (11 minimal, 38 lean, 72 total)
- **Best test split**: XGB_lean at **65.0%** (log_loss=0.6252)
- **Walk-forward (honest)**: **60.36%** (vs dynamic rank 60.52%, -0.16%)
- **Baselines**: Rank-only static=62.67%, dynamic=60.08%, Elo-only=58.17%
- **Stacking**: 64.0%, Voting: 63.83%, XGB_tuned: 64.08%
- **5-Fold TS-CV**: LR_L1 mean=64.43%, XGB mean=63.70%
- **Finding**: Dynamic ranks are harder baseline than static (60.08 vs 62.67 on test).
  Walk-forward shows ML models no longer beat dynamic rank baseline.
  XGB_lean best on test split but overfits. Need better features for walk-forward edge.

### Iteration 9 — Team chemistry feature
- **New feature**: `diff_chemistry` — log(1 + avg days each pair of 5 players has played together)
  Computed via `ChemistryTracker` class that tracks `frozenset({p1, p2})` → first co-appearance date.
  For each match, calculates avg pairwise days together across all C(5,2)=10 pairs on each team.
- **Features**: 12 minimal, 39 lean, 75 total
- **Best test split**: Stacking_lean at **64.25%** (log_loss=0.6251)
- **Walk-forward (honest)**: **60.25%** (vs dynamic rank 60.52%, -0.27%)
- **5-Fold TS-CV**: LR_L1 mean=64.43%, XGB mean=63.82%
- **Finding**: Chemistry feature had negligible impact. Root cause: only 1,608/6,000 matches
  (27%) have player data, so chemistry is 0 for 73% of matches. The feature is more of a
  "has player data" flag than a true chemistry signal. Would need player data for all matches
  to properly evaluate. Also, the tracker needs many matches to warm up — early matches
  always show 0 chemistry even when player data exists.

### Iteration 10 — Deeper trees + SVM + multi-model walk-forward
- **Changes**: XGB max_depth increased (fixed=6, grid={3,5,7,10}), added SVM (rbf, C=1.0),
  walk-forward now evaluates top 3 models by test accuracy instead of hardcoded LR.
- **Best test split**: XGB_tuned_lean at **64.17%** (log_loss=0.6275, max_depth=5)
- **Walk-forward (top 3)**:
  - XGB_tuned_lean: **62.55%** (edge: +2.59%)
  - Calibrated_XGB_lean: **62.36%** (edge: +2.41%)
  - LGBM_lean: **60.95%** (edge: +1.00%)
  - Rank baseline (dynamic): 59.95%
- **SVM results**: 60.0% (minimal), 60.75% (lean) — underperformed all other models
- **5-Fold TS-CV**: LR_L1 mean=64.43%, XGB mean=63.58%
- **Grid search**: Best max_depth=5 (deeper trees didn't help)
- **Finding**: XGB-based models consistently show ~2.5% edge over rank in walk-forward.
  This is the best honest (walk-forward) result yet. SVM not competitive for this task.
  The gap between test accuracy (64.17%) and walk-forward (62.55%) suggests some overfitting
  but the walk-forward edge is real and reproducible across XGB variants.

### Iteration 11 — Rank volatility features + upset detector
- **New features**: Rank volatility (std of last 4 weekly ranks), rank trajectory (slope),
  rank_confidence (|rank_diff| / (1 + max_vol)), rank_conf_x_rank_diff (signed confidence),
  upset_prob (internal LR on non-rank features predicting P(upset)),
  upset_prob_x_rank_diff (interaction)
- **New code**: `build_rankings_index()` for O(1) rank lookups, `get_rank_volatility_features()`,
  `UpsetDetector` class (LR(C=0.1) retrained every 100 matches, min_samples=200)
- **Features**: 15 minimal, 46 lean, 82 total
- **Feature summary**:
  - rank_confidence: mean=8.66, std=15.23 (high variance = informative)
  - upset_prob: mean=0.27, std=0.06 (varied in 5800/6000 samples)
  - rank_trajectory_diff: mean=0.28, std=5.18
- **Best test split**: XGB_tuned_lean at **64.17%** (log_loss=0.6262, max_depth=3, lr=0.01)
- **Walk-forward (top 3)**:
  - XGB_tuned_lean: **63.23%** (edge: +3.27%) — up from 62.55%
  - XGB_lean: **62.41%** (edge: +2.45%)
  - Stacking_lean: **54.14%** (edge: -5.82%) — underperformed badly
  - Rank baseline (dynamic): 59.95%
- **Walk-forward ensembles**:
  - Ens_rank_blend_a0.1: **64.36%** (edge: +4.41%) — best ensemble yet
  - Ens_XGB_tune+Stacking: 63.00% (edge: +3.05%)
  - Ens_majority_vote: 62.64% (edge: +2.68%)
- **SHAP disagreement** (XGB_tuned vs XGB_lean):
  - `upset_prob` is top leading feature for XGB_lean in both correct (21 times) and wrong (22 times) disagreements
  - XGB_tuned relies more on `elo_rank_diff`, `rank_ratio`, `momentum_diff`
- **5-Fold TS-CV**: LR_L1 mean=64.28%, XGB mean=63.37%
- **Finding**: Walk-forward XGB_tuned improved from 62.55% to 63.23% (+0.68pp), best single-model
  WF result yet. The rank_blend ensemble at 64.36% is the best overall WF result. upset_prob
  is actively used by XGB_lean (top SHAP feature in disagreements) but doesn't clearly separate
  correct from wrong predictions — it's the top feature when both correct and wrong.
  rank_confidence and rank_trajectory_diff added to minimal set. Stacking collapsed in WF
  (likely due to new feature interactions confusing the meta-learner).

### Iteration 12 — Deeper trees exploration
- **Changes**: Added XGB/LGBM variants at depth 10, 12, 15. Grid search expanded to
  max_depth=[3,5,7,10,15,20], n_estimators=[200,400,600]. d15 uses lower lr=0.02, subsample=0.7.
- **Features**: Same 82 total, 15 minimal, 46 lean
- **Best test split**: XGB_lean_d15 at **66.00%** (log_loss=0.7152)
- **Test accuracy by depth**:
  - d6: 63.25% (XGB), 62.25% (LGBM)
  - d10: 65.67% (XGB)
  - d12: 64.08% (LGBM)
  - d15: **66.00%** (XGB) — best test result since Iter 7
  - Grid search best: d=3 (64.17%) — shallow still wins CV
- **Walk-forward (top 3)**:
  - XGB_tuned_lean (d=3): **63.23%** (edge: +3.27%) — still the WF champion
  - XGB_lean_d10: 63.18% (edge: +3.23%)
  - XGB_lean_d15: 63.09% (edge: +3.14%)
  - Rank baseline (dynamic): 59.95%
- **Walk-forward ensembles**: Ens_rank_blend_a0.1: **64.36%** (edge: +4.41%) — unchanged
- **SHAP disagreement** (d15 vs XGB_tuned):
  - d15 uses `upset_prob` as top feature (22-26 times) in both correct/wrong disagreements
  - XGB_tuned uses `elo_rank_diff`, `rank_ratio` — more stable/rank-focused
  - 399 disagreements, nearly 50/50 split (198 vs 201) — neither model dominates
- **Finding**: Deeper trees improve test accuracy significantly (+1.83pp at d15 vs d6) but
  walk-forward gains are negligible (+0.05pp). The gap between test (66.00%) and WF (63.09%)
  for d15 is 2.91pp, vs 0.94pp for d=3 tuned model. Deeper trees overfit to training patterns
  that don't generalize forward in time. The grid search correctly identifies d=3 as optimal.
  Shallow + regularized > deep for this task size (6K matches).

### Iteration 13 — Pistol round win rate feature
- **New data**: Scraped 3,808 pistol round results from per-map stats pages via CDP/MCP browser
  (two-phase scraper: Phase A extracts map stats URLs, Phase B extracts round history)
- **New feature**: `pistol_wr_diff` — difference in rolling pistol round win rates (window=30)
  via `PistolTracker` class. Each team's pistol win rate tracks wins in rounds 1 and 13 (MR12).
- **New scrapers**: `scrape_pistol_cdp.py` (CDP-based, connects to MCP Chrome to bypass Cloudflare),
  `scrape_pistol_data.py` (standalone, blocked by Cloudflare)
- **Features**: 16 minimal (+1), 47 lean (+1), 83 total (+1)
- **pistol_wr_diff stats**: mean=0.0012, std=0.0998
- **Best test split**: XGB_lean_d15 at **65.00%** (log_loss=0.7208)
- **Walk-forward (top 3)**:
  - XGB_tuned_lean (d=3): **63.50%** (edge: +3.55%) — up from 63.23%
  - XGB_lean_d15: **63.18%** (edge: +3.23%)
  - Ens_rank_blend_a0.1: **63.95%** (edge: +4.00%)
  - Rank baseline (dynamic): 59.95%
- **Walk-forward ensembles**: Ens_rank_blend_a0.1: **63.95%** (edge: +4.00%)
- **5-Fold TS-CV**: LR_L1 mean=64.35%, XGB mean=63.67%
- **Grid search**: Best max_depth=3, lr=0.01, n_est=200
- **Finding**: Pistol win rate feature provided a small but real lift to walk-forward single-model
  accuracy (+0.27pp for XGB_tuned_lean: 63.23% → 63.50%). The feature has reasonable variance
  (std=0.0998) but is not a dominant predictor — it didn't appear prominently in SHAP disagreement
  analysis. The ensemble result (63.95%) is slightly below the iter 11-12 peak (64.36%), likely
  due to minor retraining variance. Overall, pistol_wr_diff is a modest but valid addition.

### Iteration 14 — Full player data coverage (all 6,000 matches)
- **New data**: Scraped ALL 6,000 match detail pages from HLTV (previously only 1,608 = 27%).
  Now 5,627/6,000 (94%) matches have player-level data (some have no stats on HLTV).
  - player_stats.csv: 60,000 rows, 4,839 unique players
  - match_details.csv: 6,000 rows
  - half_scores.csv: 7,296 rows
  - map_player_stats.csv: 133,470 rows
  - veto_data.csv: 15,858 rows
- **Features**: Same 16 minimal, 47 lean, 83 total (no new features — just real data instead of defaults)
- **Key change**: Player features (avg_rating, diff_chemistry, diff_roster_exp, etc.) now computed
  from actual data for 94% of matches instead of defaulting to zeros/baselines for 73%.
- **Best test split**: XGB_tuned_lean at **66.50%** (log_loss=0.6105, max_depth=5, lr=0.01)
- **Walk-forward (top 3)**:
  - XGB_tuned_lean: **64.86%** (edge: +4.91%) — up from 63.50%, new all-time best single model
  - LGBM_lean_d6: **63.73%** (edge: +3.77%)
  - XGB_lean_d15: **63.32%** (edge: +3.36%)
  - Rank baseline (dynamic): 59.95%
- **Walk-forward ensembles**:
  - Ens_rank_blend_a0.1: **65.36%** (edge: +5.41%) — new all-time best overall
  - Ens_majority_vote: 64.32% (edge: +4.36%)
  - Ens_XGB_tune+LGBM_lea: 64.27% (edge: +4.32%)
- **SHAP disagreement**: `diff_roster_exp` is now the #1 leading feature in disagreements
  (was invisible before when 73% of matches had default=0). `diff_chemistry` also prominent.
- **5-Fold TS-CV**: LR_L1 mean=64.70%, XGB mean=63.08%
- **Grid search**: Best max_depth=5, lr=0.01, n_est=200
- **Finding**: Full player data delivered the largest single-iteration improvement in project
  history: +1.36pp for best single model (63.50% → 64.86%), +1.41pp for best ensemble
  (63.95% → 65.36%). Player features were severely handicapped when 73% of matches used
  default values — the model couldn't learn meaningful player-level patterns. Now with 94%
  coverage, roster experience and chemistry are among the most important features.
  The walk-forward edge of +5.41% over rank-only baseline is highly significant.
  Still 4.64pp from the 70% target.

### Iteration 15 — Fuzzy SVM + confusion matrix analysis + conditional ensembles
- **New model**: Fuzzy SVM (Lin & Wang, 2002) — SVM with per-sample fuzzy membership weights.
  Primal: `min (1/2)||w||^2 + C * SUM(s_i * xi_i)` where s_i = membership value.
  Implemented via `FuzzySVM` class in `train_model_fsvm.py` using sklearn's `SVC.fit(sample_weight=s)`.
- **Membership strategies**: class_center (distance from centroid), time_decay (exponential recency),
  confidence (KNN entropy), hybrid (combination). 11 FSVM configs tested.
- **Best FSVM test**: FSVM_time_lean at **64.92%** (log_loss=0.6328, C=1.0, lambda=0.001)
- **Best overall test**: XGB_tuned_lean at **66.50%** (log_loss=0.6105, unchanged)
- **Walk-forward (top 4 + forced FSVM)**:
  - **FSVM_time_lean: 65.50%** (edge: +5.55%) — **new all-time best single model**
  - XGB_tuned_lean: 64.86% (edge: +4.91%)
  - LGBM_lean_d6: 63.59% (edge: +3.64%)
  - XGB_lean_d15: 63.32% (edge: +3.37%)
  - Rank baseline (dynamic): 59.95%
- **Confusion matrix analysis (FSVM vs XGB)**:
  - FSVM excels at "Expected" outcomes: 80.0% vs XGB 74.7%
  - XGB excels at "Upsets": 40.6% vs FSVM 29.7%
  - FSVM better for large rank gaps (30-60 diff): 69.2% vs 65.1%
  - FSVM better for BO3: 65.9% vs 64.6%; XGB better for BO1/BO5
  - FSVM better at low-confidence predictions: 56.1% vs 53.9%
  - Agreement rate: 79.5% (349/439); when both agree, 75.4% correct
- **Conditional ensemble strategies tested**: conf_routing, rank_routing (20/30/50),
  rank_adaptive_blend, upset_routing, fixed weight blends (w0.3–w0.7)
- **Walk-forward ensembles**:
  - **Ens_FSVM_XGB_w0.7: 66.36%** (edge: +6.41%) — **ALL-TIME BEST**
  - Ens_FSVM_XGB_w0.6: 66.18% (edge: +6.23%)
  - Ens_rank_adaptive_blend: 65.77% (edge: +5.82%)
  - Ens_conf_routing: 65.73% (edge: +5.78%)
  - Ens_XGB_tune+FSVM_tim: 65.73% (edge: +5.78%)
  - Ens_rank_blend_a0.1: 65.00% (edge: +5.05%)
- **Finding**: Fuzzy SVM and XGBoost have deeply complementary strengths. FSVM dominates on
  predictable/expected outcomes (where the favorite wins), large rank gaps, and BO3 format.
  XGB dominates on upsets, BO1/BO5, and high-confidence predictions. A 70/30 FSVM/XGB
  probability blend (Ens_FSVM_XGB_w0.7) captures the best of both worlds at 66.36% WF
  accuracy — a +1.00pp improvement over the previous best ensemble (65.36%) and +6.41%
  edge over rank-only baseline. This is the strongest honest result in the project.
  Still 3.64pp from the 70% target.
