import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
WEBSITE_DIR = os.path.join(PROJECT_ROOT, "website")

DB_PATH = os.path.join(WEBSITE_DIR, "predictions.db")
TRACKER_STATE_PATH = os.path.join(WEBSITE_DIR, "tracker_state.dat")

# Ensemble weights (from model_snapshot_iter19_winner.json)
FSVM_WEIGHT = 0.7
XGB_WEIGHT = 0.3

# Player/map tracker windows (must match training)
PLAYER_WINDOW = 10
MAP_WINDOW = 20

# 47 lean features in exact training order
LEAN_COLS = [
    "diff_avg_rating", "diff_avg_adr", "diff_avg_kast", "diff_avg_kd_diff",
    "diff_star_rating", "diff_weakest_rating", "diff_star_gap", "diff_consistency",
    "diff_avg_rest", "diff_roster_exp", "diff_chemistry",
    "elo_diff", "elo_expected", "rank_diff", "log_rank_ratio", "rank_ratio",
    "dyn_rank_diff", "dyn_log_rank_ratio",
    "elo_rank_diff", "log_elo_rank_ratio", "momentum_diff", "vs_strong_diff",
    "form_diff", "streak_diff", "h2h_winrate", "h2h_matches", "home_diff",
    "is_bo1", "is_bo3", "is_bo5",
    "rank_diff_x_bo", "dyn_rank_diff_x_bo", "elo_rank_diff_x_bo",
    "elo_diff_x_form", "rank_x_h2h", "momentum_x_form",
    "map_pool_depth_diff", "map_wr_overlap", "map_upset_potential",
    "rank_vol_diff", "rank_vol_max", "rank_trajectory_diff",
    "rank_confidence", "rank_conf_x_rank_diff",
    "upset_prob", "upset_prob_x_rank_diff",
    "pistol_wr_diff",
]
