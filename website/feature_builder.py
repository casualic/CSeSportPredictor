"""Build a 47-feature vector for a single upcoming match from TrackerBundle state."""
import numpy as np
from itertools import combinations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model_fsvm import (
    get_rank_volatility_features, get_home_advantage, parse_hltv_date,
)
from website.config import LEAN_COLS


def build_match_features(bundle, team1, team2, event="", bo_format="BO3",
                         match_date=None, t1_players=None, t2_players=None):
    """Build feature dict for a single match using current tracker state.

    Args:
        bundle: TrackerBundle with current tracker state
        team1, team2: team name strings
        event: event name string
        bo_format: "BO1", "BO3", or "BO5"
        match_date: datetime or None
        t1_players: list of player name strings (optional, uses last known if None)
        t2_players: list of player name strings (optional)

    Returns:
        dict with all 47 lean_cols features
    """
    t1, t2 = team1, team2
    date = parse_hltv_date(match_date) if isinstance(match_date, str) else match_date

    # --- Elo features ---
    elo1 = bundle.elo.get_rating(t1)
    elo2 = bundle.elo.get_rating(t2)

    # Static HLTV rank
    rank1 = bundle.top_teams.get(t1, 101)
    rank2 = bundle.top_teams.get(t2, 101)

    # Dynamic HLTV rank + volatility
    if bundle.rankings_index:
        dyn_rank1, rank_vol1, rank_traj1 = get_rank_volatility_features(t1, date, bundle.rankings_index)
        dyn_rank2, rank_vol2, rank_traj2 = get_rank_volatility_features(t2, date, bundle.rankings_index)
    else:
        dyn_rank1, rank_vol1, rank_traj1 = rank1, 0.0, 0.0
        dyn_rank2, rank_vol2, rank_traj2 = rank2, 0.0, 0.0

    # Dynamic Elo-based rank
    elo_rank1 = bundle.elo.get_elo_rank(t1)
    elo_rank2 = bundle.elo.get_elo_rank(t2)

    # Form
    form1 = bundle.form.get_form(t1)
    form2 = bundle.form.get_form(t2)
    streak1 = bundle.form.get_streak(t1)
    streak2 = bundle.form.get_streak(t2)

    # H2H
    h2h_rate = bundle.h2h.get_h2h(t1, t2)
    h2h_matches = bundle.h2h.get_total(t1, t2)

    # Momentum & strength
    momentum1 = bundle.elo.get_momentum(t1)
    momentum2 = bundle.elo.get_momentum(t2)
    vs_strong1 = bundle.elo.get_winrate_vs_strong(t1)
    vs_strong2 = bundle.elo.get_winrate_vs_strong(t2)

    # Rank ratios
    rank_ratio = rank2 / max(rank1, 1)
    log_rank_ratio = np.log(max(rank_ratio, 0.01))
    dyn_rank_ratio = dyn_rank2 / max(dyn_rank1, 1)
    dyn_log_rank_ratio = np.log(max(dyn_rank_ratio, 0.01))
    elo_rank_ratio = elo_rank2 / max(elo_rank1, 1)
    log_elo_rank_ratio = np.log(max(elo_rank_ratio, 0.01))

    # Map features
    map_pool_depth1 = bundle.map_tracker.get_map_pool_depth(t1)
    map_pool_depth2 = bundle.map_tracker.get_map_pool_depth(t2)
    map_wr_overlap = bundle.map_tracker.get_overlap_winrate_diff(t1, t2)

    feat = {
        "elo_diff": elo1 - elo2,
        "rank_diff": rank2 - rank1,
        "rank_ratio": rank_ratio,
        "log_rank_ratio": log_rank_ratio,
        "dyn_rank_diff": dyn_rank2 - dyn_rank1,
        "dyn_log_rank_ratio": dyn_log_rank_ratio,
        "elo_rank_diff": elo_rank2 - elo_rank1,
        "log_elo_rank_ratio": log_elo_rank_ratio,
        "form_diff": form1 - form2,
        "streak_diff": streak1 - streak2,
        "h2h_winrate": h2h_rate,
        "h2h_matches": h2h_matches,
        "elo_expected": bundle.elo.expected_score(elo1, elo2),
        "momentum_diff": momentum1 - momentum2,
        "vs_strong_diff": vs_strong1 - vs_strong2,
        "map_pool_depth_diff": map_pool_depth1 - map_pool_depth2,
        "map_wr_overlap": map_wr_overlap,
    }

    # Rank volatility
    rank_vol_max = max(rank_vol1, rank_vol2)
    dyn_rank_diff_val = feat["dyn_rank_diff"]
    rank_confidence = abs(dyn_rank_diff_val) / (1.0 + rank_vol_max)
    feat["rank_vol_diff"] = rank_vol1 - rank_vol2
    feat["rank_vol_max"] = rank_vol_max
    feat["rank_trajectory_diff"] = rank_traj1 - rank_traj2
    feat["rank_confidence"] = rank_confidence
    feat["rank_conf_x_rank_diff"] = rank_confidence * np.sign(dyn_rank_diff_val)

    # --- Player features ---
    for team_key, team_name, player_names in [("t1", t1, t1_players), ("t2", t2, t2_players)]:
        if player_names is None:
            # Use last known roster from player_tracker
            player_names = _get_last_roster(bundle.player_tracker, team_name)

        if player_names:
            pfeat_list = []
            days_since = []
            roster_exp = []

            for pname in player_names:
                pf = bundle.player_tracker.get_player_features(pname)
                pfeat_list.append(pf)
                if date:
                    days_since.append(bundle.player_tracker.get_days_since_last(pname, date))
                roster_exp.append(bundle.player_tracker.get_roster_experience(pname, team_name))

            ratings = [p["avg_rating"] for p in pfeat_list]
            adrs = [p["avg_adr"] for p in pfeat_list]
            kasts = [p["avg_kast"] for p in pfeat_list]
            kd_diffs = [p["avg_kd_diff"] for p in pfeat_list]
            consistencies = [p["consistency"] for p in pfeat_list]

            feat[f"{team_key}_avg_rating"] = np.mean(ratings)
            feat[f"{team_key}_avg_adr"] = np.mean(adrs)
            feat[f"{team_key}_avg_kast"] = np.mean(kasts)
            feat[f"{team_key}_avg_kd_diff"] = np.mean(kd_diffs)
            feat[f"{team_key}_star_rating"] = max(ratings)
            feat[f"{team_key}_weakest_rating"] = min(ratings)
            feat[f"{team_key}_star_gap"] = max(ratings) - min(ratings)
            feat[f"{team_key}_consistency"] = np.mean(consistencies)
            feat[f"{team_key}_roster_exp"] = np.mean(roster_exp)
            feat[f"{team_key}_avg_rest"] = np.mean(days_since) if days_since else 14.0

            raw_chem = bundle.chemistry.get_team_chemistry(player_names, date)
            feat[f"{team_key}_chemistry"] = np.log1p(raw_chem)
        else:
            # Default player features
            feat[f"{team_key}_avg_rating"] = 1.0
            feat[f"{team_key}_avg_adr"] = 70.0
            feat[f"{team_key}_avg_kast"] = 70.0
            feat[f"{team_key}_avg_kd_diff"] = 0.0
            feat[f"{team_key}_star_rating"] = 1.0
            feat[f"{team_key}_weakest_rating"] = 1.0
            feat[f"{team_key}_star_gap"] = 0.0
            feat[f"{team_key}_consistency"] = 0.1
            feat[f"{team_key}_roster_exp"] = 0.0
            feat[f"{team_key}_avg_rest"] = 14.0
            feat[f"{team_key}_chemistry"] = 0.0

    # Difference features
    for stat in ["avg_rating", "avg_adr", "avg_kast", "avg_kd_diff",
                 "star_rating", "weakest_rating", "star_gap", "consistency"]:
        k1, k2 = f"t1_{stat}", f"t2_{stat}"
        feat[f"diff_{stat}"] = feat.get(k1, 0) - feat.get(k2, 0)
    feat["diff_avg_rest"] = feat.get("t1_avg_rest", 14.0) - feat.get("t2_avg_rest", 14.0)
    feat["diff_roster_exp"] = feat.get("t1_roster_exp", 0) - feat.get("t2_roster_exp", 0)
    feat["diff_chemistry"] = feat.get("t1_chemistry", 0) - feat.get("t2_chemistry", 0)

    # Home advantage
    feat["home_diff"] = get_home_advantage(t1, event) - get_home_advantage(t2, event)

    # BO format
    bo = bo_format.upper() if bo_format else "BO3"
    feat["is_bo1"] = 1 if bo == "BO1" else 0
    feat["is_bo3"] = 1 if bo == "BO3" else 0
    feat["is_bo5"] = 1 if bo == "BO5" else 0

    # Interaction features
    bo_weight = 1.0 if feat["is_bo1"] else (1.3 if feat["is_bo3"] else 1.5)
    feat["rank_diff_x_bo"] = feat["rank_diff"] * bo_weight
    feat["dyn_rank_diff_x_bo"] = feat["dyn_rank_diff"] * bo_weight
    feat["elo_rank_diff_x_bo"] = feat["elo_rank_diff"] * bo_weight
    feat["elo_diff_x_form"] = feat["elo_diff"] * feat["form_diff"] if feat["form_diff"] != 0 else 0
    feat["rank_x_h2h"] = feat["rank_diff"] * (feat["h2h_winrate"] - 0.5)
    feat["momentum_x_form"] = feat["momentum_diff"] * feat["form_diff"] if feat["form_diff"] != 0 else 0

    # Map upset potential (unknown map for upcoming match)
    feat["map_upset_potential"] = 0.0

    # Upset detector
    upset_prob = bundle.upset_detector.predict_upset_prob(feat)
    feat["upset_prob"] = upset_prob
    feat["upset_prob_x_rank_diff"] = upset_prob * dyn_rank_diff_val

    # Pistol
    pistol_wr1 = bundle.pistol_tracker.get_pistol_winrate(t1)
    pistol_wr2 = bundle.pistol_tracker.get_pistol_winrate(t2)
    feat["pistol_wr_diff"] = pistol_wr1 - pistol_wr2

    # Extract just the lean_cols in order
    return {col: feat.get(col, 0.0) for col in LEAN_COLS}


def _get_last_roster(player_tracker, team_name, n=5):
    """Get last known roster for a team from player history."""
    candidates = []
    for (player, team), count in player_tracker.roster_history.items():
        if team == team_name:
            hist = player_tracker.history.get(player, [])
            if hist:
                last_date = hist[-1].get("date")
                candidates.append((player, last_date, count))

    if not candidates:
        return None

    # Sort by most recent appearance, then by match count
    candidates.sort(key=lambda x: (x[1] is not None, x[1], x[2]), reverse=True)
    return [c[0] for c in candidates[:n]]
