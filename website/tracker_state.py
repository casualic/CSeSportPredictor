"""
TrackerBundle: wraps all 8 trackers and replays historical data to build current state.

Usage:
    python -m website.tracker_state          # initialize from historical CSVs
    python -m website.tracker_state --force  # re-initialize even if state exists
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

# Add project root to path so we can import tracker classes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model_fsvm import (
    EloSystem, FormTracker, H2HTracker, PlayerStatsTracker,
    MapStatsTracker, PistolTracker, ChemistryTracker, UpsetDetector,
    parse_hltv_date, build_rankings_index, get_rank_volatility_features,
    get_home_advantage,
    load_rankings_history, load_map_results, load_pistol_rounds,
    load_player_stats, load_match_details,
    PLAYER_WINDOW, MAP_WINDOW,
)
from website.config import (
    DATA_DIR, TRACKER_STATE_PATH,
)


class TrackerBundle:
    """Wraps all 8 trackers with save/load and incremental update."""

    def __init__(self):
        self.elo = EloSystem(k=32)
        self.form = FormTracker(window=10)
        self.h2h = H2HTracker()
        self.player_tracker = PlayerStatsTracker(window=PLAYER_WINDOW)
        self.map_tracker = MapStatsTracker(window=MAP_WINDOW)
        self.pistol_tracker = PistolTracker(window=30)
        self.chemistry = ChemistryTracker()
        self.upset_detector = UpsetDetector()
        self.rankings_index = {}
        self.top_teams = {}
        self.matches_processed = 0

    def initialize_from_history(self):
        """Replay all historical CSVs through trackers. Takes ~60s."""
        import json

        # Load match data (oldest first)
        matches_path = os.path.join(DATA_DIR, "all_results_with_urls.csv")
        if not os.path.exists(matches_path):
            matches_path = os.path.join(DATA_DIR, "top100_matches_with_urls.csv")
        df = pd.read_csv(matches_path)

        with open(os.path.join(DATA_DIR, "top_teams.json")) as f:
            self.top_teams = json.load(f)

        if "match_url" in df.columns:
            df["match_id"] = df["match_url"].apply(
                lambda u: str(u).split("/matches/")[1].split("/")[0]
                if pd.notna(u) and "/matches/" in str(u) else ""
            )
        else:
            df["match_id"] = ""

        df = df.iloc[::-1].reset_index(drop=True)  # oldest first
        print(f"Replaying {len(df)} historical matches...")

        # Load auxiliary data
        player_stats_df = load_player_stats()
        match_details_df = load_match_details()
        rankings_df = load_rankings_history()
        map_results_df = load_map_results()
        pistol_rounds_df = load_pistol_rounds()

        # Build indexes
        self.rankings_index = build_rankings_index(rankings_df) if rankings_df is not None else {}
        has_rankings = rankings_df is not None and not rankings_df.empty
        has_maps = map_results_df is not None and not map_results_df.empty
        has_pistol = pistol_rounds_df is not None and not pistol_rounds_df.empty

        match_players = {}
        if player_stats_df is not None:
            for mid, group in player_stats_df.groupby("match_id"):
                match_players[str(mid)] = group

        match_info = {}
        if match_details_df is not None:
            for _, row in match_details_df.iterrows():
                match_info[str(row["match_id"])] = row

        match_maps = {}
        if has_maps:
            for mid, group in map_results_df.groupby("match_id"):
                match_maps[str(mid)] = group

        match_pistol = {}
        if has_pistol:
            for mid, group in pistol_rounds_df.groupby("match_id"):
                match_pistol[str(mid)] = group.to_dict("records")

        # Replay each match (same logic as build_features tracker updates)
        for i, row in df.iterrows():
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i}/{len(df)} matches...")

            t1, t2 = row["team1"], row["team2"]
            winner = row["winner"]
            event = row.get("event", "")
            match_id = str(row.get("match_id", ""))
            date = parse_hltv_date(row.get("date", ""))

            mi = match_info.get(match_id)
            if mi is not None and date is None:
                date = parse_hltv_date(mi.get("date", ""))

            # Compute features needed for upset_detector update
            rank1 = self.top_teams.get(t1, 101)
            rank2 = self.top_teams.get(t2, 101)
            if has_rankings:
                dyn_rank1, _, _ = get_rank_volatility_features(t1, date, self.rankings_index)
                dyn_rank2, _, _ = get_rank_volatility_features(t2, date, self.rankings_index)
            else:
                dyn_rank1, dyn_rank2 = rank1, rank2

            dyn_rank_diff_val = dyn_rank2 - dyn_rank1

            # Build partial feature dict for upset detector
            feat = {
                "form_diff": self.form.get_form(t1) - self.form.get_form(t2),
                "momentum_diff": self.elo.get_momentum(t1) - self.elo.get_momentum(t2),
                "streak_diff": self.form.get_streak(t1) - self.form.get_streak(t2),
                "h2h_winrate": self.h2h.get_h2h(t1, t2),
                "map_pool_depth_diff": self.map_tracker.get_map_pool_depth(t1) - self.map_tracker.get_map_pool_depth(t2),
                "map_wr_overlap": self.map_tracker.get_overlap_winrate_diff(t1, t2),
                "vs_strong_diff": self.elo.get_winrate_vs_strong(t1) - self.elo.get_winrate_vs_strong(t2),
                "diff_avg_rest": 0.0,
                "diff_chemistry": 0.0,
            }

            # BO format
            if mi is not None:
                bo = str(mi.get("bo_format", "BO3"))
            else:
                s1, s2 = row.get("score1", 0), row.get("score2", 0)
                total = int(s1) + int(s2) if pd.notna(s1) and pd.notna(s2) else 0
                bo = "BO1" if total == 1 else "BO3" if 2 <= total <= 3 else "BO5" if total >= 4 else "BO3"
            feat["is_bo1"] = 1 if bo == "BO1" else 0
            feat["is_bo3"] = 1 if bo == "BO3" else 0
            feat["is_bo5"] = 1 if bo == "BO5" else 0

            # Update upset detector
            label = 1 if winner == t1 else 0
            was_upset = (dyn_rank_diff_val > 0 and label == 0) or (dyn_rank_diff_val < 0 and label == 1)
            self.upset_detector.update(feat, was_upset)

            # Update trackers
            loser = t2 if winner == t1 else t1
            maps_played = mi["maps_played"] if mi is not None else 1
            try:
                maps_played = int(maps_played)
            except (ValueError, TypeError):
                maps_played = 1

            self.elo.update(winner, loser, maps_played=maps_played)
            self.form.update(winner, loser)
            self.h2h.update(winner, loser)

            # Player stats + chemistry
            players_data = match_players.get(match_id)
            if players_data is not None:
                for team_name in [t1, t2]:
                    tp = players_data[players_data["team"] == team_name]
                    if tp.empty:
                        half = len(players_data) // 2
                        tp = players_data.iloc[:half] if team_name == t1 else players_data.iloc[half:]
                    names = tp["player"].tolist()
                    self.chemistry.update(names, date)

                for _, p in players_data.iterrows():
                    self.player_tracker.update(
                        player=p["player"], team=p["team"],
                        rating=p.get("rating", 1.0), adr=p.get("adr", 70.0),
                        kast=p.get("kast", 70.0), kills=p.get("kills", 0),
                        deaths=p.get("deaths", 0), date=date,
                    )

            # Map tracker
            maps_data = match_maps.get(match_id)
            if maps_data is not None:
                for _, mp in maps_data.iterrows():
                    map_name = mp["map_name"]
                    map_winner_name = mp["map_winner"]
                    self.map_tracker.update(t1, map_name, map_winner_name == t1)
                    self.map_tracker.update(t2, map_name, map_winner_name == t2)

            # Pistol tracker
            pistol_data = match_pistol.get(match_id)
            if pistol_data is not None:
                for pr in pistol_data:
                    for pw_key in ["pistol1_winner", "pistol2_winner"]:
                        pw = str(pr.get(pw_key, "")).strip()
                        if pw:
                            pt1 = str(pr.get("team1", "")).strip()
                            pt2 = str(pr.get("team2", "")).strip()
                            if pt1:
                                self.pistol_tracker.update(pt1, pw == pt1)
                            if pt2:
                                self.pistol_tracker.update(pt2, pw == pt2)

            self.matches_processed += 1

        print(f"Done. Processed {self.matches_processed} matches.")

    def save(self, path=None):
        """Save tracker state to disk using pickle (required for complex tracker objects)."""
        path = path or TRACKER_STATE_PATH
        # Convert lambda defaultdicts to regular dicts before pickling
        orig_wins = self.elo.wins_vs_strong
        self.elo.wins_vs_strong = dict(orig_wins)
        orig_map_hist = self.map_tracker.history
        self.map_tracker.history = {k: dict(v) for k, v in orig_map_hist.items()}
        with open(path, "wb") as f:
            pickle.dump(self, f)
        # Restore after save
        self.elo.wins_vs_strong = orig_wins
        self.map_tracker.history = orig_map_hist
        print(f"Tracker state saved to {path}")

    @staticmethod
    def load(path=None):
        """Load tracker state from disk. Only loads files we created ourselves."""
        import train_model_fsvm as _tsm
        from website import tracker_state as _ts

        class _BundleUnpickler(pickle.Unpickler):
            """Redirect __main__ and website.tracker_state classes for pickle compat."""
            _CLASS_MAP = {
                ("__main__", "TrackerBundle"): _ts.TrackerBundle,
                ("__main__", "EloSystem"): _tsm.EloSystem,
                ("__main__", "FormTracker"): _tsm.FormTracker,
                ("__main__", "H2HTracker"): _tsm.H2HTracker,
                ("__main__", "PlayerStatsTracker"): _tsm.PlayerStatsTracker,
                ("__main__", "MapStatsTracker"): _tsm.MapStatsTracker,
                ("__main__", "PistolTracker"): _tsm.PistolTracker,
                ("__main__", "ChemistryTracker"): _tsm.ChemistryTracker,
                ("__main__", "UpsetDetector"): _tsm.UpsetDetector,
                ("__main__", "FuzzySVM"): _tsm.FuzzySVM,
            }

            def find_class(self, module, name):
                cls = self._CLASS_MAP.get((module, name))
                if cls is not None:
                    return cls
                return super().find_class(module, name)

        path = path or TRACKER_STATE_PATH
        with open(path, "rb") as f:
            bundle = _BundleUnpickler(f).load()
        # Restore lambda defaultdicts
        wins = defaultdict(lambda: [0, 0])
        wins.update(bundle.elo.wins_vs_strong)
        bundle.elo.wins_vs_strong = wins
        map_hist = defaultdict(lambda: defaultdict(list))
        for k, v in bundle.map_tracker.history.items():
            map_hist[k] = defaultdict(list, v)
        bundle.map_tracker.history = map_hist
        print(f"Tracker state loaded ({bundle.matches_processed} matches)")
        return bundle

    def update_with_result(self, match_dict):
        """Incrementally update trackers with a new match result.

        match_dict keys: team1, team2, winner, event, date, bo_format,
                         players (optional list of dicts), maps (optional list of dicts),
                         pistol_rounds (optional list of dicts)
        """
        t1 = match_dict["team1"]
        t2 = match_dict["team2"]
        winner = match_dict["winner"]
        loser = t2 if winner == t1 else t1
        date = parse_hltv_date(match_dict.get("date", ""))
        bo = match_dict.get("bo_format", "BO3")

        maps_played = match_dict.get("maps_played", 1)
        try:
            maps_played = int(maps_played)
        except (ValueError, TypeError):
            maps_played = 1

        self.elo.update(winner, loser, maps_played=maps_played)
        self.form.update(winner, loser)
        self.h2h.update(winner, loser)

        # Player data
        players = match_dict.get("players", [])
        if players:
            t1_names = [p["player"] for p in players if p.get("team") == t1]
            t2_names = [p["player"] for p in players if p.get("team") == t2]
            self.chemistry.update(t1_names, date)
            self.chemistry.update(t2_names, date)
            for p in players:
                self.player_tracker.update(
                    player=p["player"], team=p.get("team", ""),
                    rating=p.get("rating", 1.0), adr=p.get("adr", 70.0),
                    kast=p.get("kast", 70.0), kills=p.get("kills", 0),
                    deaths=p.get("deaths", 0), date=date,
                )

        # Map data
        for mp in match_dict.get("maps", []):
            map_name = mp["map_name"]
            map_winner_name = mp["map_winner"]
            self.map_tracker.update(t1, map_name, map_winner_name == t1)
            self.map_tracker.update(t2, map_name, map_winner_name == t2)

        # Pistol data
        for pr in match_dict.get("pistol_rounds", []):
            for pw_key in ["pistol1_winner", "pistol2_winner"]:
                pw = str(pr.get(pw_key, "")).strip()
                if pw:
                    pt1 = str(pr.get("team1", "")).strip()
                    pt2 = str(pr.get("team2", "")).strip()
                    if pt1:
                        self.pistol_tracker.update(pt1, pw == pt1)
                    if pt2:
                        self.pistol_tracker.update(pt2, pw == pt2)

        self.matches_processed += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-initialize even if state exists")
    args = parser.parse_args()

    if os.path.exists(TRACKER_STATE_PATH) and not args.force:
        print(f"Tracker state already exists at {TRACKER_STATE_PATH}")
        print("Use --force to re-initialize.")
        bundle = TrackerBundle.load()
    else:
        bundle = TrackerBundle()
        bundle.initialize_from_history()
        bundle.save()
