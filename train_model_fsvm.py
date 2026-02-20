import gc
import pandas as pd
import numpy as np
import json
import os
import sys
import re
from collections import defaultdict
from itertools import combinations
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PLAYER_WINDOW = 10  # rolling window for player stats
MAP_WINDOW = 20  # rolling window for map stats


# ---------------------------------------------------------------------------
# Fuzzy SVM (Lin & Wang, 2002)
#
# Standard SVM: min (1/2)||w||^2 + C * SUM(xi_i)
# Fuzzy SVM:    min (1/2)||w||^2 + C * SUM(s_i * xi_i)
#
# Each sample i gets a fuzzy membership s_i in (0, 1].
# In the dual, the box constraint becomes 0 <= alpha_i <= s_i * C.
# sklearn's SVC.fit(sample_weight=s) does exactly this.
# ---------------------------------------------------------------------------

class FuzzySVM(BaseEstimator, ClassifierMixin):
    """Fuzzy Support Vector Machine using sklearn SVC with membership weighting.

    Membership strategies:
        - 'class_center': samples far from class centroid get lower weight
        - 'time_decay': older samples get lower weight (exponential decay)
        - 'confidence': samples in ambiguous regions (high KNN entropy) get lower weight
        - 'hybrid': multiplicative combination of time_decay and class_center
    """

    def __init__(self, C=1.0, kernel='rbf', gamma='scale',
                 membership='hybrid', lambda_decay=0.005,
                 delta=1e-4, sigma_floor=0.1, knn_k=7,
                 hybrid_alpha=0.5, random_state=42):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.membership = membership
        self.lambda_decay = lambda_decay
        self.delta = delta
        self.sigma_floor = sigma_floor
        self.knn_k = knn_k
        self.hybrid_alpha = hybrid_alpha
        self.random_state = random_state
        self.svc_ = None
        self.sample_weight_ = None

    def _membership_class_center(self, X, y):
        """s_i = 1 - d_i / (r_k + delta). Outliers from class center get lower weight."""
        s = np.ones(len(X))
        for label in np.unique(y):
            mask = (y == label)
            X_class = X[mask]
            centroid = X_class.mean(axis=0)
            distances = np.linalg.norm(X_class - centroid, axis=1)
            radius = distances.max()
            s[mask] = 1.0 - distances / (radius + self.delta)
        return np.clip(s, self.sigma_floor, 1.0)

    def _membership_time_decay(self, X, y, timestamps):
        """s_i = exp(-lambda * age). Recent matches weighted higher."""
        t_max = timestamps.max()
        age = t_max - timestamps
        s = np.exp(-self.lambda_decay * age)
        return np.clip(s, self.sigma_floor, 1.0)

    def _membership_confidence(self, X, y):
        """KNN entropy-based. Ambiguous samples (mixed neighborhoods) get lower weight."""
        k = min(self.knn_k, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, indices = nn.kneighbors(X)
        neighbor_labels = y[indices[:, 1:]]  # exclude self

        s = np.ones(len(X))
        for i in range(len(X)):
            same = np.sum(neighbor_labels[i] == y[i])
            p = same / k
            if p == 0 or p == 1:
                entropy = 0.0
            else:
                entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
            s[i] = 1.0 - entropy
        return np.clip(s, self.sigma_floor, 1.0)

    def _membership_hybrid(self, X, y, timestamps):
        """Multiplicative: s = s_time * s_distance. Both recency and typicality matter."""
        s_time = self._membership_time_decay(X, y, timestamps)
        s_dist = self._membership_class_center(X, y)
        s = self.hybrid_alpha * s_time + (1 - self.hybrid_alpha) * s_dist
        return np.clip(s, self.sigma_floor, 1.0)

    def compute_membership(self, X, y, timestamps=None):
        """Compute fuzzy membership values for all samples."""
        if self.membership == 'class_center':
            return self._membership_class_center(X, y)
        elif self.membership == 'time_decay':
            if timestamps is None:
                timestamps = np.arange(len(X), dtype=float)
            return self._membership_time_decay(X, y, timestamps)
        elif self.membership == 'confidence':
            return self._membership_confidence(X, y)
        elif self.membership == 'hybrid':
            if timestamps is None:
                timestamps = np.arange(len(X), dtype=float)
            return self._membership_hybrid(X, y, timestamps)
        else:
            raise ValueError(f"Unknown membership: {self.membership}")

    def fit(self, X, y, sample_weight=None, timestamps=None):
        """Fit Fuzzy SVM. Computes membership and passes as sample_weight to SVC."""
        s = self.compute_membership(X, y, timestamps)
        # If external sample_weight provided (e.g. time decay from training loop),
        # multiply with fuzzy membership
        if sample_weight is not None:
            s = s * sample_weight
            s = s / s.sum() * len(s)  # re-normalize
        self.sample_weight_ = s

        self.svc_ = SVC(
            C=self.C, kernel=self.kernel, gamma=self.gamma,
            probability=True, random_state=self.random_state,
        )
        self.svc_.fit(X, y, sample_weight=s)
        return self

    def predict(self, X):
        return self.svc_.predict(X)

    def predict_proba(self, X):
        return self.svc_.predict_proba(X)

    def decision_function(self, X):
        return self.svc_.decision_function(X)


# ---------------------------------------------------------------------------
# Data loaders for new data sources
# ---------------------------------------------------------------------------

def load_rankings_history():
    """Load weekly HLTV ranking history CSV."""
    path = os.path.join(DATA_DIR, "rankings_history.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_map_results():
    """Load map-level results CSV."""
    path = os.path.join(DATA_DIR, "map_results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["match_id"] = df["match_id"].astype(str)
    return df


def load_pistol_rounds():
    """Load pistol round data CSV."""
    path = os.path.join(DATA_DIR, "pistol_rounds.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["match_id"] = df["match_id"].astype(str)
    return df


def get_dynamic_rank(team, match_date, rankings_df):
    """Look up team's HLTV rank from the most recent ranking date <= match_date."""
    if rankings_df is None or match_date is None:
        return 150
    team_rows = rankings_df[rankings_df["team"] == team]
    if team_rows.empty:
        return 150
    valid = team_rows[team_rows["date"] <= match_date]
    if valid.empty:
        return 150
    latest = valid.loc[valid["date"].idxmax()]
    return int(latest["rank"])


def build_rankings_index(rankings_df):
    """Pre-build {team -> [(date, rank), ...]} dict from rankings_df.

    One-time O(n) pass — replaces per-match DataFrame filtering.
    """
    index = defaultdict(list)
    if rankings_df is None:
        return index
    for _, row in rankings_df.iterrows():
        index[row["team"]].append((row["date"], int(row["rank"])))
    # Sort each team's entries by date for efficient lookup
    for team in index:
        index[team].sort(key=lambda x: x[0])
    return index


def get_rank_volatility_features(team, match_date, rankings_index):
    """Look up rank volatility features from pre-built rankings index.

    Uses last 4 weekly ranking snapshots before match_date.
    Returns:
        current_rank: most recent rank (replaces get_dynamic_rank)
        rank_volatility: std dev of ranks in 4-week window
        rank_trajectory: linear slope (negative = improving)
    """
    default = (150, 0.0, 0.0)
    if match_date is None:
        return default
    entries = rankings_index.get(team)
    if not entries:
        return default

    # Find entries before match_date
    valid = [(d, r) for d, r in entries if d <= match_date]
    if not valid:
        return default

    current_rank = valid[-1][1]

    # Last 4 snapshots
    recent = valid[-4:]
    ranks = [r for _, r in recent]

    if len(ranks) < 2:
        return (current_rank, 0.0, 0.0)

    rank_volatility = float(np.std(ranks))
    # Linear slope via simple regression: slope = (y_n - y_0) / (n-1)
    # Negative slope = improving (rank going down = getting better)
    rank_trajectory = (ranks[-1] - ranks[0]) / (len(ranks) - 1)

    return (current_rank, rank_volatility, rank_trajectory)


# ---------------------------------------------------------------------------
# Team-level trackers (retained from iteration 1)
# ---------------------------------------------------------------------------

class EloSystem:
    def __init__(self, k=32, default_rating=1500):
        self.k = k
        self.default_rating = default_rating
        self.ratings = {}
        self.match_counts = defaultdict(int)
        self.rating_history = defaultdict(list)  # track last N ratings for momentum
        self.wins_vs_strong = defaultdict(lambda: [0, 0])  # [wins, total] vs top-Elo teams

    def get_rating(self, team):
        return self.ratings.get(team, self.default_rating)

    def expected_score(self, ra, rb):
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    def get_elo_rank(self, team):
        """Dynamic rank based on current Elo (1=best). No future leakage."""
        if not self.ratings:
            return 50
        sorted_teams = sorted(self.ratings.items(), key=lambda x: -x[1])
        for i, (t, _) in enumerate(sorted_teams):
            if t == team:
                return i + 1
        return len(sorted_teams) + 1

    def get_momentum(self, team, window=5):
        """Elo trend: positive = improving, negative = declining."""
        hist = self.rating_history.get(team, [])
        if len(hist) < 2:
            return 0.0
        recent = hist[-window:]
        if len(recent) < 2:
            return 0.0
        return recent[-1] - recent[0]

    def get_winrate_vs_strong(self, team, min_games=2):
        """Win rate against top-Elo opponents."""
        w, t = self.wins_vs_strong[team]
        if t < min_games:
            return 0.5
        return w / t

    def update(self, winner, loser, maps_played=1):
        ra, rb = self.get_rating(winner), self.get_rating(loser)
        ea = self.expected_score(ra, rb)
        # Weight K-factor: BO3/BO5 victories worth more; new teams have higher K
        map_weight = 1.0 + 0.15 * (maps_played - 1)  # BO3(3maps)=1.3x, BO5(5maps)=1.6x
        # Adaptive K: higher for teams with fewer matches (faster convergence)
        n_w = self.match_counts[winner]
        n_l = self.match_counts[loser]
        k_w = self.k * (1.5 if n_w < 10 else 1.2 if n_w < 30 else 1.0)
        k_l = self.k * (1.5 if n_l < 10 else 1.2 if n_l < 30 else 1.0)
        self.ratings[winner] = ra + k_w * map_weight * (1 - ea)
        self.ratings[loser] = rb + k_l * map_weight * (0 - (1 - ea))
        self.match_counts[winner] += 1
        self.match_counts[loser] += 1
        # Track history for momentum
        self.rating_history[winner].append(self.ratings[winner])
        self.rating_history[loser].append(self.ratings[loser])
        # Track wins vs strong opponents (top-30 Elo)
        top_threshold = sorted(self.ratings.values(), reverse=True)[:30][-1] if len(self.ratings) >= 30 else self.default_rating
        if rb >= top_threshold:
            self.wins_vs_strong[winner][0] += 1
            self.wins_vs_strong[winner][1] += 1
        if ra >= top_threshold:
            self.wins_vs_strong[loser][1] += 1


class FormTracker:
    def __init__(self, window=10):
        self.window = window
        self.history = {}

    def get_form(self, team):
        hist = self.history.get(team, [])
        if not hist:
            return 0.5
        recent = hist[-self.window:]
        return sum(recent) / len(recent)

    def get_streak(self, team):
        hist = self.history.get(team, [])
        if not hist:
            return 0
        streak = 0
        last = hist[-1]
        for r in reversed(hist):
            if r == last:
                streak += 1
            else:
                break
        return streak if last == 1 else -streak

    def update(self, winner, loser):
        self.history.setdefault(winner, []).append(1)
        self.history.setdefault(loser, []).append(0)


class H2HTracker:
    def __init__(self):
        self.records = {}

    def get_h2h(self, t1, t2):
        key = tuple(sorted([t1, t2]))
        if key not in self.records:
            return 0.5
        w = self.records[key]
        total = w[0] + w[1]
        if total == 0:
            return 0.5
        return w[0] / total if key[0] == t1 else w[1] / total

    def get_total(self, t1, t2):
        key = tuple(sorted([t1, t2]))
        if key not in self.records:
            return 0
        return sum(self.records[key])

    def update(self, winner, loser):
        key = tuple(sorted([winner, loser]))
        if key not in self.records:
            self.records[key] = [0, 0]
        if key[0] == winner:
            self.records[key][0] += 1
        else:
            self.records[key][1] += 1


# ---------------------------------------------------------------------------
# Player-level tracker
# ---------------------------------------------------------------------------

class PlayerStatsTracker:
    """Track per-player rolling statistics across matches."""

    def __init__(self, window=PLAYER_WINDOW):
        self.window = window
        # player -> list of {rating, adr, kast, kd_diff, date, team}
        self.history = defaultdict(list)
        # (player, team) -> match count together
        self.roster_history = defaultdict(int)

    def get_player_features(self, player):
        hist = self.history.get(player, [])
        if not hist:
            return {
                "avg_rating": 1.0, "avg_adr": 70.0, "avg_kast": 70.0,
                "avg_kd_diff": 0.0, "consistency": 0.1, "n_matches": 0
            }
        recent = hist[-self.window:]
        ratings = [h["rating"] for h in recent]
        return {
            "avg_rating": np.mean(ratings),
            "avg_adr": np.mean([h["adr"] for h in recent]),
            "avg_kast": np.mean([h["kast"] for h in recent]),
            "avg_kd_diff": np.mean([h["kd_diff"] for h in recent]),
            "consistency": np.std(ratings) if len(ratings) > 1 else 0.1,
            "n_matches": len(hist),
        }

    def get_days_since_last(self, player, current_date):
        hist = self.history.get(player, [])
        if not hist or current_date is None:
            return 14  # default: 2 weeks
        last_date = hist[-1].get("date")
        if last_date is None:
            return 14
        diff = (current_date - last_date).days
        return max(0, diff)

    def get_roster_experience(self, player, team):
        return self.roster_history.get((player, team), 0)

    def update(self, player, team, rating, adr, kast, kills, deaths, date):
        self.history[player].append({
            "rating": rating, "adr": adr, "kast": kast,
            "kd_diff": kills - deaths, "date": date, "team": team
        })
        self.roster_history[(player, team)] += 1


# ---------------------------------------------------------------------------
# Map-level tracker
# ---------------------------------------------------------------------------

class MapStatsTracker:
    """Track per-team rolling map win rates."""

    def __init__(self, window=MAP_WINDOW):
        self.window = window
        # team -> map_name -> list of bools (True=win)
        self.history = defaultdict(lambda: defaultdict(list))

    def get_team_map_winrate(self, team, map_name):
        hist = self.history[team][map_name]
        if not hist:
            return 0.5
        recent = hist[-self.window:]
        return sum(recent) / len(recent)

    def get_map_pool_depth(self, team, min_plays=5):
        """How many maps this team has played at least min_plays times."""
        count = 0
        for map_name, hist in self.history[team].items():
            if len(hist) >= min_plays:
                count += 1
        return count

    def get_common_maps(self, t1, t2, min_plays=3):
        """Maps both teams have played at least min_plays times."""
        maps1 = {m for m, h in self.history[t1].items() if len(h) >= min_plays}
        maps2 = {m for m, h in self.history[t2].items() if len(h) >= min_plays}
        return maps1 & maps2

    def get_overlap_winrate_diff(self, t1, t2, min_plays=3):
        """Average win rate difference on maps both teams play."""
        common = self.get_common_maps(t1, t2, min_plays)
        if not common:
            return 0.0
        diffs = []
        for m in common:
            wr1 = self.get_team_map_winrate(t1, m)
            wr2 = self.get_team_map_winrate(t2, m)
            diffs.append(wr1 - wr2)
        return np.mean(diffs)

    def update(self, team, map_name, won):
        self.history[team][map_name].append(won)


# ---------------------------------------------------------------------------
# Pistol round tracker
# ---------------------------------------------------------------------------

class PistolTracker:
    """Track per-team rolling pistol round win rates."""

    def __init__(self, window=30):
        self.window = window
        # team -> list of bools (True = won pistol)
        self.history = defaultdict(list)

    def get_pistol_winrate(self, team):
        hist = self.history.get(team, [])
        if not hist:
            return 0.5
        recent = hist[-self.window:]
        return sum(recent) / len(recent)

    def update(self, team, won_pistol):
        self.history[team].append(won_pistol)


# ---------------------------------------------------------------------------
# Team chemistry tracker (pairwise days together)
# ---------------------------------------------------------------------------

class ChemistryTracker:
    """Track how long each pair of players has been teammates.

    For each match, computes the average number of days each pair of the 5
    players on a team has played together (since their first co-appearance).
    This serves as a proxy for team chemistry / lineup stability.
    """

    def __init__(self):
        # frozenset({player1, player2}) -> first date they appeared on the same team
        self.pair_first_seen = {}

    def get_team_chemistry(self, player_names, match_date):
        """Return avg days together for all pairs on this team.

        Args:
            player_names: list of player name strings (typically 5)
            match_date: datetime of the current match

        Returns:
            avg_days: average days together across all C(n,2) pairs
        """
        if not player_names or match_date is None or len(player_names) < 2:
            return 0.0

        pairs = list(combinations(player_names, 2))
        days_list = []
        for p1, p2 in pairs:
            key = frozenset({p1, p2})
            first = self.pair_first_seen.get(key)
            if first is not None:
                days = (match_date - first).days
            else:
                days = 0
            days_list.append(days)

        return np.mean(days_list) if days_list else 0.0

    def update(self, player_names, match_date):
        """Record first co-appearance for all pairs in this lineup."""
        if not player_names or match_date is None:
            return
        for p1, p2 in combinations(player_names, 2):
            key = frozenset({p1, p2})
            if key not in self.pair_first_seen:
                self.pair_first_seen[key] = match_date


# ---------------------------------------------------------------------------
# Upset detector (internal LR on non-rank features)
# ---------------------------------------------------------------------------

class UpsetDetector:
    """Predict P(upset) using only non-rank features.

    Trains an internal LogisticRegression(C=0.1) on features that don't
    include rank information, so the main model can learn when rank is
    unreliable.

    Follows the same get/update pattern as other trackers.
    """

    NON_RANK_FEATURES = [
        "form_diff", "momentum_diff", "streak_diff", "h2h_winrate",
        "map_pool_depth_diff", "map_wr_overlap", "is_bo1", "is_bo3", "is_bo5",
        "vs_strong_diff", "diff_avg_rest", "diff_chemistry",
    ]

    def __init__(self, min_samples=200, retrain_interval=100):
        self.min_samples = min_samples
        self.retrain_interval = retrain_interval
        self.X_history = []
        self.y_history = []
        self.model = None
        self.scaler = StandardScaler()
        self._samples_since_train = 0

    def _extract_features(self, feat_dict):
        """Extract non-rank feature vector from a feat dict."""
        return [feat_dict.get(f, 0.0) for f in self.NON_RANK_FEATURES]

    def predict_upset_prob(self, feat_dict):
        """Predict P(upset) from current historical model. Returns 0.5 if not enough data."""
        if self.model is None or len(self.X_history) < self.min_samples:
            return 0.5
        x = np.array(self._extract_features(feat_dict)).reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        return float(self.model.predict_proba(x_scaled)[0, 1])

    def update(self, feat_dict, was_upset):
        """Record outcome and retrain if needed."""
        self.X_history.append(self._extract_features(feat_dict))
        self.y_history.append(int(was_upset))
        self._samples_since_train += 1

        if (len(self.X_history) >= self.min_samples and
                self._samples_since_train >= self.retrain_interval):
            self._retrain()

    def _retrain(self):
        """Retrain internal LR on all accumulated data."""
        X = np.array(self.X_history)
        y = np.array(self.y_history)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = LogisticRegression(C=0.1, max_iter=500, solver="lbfgs")
        self.model.fit(X_scaled, y)
        self._samples_since_train = 0


# ---------------------------------------------------------------------------
# Home/away inference
# ---------------------------------------------------------------------------

EVENT_REGION_KEYWORDS = {
    "Katowice": "EU", "Cologne": "EU", "Copenhagen": "EU", "Stockholm": "EU",
    "Paris": "EU", "Antwerp": "EU", "Berlin": "EU", "London": "EU",
    "Bucharest": "EU", "Malta": "EU", "Belgrade": "EU", "Lanxess": "EU",
    "Rotterdam": "EU", "Jönköping": "EU", "Valencia": "EU",
    "Atlanta": "NA", "Dallas": "NA", "Los Angeles": "NA", "Austin": "NA",
    "Denver": "NA", "Las Vegas": "NA", "New York": "NA", "Columbus": "NA",
    "Arlington": "NA", "Boston": "NA",
    "Rio": "SA", "São Paulo": "SA", "Sao Paulo": "SA",
    "Shanghai": "ASIA", "Beijing": "ASIA", "Bangkok": "ASIA",
    "Tokyo": "ASIA", "Seoul": "ASIA",
    "Sydney": "OCE", "Melbourne": "OCE",
}

TEAM_REGION = {
    "Vitality": "EU", "G2": "EU", "FaZe": "EU", "Natus Vincere": "EU",
    "MOUZ": "EU", "HEROIC": "EU", "Astralis": "EU", "BIG": "EU",
    "fnatic": "EU", "ENCE": "EU", "Virtus.pro": "EU", "3DMAX": "EU",
    "GamerLegion": "EU", "Monte": "EU", "Falcons": "EU", "OG": "EU",
    "BetBoom": "EU", "Aurora": "EU", "Nemiga": "EU", "B8": "EU",
    "SINNERS": "EU", "Apogee": "EU", "SAW": "EU", "ECSTATIC": "EU",
    "FAVBET": "EU", "Alliance": "EU", "Endpoint": "EU", "ALTERNATE aTTaX": "EU",
    "Johnny Speeds": "EU", "BET-M": "EU", "Passion UA": "EU",
    "Inner Circle": "EU", "los kogutos": "EU", "Zero Tenacity": "EU",
    "Spirit": "EU", "Cloud9": "EU", "Rebels": "EU",
    "Liquid": "NA", "Complexity": "NA", "NRG": "NA", "MIBR": "NA",
    "Nouns": "NA", "M80": "NA", "Party Astronauts": "NA", "Wildcard": "NA",
    "Voca": "NA", "Legacy": "NA", "timbermen": "NA", "Take Flyte": "NA",
    "FURIA": "SA", "paiN": "SA", "Imperial": "SA", "RED Canids": "SA",
    "Sharks": "SA", "Solid": "SA", "Dusty Roots": "SA", "KRÜ": "SA",
    "9z": "SA", "BESTIA": "SA", "Galorys": "SA",
    "The MongolZ": "ASIA", "TYLOO": "ASIA", "Lynn Vision": "ASIA",
    "Rare Atom": "ASIA", "GR": "ASIA",
    "Rooster": "OCE", "Grayhound": "OCE",
}


def infer_event_region(event_name):
    for keyword, region in EVENT_REGION_KEYWORDS.items():
        if keyword.lower() in event_name.lower():
            return region
    return None


def get_home_advantage(team, event_name):
    team_region = TEAM_REGION.get(team)
    event_region = infer_event_region(event_name)
    if team_region and event_region:
        return 1.0 if team_region == event_region else 0.0
    return 0.5


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_hltv_date(date_str):
    """Parse HLTV date strings like '11th of February 2026' or 'Results for February 11th 2026'."""
    if not date_str or pd.isna(date_str):
        return None
    s = str(date_str)
    # Remove ordinal suffixes
    s = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", s)
    # Try "Results for Month Day Year"
    s = s.replace("Results for ", "")
    # Try "Day of Month Year"
    s = s.replace(" of ", " ")
    for fmt in ("%d %B %Y", "%B %d %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Feature engineering (with player stats)
# ---------------------------------------------------------------------------

def load_player_stats():
    """Load player stats CSV and index by match_id."""
    path = os.path.join(DATA_DIR, "player_stats.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["match_id"] = df["match_id"].astype(str)
    for col in ["kills", "deaths", "adr", "kast", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def load_match_details():
    """Load match details CSV and index by match_id."""
    path = os.path.join(DATA_DIR, "match_details.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["match_id"] = df["match_id"].astype(str)
    return df


def build_features(df, top_teams, player_stats_df=None, match_details_df=None,
                    rankings_df=None, map_results_df=None, pistol_rounds_df=None):
    """Build rich feature matrix combining team-level and player-level features."""
    has_player_data = player_stats_df is not None and not player_stats_df.empty
    has_rankings = rankings_df is not None and not rankings_df.empty
    has_maps = map_results_df is not None and not map_results_df.empty
    has_pistol = pistol_rounds_df is not None and not pistol_rounds_df.empty

    elo = EloSystem(k=32)
    form = FormTracker(window=10)
    h2h = H2HTracker()
    ptracker = PlayerStatsTracker(window=PLAYER_WINDOW)
    mtracker = MapStatsTracker(window=MAP_WINDOW)
    chem = ChemistryTracker()
    upset_detector = UpsetDetector()
    pistol_tracker = PistolTracker(window=30)

    # Pre-build rankings index for O(1) lookups
    rankings_index = build_rankings_index(rankings_df) if has_rankings else {}

    # Build match_id -> player stats lookup
    match_players = {}
    if has_player_data:
        for mid, group in player_stats_df.groupby("match_id"):
            match_players[str(mid)] = group

    # Build match_id -> match details lookup
    match_info = {}
    if match_details_df is not None:
        for _, row in match_details_df.iterrows():
            match_info[str(row["match_id"])] = row

    # Build match_id -> map results lookup
    match_maps = {}
    if has_maps:
        for mid, group in map_results_df.groupby("match_id"):
            match_maps[str(mid)] = group

    # Build match_id -> pistol rounds lookup
    match_pistol = {}
    if has_pistol:
        for mid, group in pistol_rounds_df.groupby("match_id"):
            match_pistol[str(mid)] = group.to_dict("records")

    features = []
    labels = []
    has_player_mask = []
    matches_with_players = 0

    for _, row in df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        winner = row["winner"]
        event = row.get("event", "")
        match_id = str(row.get("match_id", ""))
        date = parse_hltv_date(row.get("date", ""))

        # Get match detail info if available
        mi = match_info.get(match_id)
        if mi is not None and date is None:
            date = parse_hltv_date(mi.get("date", ""))

        # --- Team-level features ---
        elo1 = elo.get_rating(t1)
        elo2 = elo.get_rating(t2)
        # Static HLTV rank (potential leakage but strong signal)
        rank1 = top_teams.get(t1, 101)
        rank2 = top_teams.get(t2, 101)
        # Dynamic HLTV rank + volatility (no leakage — uses ranking at time of match)
        if has_rankings:
            dyn_rank1, rank_vol1, rank_traj1 = get_rank_volatility_features(t1, date, rankings_index)
            dyn_rank2, rank_vol2, rank_traj2 = get_rank_volatility_features(t2, date, rankings_index)
        else:
            dyn_rank1, rank_vol1, rank_traj1 = rank1, 0.0, 0.0
            dyn_rank2, rank_vol2, rank_traj2 = rank2, 0.0, 0.0
        # Dynamic Elo-based rank (no leakage)
        elo_rank1 = elo.get_elo_rank(t1)
        elo_rank2 = elo.get_elo_rank(t2)
        form1 = form.get_form(t1)
        form2 = form.get_form(t2)
        streak1 = form.get_streak(t1)
        streak2 = form.get_streak(t2)
        h2h_rate = h2h.get_h2h(t1, t2)
        h2h_matches = h2h.get_total(t1, t2)
        momentum1 = elo.get_momentum(t1)
        momentum2 = elo.get_momentum(t2)
        vs_strong1 = elo.get_winrate_vs_strong(t1)
        vs_strong2 = elo.get_winrate_vs_strong(t2)

        # Rank ratio (static)
        rank_ratio = rank2 / max(rank1, 1)
        log_rank_ratio = np.log(max(rank_ratio, 0.01))
        # Dynamic rank ratio (no leakage)
        dyn_rank_ratio = dyn_rank2 / max(dyn_rank1, 1)
        dyn_log_rank_ratio = np.log(max(dyn_rank_ratio, 0.01))
        # Elo-rank ratio (dynamic, no leakage)
        elo_rank_ratio = elo_rank2 / max(elo_rank1, 1)
        log_elo_rank_ratio = np.log(max(elo_rank_ratio, 0.01))

        # --- Map features ---
        map_pool_depth1 = mtracker.get_map_pool_depth(t1)
        map_pool_depth2 = mtracker.get_map_pool_depth(t2)
        map_wr_overlap = mtracker.get_overlap_winrate_diff(t1, t2)

        feat = {
            "elo_diff": elo1 - elo2,
            "rank_diff": rank2 - rank1,
            "rank_ratio": rank_ratio,
            "log_rank_ratio": log_rank_ratio,
            "rank1": rank1,
            "rank2": rank2,
            "dyn_rank1": dyn_rank1,
            "dyn_rank2": dyn_rank2,
            "dyn_rank_diff": dyn_rank2 - dyn_rank1,
            "dyn_log_rank_ratio": dyn_log_rank_ratio,
            "elo_rank_diff": elo_rank2 - elo_rank1,
            "log_elo_rank_ratio": log_elo_rank_ratio,
            "form_diff": form1 - form2,
            "streak_diff": streak1 - streak2,
            "h2h_winrate": h2h_rate,
            "h2h_matches": h2h_matches,
            "elo_expected": elo.expected_score(elo1, elo2),
            "momentum_diff": momentum1 - momentum2,
            "vs_strong_diff": vs_strong1 - vs_strong2,
            "n_matches1": elo.match_counts.get(t1, 0),
            "n_matches2": elo.match_counts.get(t2, 0),
            "map_pool_depth_diff": map_pool_depth1 - map_pool_depth2,
            "map_wr_overlap": map_wr_overlap,
        }

        # --- Rank volatility features ---
        rank_vol_max = max(rank_vol1, rank_vol2)
        dyn_rank_diff_val = feat["dyn_rank_diff"]
        rank_confidence = abs(dyn_rank_diff_val) / (1.0 + rank_vol_max)
        feat["rank_vol_diff"] = rank_vol1 - rank_vol2
        feat["rank_vol_max"] = rank_vol_max
        feat["rank_trajectory_diff"] = rank_traj1 - rank_traj2
        feat["rank_confidence"] = rank_confidence
        feat["rank_conf_x_rank_diff"] = rank_confidence * np.sign(dyn_rank_diff_val)

        # --- Player-level features ---
        players_data = match_players.get(match_id)
        if players_data is not None and len(players_data) >= 6:
            matches_with_players += 1
            for team_key, team_name in [("t1", t1), ("t2", t2)]:
                team_players = players_data[players_data["team"] == team_name]
                if team_players.empty:
                    # Fallback: try matching by position (first 5 = team1, last 5 = team2)
                    half = len(players_data) // 2
                    if team_key == "t1":
                        team_players = players_data.iloc[:half]
                    else:
                        team_players = players_data.iloc[half:]

                pfeat_list = []
                days_since = []
                roster_exp = []
                new_flags = []
                player_names = []

                for _, p in team_players.iterrows():
                    pname = p["player"]
                    player_names.append(pname)
                    pf = ptracker.get_player_features(pname)
                    pfeat_list.append(pf)
                    if date:
                        days_since.append(ptracker.get_days_since_last(pname, date))
                    rexp = ptracker.get_roster_experience(pname, team_name)
                    roster_exp.append(rexp)
                    new_flags.append(1 if rexp < 5 else 0)

                if pfeat_list:
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
                    feat[f"{team_key}_new_player"] = max(new_flags)

                    if days_since:
                        feat[f"{team_key}_avg_rest"] = np.mean(days_since)
                        feat[f"{team_key}_min_rest"] = min(days_since)
                        feat[f"{team_key}_max_rest"] = max(days_since)

                    # Team chemistry: avg days each pair has played together
                    raw_chem = chem.get_team_chemistry(player_names, date)
                    feat[f"{team_key}_chemistry"] = np.log1p(raw_chem)

            # Difference features
            for stat in ["avg_rating", "avg_adr", "avg_kast", "avg_kd_diff",
                         "star_rating", "weakest_rating", "star_gap", "consistency"]:
                k1, k2 = f"t1_{stat}", f"t2_{stat}"
                if k1 in feat and k2 in feat:
                    feat[f"diff_{stat}"] = feat[k1] - feat[k2]

            # Rest difference
            if "t1_avg_rest" in feat and "t2_avg_rest" in feat:
                feat["diff_avg_rest"] = feat["t1_avg_rest"] - feat["t2_avg_rest"]

            # Roster difference
            if "t1_roster_exp" in feat and "t2_roster_exp" in feat:
                feat["diff_roster_exp"] = feat["t1_roster_exp"] - feat["t2_roster_exp"]

            # Chemistry difference
            if "t1_chemistry" in feat and "t2_chemistry" in feat:
                feat["diff_chemistry"] = feat["t1_chemistry"] - feat["t2_chemistry"]

        else:
            # No player data for this match — fill with defaults
            for team_key in ["t1", "t2"]:
                feat[f"{team_key}_avg_rating"] = 1.0
                feat[f"{team_key}_avg_adr"] = 70.0
                feat[f"{team_key}_avg_kast"] = 70.0
                feat[f"{team_key}_avg_kd_diff"] = 0.0
                feat[f"{team_key}_star_rating"] = 1.0
                feat[f"{team_key}_weakest_rating"] = 1.0
                feat[f"{team_key}_star_gap"] = 0.0
                feat[f"{team_key}_consistency"] = 0.1
                feat[f"{team_key}_roster_exp"] = 0.0
                feat[f"{team_key}_new_player"] = 1
                feat[f"{team_key}_avg_rest"] = 14.0
                feat[f"{team_key}_min_rest"] = 14.0
                feat[f"{team_key}_max_rest"] = 14.0
                feat[f"{team_key}_chemistry"] = 0.0
            for stat in ["avg_rating", "avg_adr", "avg_kast", "avg_kd_diff",
                         "star_rating", "weakest_rating", "star_gap", "consistency"]:
                feat[f"diff_{stat}"] = 0.0
            feat["diff_avg_rest"] = 0.0
            feat["diff_roster_exp"] = 0.0
            feat["diff_chemistry"] = 0.0

        # --- Match context features ---
        feat["is_home_t1"] = get_home_advantage(t1, event)
        feat["is_home_t2"] = get_home_advantage(t2, event)
        feat["home_diff"] = feat["is_home_t1"] - feat["is_home_t2"]

        # BO format
        if mi is not None:
            bo = str(mi.get("bo_format", "BO3"))
            feat["is_bo1"] = 1 if bo == "BO1" else 0
            feat["is_bo3"] = 1 if bo == "BO3" else 0
            feat["is_bo5"] = 1 if bo == "BO5" else 0
        else:
            s1, s2 = row.get("score1", 0), row.get("score2", 0)
            total = int(s1) + int(s2) if pd.notna(s1) and pd.notna(s2) else 0
            feat["is_bo1"] = 1 if total == 1 else 0
            feat["is_bo3"] = 1 if 2 <= total <= 3 else 0
            feat["is_bo5"] = 1 if total >= 4 else 0

        # --- Interaction features ---
        bo_weight = 1.0 if feat.get("is_bo1") else (1.3 if feat.get("is_bo3") else 1.5)
        feat["rank_diff_x_bo"] = feat["rank_diff"] * bo_weight
        feat["dyn_rank_diff_x_bo"] = feat["dyn_rank_diff"] * bo_weight
        feat["elo_rank_diff_x_bo"] = feat["elo_rank_diff"] * bo_weight
        feat["elo_diff_x_form"] = feat["elo_diff"] * feat["form_diff"] if feat["form_diff"] != 0 else 0
        feat["rank_x_h2h"] = feat["rank_diff"] * (feat["h2h_winrate"] - 0.5)
        feat["momentum_x_form"] = feat["momentum_diff"] * feat["form_diff"] if feat["form_diff"] != 0 else 0

        # Map upset potential: for BO1, if the specific map favors the lower-ranked team
        if feat.get("is_bo1") and has_maps:
            maps_data = match_maps.get(match_id)
            if maps_data is not None and len(maps_data) >= 1:
                played_map = maps_data.iloc[0]["map_name"]
                wr1 = mtracker.get_team_map_winrate(t1, played_map)
                wr2 = mtracker.get_team_map_winrate(t2, played_map)
                # Positive = map favors t1 relative to rank expectation
                feat["map_upset_potential"] = (wr1 - wr2) - (0.5 if dyn_rank1 < dyn_rank2 else -0.5 if dyn_rank1 > dyn_rank2 else 0)
            else:
                feat["map_upset_potential"] = 0.0
        else:
            feat["map_upset_potential"] = 0.0

        # --- Upset detector (uses only historical data) ---
        upset_prob = upset_detector.predict_upset_prob(feat)
        feat["upset_prob"] = upset_prob
        feat["upset_prob_x_rank_diff"] = upset_prob * dyn_rank_diff_val

        # --- Pistol round features ---
        pistol_wr1 = pistol_tracker.get_pistol_winrate(t1)
        pistol_wr2 = pistol_tracker.get_pistol_winrate(t2)
        feat["pistol_wr_diff"] = pistol_wr1 - pistol_wr2

        features.append(feat)
        label = 1 if winner == t1 else 0
        labels.append(label)
        has_player_mask.append(players_data is not None and len(players_data) >= 6)

        # Determine if this was an upset (lower-ranked team won)
        # dyn_rank_diff = dyn_rank2 - dyn_rank1; positive means t1 ranked higher
        was_upset = (dyn_rank_diff_val > 0 and label == 0) or (dyn_rank_diff_val < 0 and label == 1)
        upset_detector.update(feat, was_upset)

        # --- Update trackers AFTER recording features ---
        loser = t2 if winner == t1 else t1
        maps_played = mi["maps_played"] if mi is not None else 1
        try:
            maps_played = int(maps_played)
        except (ValueError, TypeError):
            maps_played = 1
        elo.update(winner, loser, maps_played=maps_played)
        form.update(winner, loser)
        h2h.update(winner, loser)

        # Update player tracker + chemistry tracker
        if players_data is not None:
            for team_name in [t1, t2]:
                tp = players_data[players_data["team"] == team_name]
                if tp.empty:
                    half = len(players_data) // 2
                    tp = players_data.iloc[:half] if team_name == t1 else players_data.iloc[half:]
                names = tp["player"].tolist()
                chem.update(names, date)
            for _, p in players_data.iterrows():
                ptracker.update(
                    player=p["player"], team=p["team"],
                    rating=p.get("rating", 1.0), adr=p.get("adr", 70.0),
                    kast=p.get("kast", 70.0), kills=p.get("kills", 0),
                    deaths=p.get("deaths", 0), date=date
                )

        # Update map tracker
        maps_data = match_maps.get(match_id)
        if maps_data is not None:
            for _, mp in maps_data.iterrows():
                map_name = mp["map_name"]
                map_winner_name = mp["map_winner"]
                mtracker.update(t1, map_name, map_winner_name == t1)
                mtracker.update(t2, map_name, map_winner_name == t2)

        # Update pistol tracker
        pistol_data = match_pistol.get(match_id)
        if pistol_data is not None:
            for pr in pistol_data:
                for pw_key in ["pistol1_winner", "pistol2_winner"]:
                    pw = str(pr.get(pw_key, "")).strip()
                    if pw:
                        pt1 = str(pr.get("team1", "")).strip()
                        pt2 = str(pr.get("team2", "")).strip()
                        if pt1:
                            pistol_tracker.update(pt1, pw == pt1)
                        if pt2:
                            pistol_tracker.update(pt2, pw == pt2)

    print(f"  Matches with player data: {matches_with_players}/{len(df)}")
    return pd.DataFrame(features), np.array(labels), np.array(has_player_mask)


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def evaluate_models(X_train, y_train, X_test, y_test, feature_names):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            random_state=42, verbose=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        if name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        results[name] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "model": model}
        print(f"  {name}: acc={acc:.4f}, log_loss={ll:.4f}")

    # Feature importance
    best_tree_name = max(
        [n for n in results if n != "LogisticRegression"],
        key=lambda n: results[n]["accuracy"],
    )
    best_tree = results[best_tree_name]["model"]
    importances = best_tree.feature_importances_
    fi = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    print(f"\n  Feature importance ({best_tree_name}):")
    for fname, imp in fi[:15]:
        print(f"    {fname}: {imp:.4f}")

    return results


def run_xgb_gridsearch(X_train, y_train):
    """Quick grid search for XGBoost hyperparameters."""
    print("\n--- XGBoost Grid Search ---")
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.03, 0.08],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    xgb = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
    tscv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(xgb, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV accuracy: {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(iteration=5):
    print(f"=== Iteration {iteration}: Ensemble + Feature Selection CS Match Predictor ===\n")

    # Load match data — use ALL matches for more training data
    matches_path = os.path.join(DATA_DIR, "all_results_with_urls.csv")
    if not os.path.exists(matches_path):
        matches_path = os.path.join(DATA_DIR, "top100_matches_with_urls.csv")
    if not os.path.exists(matches_path):
        matches_path = os.path.join(DATA_DIR, "top100_matches.csv")
        print("WARNING: Using matches without URLs (no match_id linkage)")

    df = pd.read_csv(matches_path)
    with open(os.path.join(DATA_DIR, "top_teams.json")) as f:
        top_teams = json.load(f)

    if "match_url" in df.columns:
        df["match_id"] = df["match_url"].apply(
            lambda u: str(u).split("/matches/")[1].split("/")[0]
            if pd.notna(u) and "/matches/" in str(u) else ""
        )
    else:
        df["match_id"] = ""

    print(f"Loaded {len(df)} matches")
    df = df.iloc[::-1].reset_index(drop=True)  # oldest first

    player_stats_df = load_player_stats()
    match_details_df = load_match_details()
    rankings_df = load_rankings_history()
    map_results_df = load_map_results()
    pistol_rounds_df = load_pistol_rounds()

    if player_stats_df is not None:
        print(f"Player stats: {len(player_stats_df)} rows, "
              f"{player_stats_df['player'].nunique()} unique players")
    if match_details_df is not None:
        print(f"Match details: {len(match_details_df)} rows")
    if rankings_df is not None:
        print(f"Rankings history: {len(rankings_df)} rows, "
              f"{rankings_df['date'].nunique()} weeks")
    if map_results_df is not None:
        print(f"Map results: {len(map_results_df)} rows, "
              f"{map_results_df['map_name'].nunique()} unique maps")
    if pistol_rounds_df is not None:
        print(f"Pistol rounds: {len(pistol_rounds_df)} rows, "
              f"{pistol_rounds_df['match_id'].nunique()} matches")

    # Build features
    print("\nBuilding features...")
    X, y, has_players = build_features(df, top_teams, player_stats_df, match_details_df,
                                        rankings_df, map_results_df, pistol_rounds_df)
    feature_names = list(X.columns)
    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {len(X)}")
    print(f"  Class balance: {y.mean():.3f} (team1 win rate)")

    # Summary stats for new features
    new_feats = ["rank_vol_diff", "rank_vol_max", "rank_trajectory_diff",
                 "rank_confidence", "rank_conf_x_rank_diff", "upset_prob", "upset_prob_x_rank_diff",
                 "pistol_wr_diff"]
    print("\n  New feature summary:")
    for f in new_feats:
        if f in X.columns:
            col = X[f]
            print(f"    {f}: mean={col.mean():.4f}, std={col.std():.4f}, min={col.min():.4f}, max={col.max():.4f}")
    # Verify upset_prob varies after warmup
    if "upset_prob" in X.columns:
        n_varied = (X["upset_prob"] != 0.5).sum()
        print(f"    upset_prob varied (!=0.5): {n_varied}/{len(X)} samples")

    # Filter to only matches with player data if requested
    players_only = os.environ.get("PLAYERS_ONLY", "0") == "1"
    if players_only:
        mask = has_players
        X = X[mask].reset_index(drop=True)
        y = y[mask]
        print(f"  Filtered to player-data matches: {len(X)}")
        print(f"  Class balance (filtered): {y.mean():.3f}")

    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Sample weights: weight recent matches more heavily
    DECAY = 0.985
    sample_weights = np.array([DECAY ** (len(X_train) - i) for i in range(len(X_train))])
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    print(f"  Sample weighting: decay={DECAY}")

    results = {}

    # =============================================
    # Strategy 1: Curated minimal feature sets
    # =============================================

    # Minimal: only the strongest signals (dynamic rank replaces static)
    minimal_cols = ["dyn_log_rank_ratio", "dyn_rank_diff_x_bo", "elo_expected", "form_diff",
                    "h2h_winrate", "diff_avg_rating", "log_elo_rank_ratio",
                    "momentum_diff", "vs_strong_diff",
                    "map_pool_depth_diff", "map_wr_overlap", "diff_chemistry",
                    "rank_confidence", "rank_trajectory_diff", "upset_prob",
                    "pistol_wr_diff"]
    minimal_cols = [c for c in minimal_cols if c in feature_names]

    # Lean: differences + team-level (includes dynamic rank + map features + volatility + upset)
    lean_cols = [c for c in feature_names if c.startswith("diff_") or c in [
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
    ]]

    print(f"\n  Minimal features ({len(minimal_cols)}): {minimal_cols}")
    print(f"  Lean features ({len(lean_cols)})")

    # =============================================
    # Strategy 2: L1-regularized LogisticRegression (auto feature selection)
    # =============================================
    print("\n--- Strategy 1: L1 Logistic Regression (auto feature selection) ---")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    for C_val in [0.005, 0.008, 0.01, 0.015, 0.02, 0.05, 0.1, 0.5]:
        lr_l1 = LogisticRegression(penalty="l1", C=C_val, max_iter=2000, solver="saga")
        lr_l1.fit(X_train_sc, y_train, sample_weight=sample_weights)
        y_pred = lr_l1.predict(X_test_sc)
        y_prob = lr_l1.predict_proba(X_test_sc)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        n_nonzero = np.sum(lr_l1.coef_[0] != 0)
        name = f"LR_L1_C{C_val}"
        results[name] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "model": lr_l1}
        print(f"  {name}: acc={acc:.4f}, log_loss={ll:.4f}, nonzero_features={n_nonzero}/{len(feature_names)}")

        if n_nonzero <= 15 or C_val == 0.1:
            # Show which features survived
            coefs = list(zip(feature_names, lr_l1.coef_[0]))
            active = [(n, c) for n, c in coefs if c != 0]
            active.sort(key=lambda x: -abs(x[1]))
            print(f"    Active features: {[n for n, _ in active[:15]]}")

    # =============================================
    # Strategy 3: Minimal feature models
    # =============================================
    print("\n--- Strategy 2: Minimal feature set ---")
    X_min_train = X_train[minimal_cols]
    X_min_test = X_test[minimal_cols]
    min_scaler = StandardScaler()
    X_min_train_sc = min_scaler.fit_transform(X_min_train)
    X_min_test_sc = min_scaler.transform(X_min_test)

    for name, model in [
        ("LR_minimal", LogisticRegression(C=0.01, max_iter=2000)),
        ("LR_minimal_C05", LogisticRegression(C=0.05, max_iter=2000)),
        ("SVM_minimal", SVC(C=1.0, kernel="rbf", probability=True, random_state=42)),
        ("XGB_minimal_d6", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                                       subsample=0.8, eval_metric="logloss",
                                       random_state=42, verbosity=0)),
        ("XGB_minimal_d12", XGBClassifier(n_estimators=300, max_depth=12, learning_rate=0.03,
                                       subsample=0.8, eval_metric="logloss",
                                       random_state=42, verbosity=0)),
    ]:
        if "LR" in name or "SVM" in name:
            model.fit(X_min_train_sc, y_train, sample_weight=sample_weights)
            y_pred = model.predict(X_min_test_sc)
            y_prob = model.predict_proba(X_min_test_sc)[:, 1]
        else:
            model.fit(X_min_train, y_train, sample_weight=sample_weights)
            y_pred = model.predict(X_min_test)
            y_prob = model.predict_proba(X_min_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        results[name] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "model": model}
        print(f"  {name}: acc={acc:.4f}, log_loss={ll:.4f}")

    # =============================================
    # Strategy 4: Lean features with tuned models
    # =============================================
    print("\n--- Strategy 3: Lean features ---")
    X_lean_train = X_train[lean_cols]
    X_lean_test = X_test[lean_cols]
    lean_scaler = StandardScaler()
    X_lean_train_sc = lean_scaler.fit_transform(X_lean_train)
    X_lean_test_sc = lean_scaler.transform(X_lean_test)

    lean_models = {
        "LR_lean": LogisticRegression(C=0.01, max_iter=2000),
        "SVM_lean": SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
        "XGB_lean_d6": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.8,
                                   eval_metric="logloss", random_state=42, verbosity=0),
        "XGB_lean_d10": XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.03,
                                   subsample=0.8, colsample_bytree=0.8,
                                   eval_metric="logloss", random_state=42, verbosity=0),
        "XGB_lean_d15": XGBClassifier(n_estimators=300, max_depth=15, learning_rate=0.02,
                                   subsample=0.7, colsample_bytree=0.7,
                                   eval_metric="logloss", random_state=42, verbosity=0),
        "LGBM_lean_d6": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, verbose=-1),
        "LGBM_lean_d12": LGBMClassifier(n_estimators=300, max_depth=12, learning_rate=0.03,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, verbose=-1),
    }
    for name, model in lean_models.items():
        if "LR" in name or "SVM" in name:
            model.fit(X_lean_train_sc, y_train, sample_weight=sample_weights)
            y_pred = model.predict(X_lean_test_sc)
            y_prob = model.predict_proba(X_lean_test_sc)[:, 1]
        else:
            model.fit(X_lean_train, y_train, sample_weight=sample_weights)
            y_pred = model.predict(X_lean_test)
            y_prob = model.predict_proba(X_lean_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        results[name] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "model": model}
        print(f"  {name}: acc={acc:.4f}, log_loss={ll:.4f}")

    # =============================================
    # Strategy FSVM: Fuzzy SVM variants
    # =============================================
    print("\n--- Fuzzy SVM models ---")
    # Timestamps for time-based membership (match index = chronological order)
    train_timestamps = np.arange(len(X_train), dtype=float)

    fsvm_configs = [
        # (name, membership, C, lambda_decay, hybrid_alpha)
        ("FSVM_center_lean", "class_center", 1.0, None, None),
        ("FSVM_center_C10_lean", "class_center", 10.0, None, None),
        ("FSVM_time_lean", "time_decay", 1.0, 0.001, None),
        ("FSVM_time_C10_lean", "time_decay", 10.0, 0.001, None),
        ("FSVM_conf_lean", "confidence", 1.0, None, None),
        ("FSVM_conf_C10_lean", "confidence", 10.0, None, None),
        ("FSVM_hybrid_lean", "hybrid", 1.0, 0.001, 0.5),
        ("FSVM_hybrid_C10_lean", "hybrid", 10.0, 0.001, 0.5),
        ("FSVM_hybrid_C5_lean", "hybrid", 5.0, 0.002, 0.6),
        # Minimal feature set variants
        ("FSVM_hybrid_min", "hybrid", 10.0, 0.001, 0.5),
        ("FSVM_center_min", "class_center", 10.0, None, None),
    ]

    for name, membership, C, lam, alpha in fsvm_configs:
        is_min = name.endswith("_min")
        X_tr = X_min_train_sc if is_min else X_lean_train_sc
        X_te = X_min_test_sc if is_min else X_lean_test_sc

        fsvm = FuzzySVM(
            C=C, kernel='rbf', gamma='scale',
            membership=membership,
            lambda_decay=lam if lam else 0.005,
            sigma_floor=0.1,
            hybrid_alpha=alpha if alpha else 0.5,
        )
        fsvm.fit(X_tr, y_train, timestamps=train_timestamps)
        y_pred = fsvm.predict(X_te)
        y_prob = fsvm.predict_proba(X_te)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob)
        results[name] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "model": fsvm}

        # Show membership stats
        s = fsvm.sample_weight_
        print(f"  {name}: acc={acc:.4f}, log_loss={ll:.4f}  "
              f"[membership: mean={s.mean():.3f}, min={s.min():.3f}, max={s.max():.3f}]")

    gc.collect()

    # =============================================
    # Strategy 5: XGBoost grid search (wider search)
    # =============================================
    print("\n--- Strategy 4: XGBoost Grid Search (lean) ---")
    # Wider depth grid: 3×6×3×1×1 = 54 combos × 3 folds = 162 fits
    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 5, 7, 10, 15, 20],
        "learning_rate": [0.01, 0.03, 0.08],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    xgb_gs = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
    tscv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(xgb_gs, param_grid, cv=tscv, scoring="accuracy", n_jobs=1, verbose=0)
    gs.fit(X_lean_train, y_train)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV accuracy: {gs.best_score_:.4f}")

    best_xgb = gs.best_estimator_
    best_params = gs.best_params_
    best_cv_score = gs.best_score_
    del gs; gc.collect()  # free the 54 fitted models

    best_xgb.fit(X_lean_train, y_train)
    y_pred = best_xgb.predict(X_lean_test)
    y_prob = best_xgb.predict_proba(X_lean_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    results["XGB_tuned_lean"] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "model": best_xgb}
    print(f"  XGB_tuned_lean: acc={acc:.4f}, log_loss={ll:.4f}")

    # Free memory from grid search
    gc.collect()

    # =============================================
    # Strategy 6: Stacking ensemble
    # =============================================
    print("\n--- Strategy 5: Stacking Ensemble ---")
    base_estimators = [
        ("lr", LogisticRegression(C=0.1, max_iter=1000)),
        ("xgb", XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.03,
                               subsample=0.8, eval_metric="logloss",
                               random_state=42, verbosity=0)),
        ("lgbm", LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03,
                                  subsample=0.8, random_state=42, verbose=-1)),
    ]
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=3,  # regular k-fold (TimeSeriesSplit doesn't cover all samples)
        passthrough=False,
        n_jobs=1,
    )
    stack.fit(X_lean_train, y_train)
    y_pred = stack.predict(X_lean_test)
    y_prob = stack.predict_proba(X_lean_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    results["Stacking_lean"] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "model": stack}
    print(f"  Stacking_lean: acc={acc:.4f}, log_loss={ll:.4f}")

    gc.collect()

    # =============================================
    # Strategy 7: Voting ensemble (soft)
    # =============================================
    print("\n--- Strategy 6: Voting Ensemble ---")
    vote = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(C=0.1, max_iter=1000)),
            ("xgb", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                                   subsample=0.8, eval_metric="logloss",
                                   random_state=42, verbosity=0)),
            ("lgbm", LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                                     subsample=0.8, random_state=42, verbose=-1)),
        ],
        voting="soft",
    )
    vote.fit(X_lean_train, y_train)
    y_pred = vote.predict(X_lean_test)
    y_prob = vote.predict_proba(X_lean_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    results["Voting_lean"] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "model": vote}
    print(f"  Voting_lean: acc={acc:.4f}, log_loss={ll:.4f}")

    gc.collect()

    # =============================================
    # Strategy 8: Calibrated models (optimize log_loss)
    # =============================================
    print("\n--- Strategy 7: Calibrated XGBoost ---")
    cal_xgb = CalibratedClassifierCV(
        XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                       subsample=0.8, eval_metric="logloss",
                       random_state=42, verbosity=0),
        cv=TimeSeriesSplit(n_splits=3),
        method="isotonic",
    )
    cal_xgb.fit(X_lean_train, y_train)
    y_pred = cal_xgb.predict(X_lean_test)
    y_prob = cal_xgb.predict_proba(X_lean_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    results["Calibrated_XGB_lean"] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "model": cal_xgb}
    print(f"  Calibrated_XGB_lean: acc={acc:.4f}, log_loss={ll:.4f}")

    gc.collect()

    # =============================================
    # Strategy 9: Rank-augmented ensemble
    # Combine rank prediction with ML model prediction
    # =============================================
    print("\n--- Strategy 8: Rank-augmented ensemble ---")
    rank_col = "dyn_rank_diff" if "dyn_rank_diff" in lean_cols else "rank_diff"
    rank_pred_train = (X_lean_train[rank_col] > 0).astype(float).values
    rank_pred_test = (X_lean_test[rank_col] > 0).astype(float).values

    # Train XGBoost on lean features
    xgb_for_aug = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                                 subsample=0.8, eval_metric="logloss",
                                 random_state=42, verbosity=0)
    xgb_for_aug.fit(X_lean_train, y_train)
    ml_prob_train = xgb_for_aug.predict_proba(X_lean_train)[:, 1]
    ml_prob_test = xgb_for_aug.predict_proba(X_lean_test)[:, 1]

    # Try different blend weights
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blended = alpha * ml_prob_test + (1 - alpha) * rank_pred_test
        y_pred = (blended > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, np.clip(blended, 1e-7, 1 - 1e-7))
        name = f"RankBlend_a{alpha}"
        results[name] = {"accuracy": round(acc, 4), "log_loss": round(ll, 4)}
        print(f"  {name}: acc={acc:.4f}, log_loss={ll:.4f}")

    # =============================================
    # Cross-validation for best strategies
    # =============================================
    print("\n--- 5-Fold Time-Series CV (lean features) ---")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_models = {
        "LR_L1_cv": LogisticRegression(penalty="l1", C=0.1, max_iter=2000, solver="saga"),
        "XGB_cv": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=42, verbosity=0),
    }
    cv_results = {}
    for name, model in cv_models.items():
        scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_lean_train)):
            xt = X_lean_train.iloc[train_idx]
            xv = X_lean_train.iloc[val_idx]
            yt = y_train[train_idx]
            yv = y_train[val_idx]
            if "LR" in name:
                sc = StandardScaler()
                xt = sc.fit_transform(xt)
                xv = sc.transform(xv)
            model.fit(xt, yt)
            scores.append(accuracy_score(yv, model.predict(xv)))
        cv_results[name] = {"mean": np.mean(scores), "std": np.std(scores)}
        print(f"  {name}: CV mean={np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # =============================================
    # Walk-forward evaluation (honest, no look-ahead)
    # Top 3 models by test accuracy + rank baseline
    # =============================================
    wf_window = min(200, max(50, len(X) // 10))
    wf_start = max(500, int(len(X) * 0.6))
    print(f"\n--- Walk-Forward Evaluation ({wf_window}-match windows, start={wf_start}) ---")

    # Pick top 3 models by test accuracy + always include best FSVM
    sorted_models = sorted(results.items(), key=lambda x: -x[1]["accuracy"])
    top3 = sorted_models[:3]
    # Ensure at least one FSVM is included in walk-forward
    fsvm_in_top3 = any("FSVM" in n for n, _ in top3)
    if not fsvm_in_top3:
        best_fsvm = max(
            [(n, r) for n, r in results.items() if "FSVM" in n],
            key=lambda x: x[1]["accuracy"], default=None
        )
        if best_fsvm:
            top3.append(best_fsvm)
    print(f"  Walk-forward models: {[n for n, _ in top3]}")

    # Define walk-forward model constructors (fresh model each window)
    wf_models = {}
    for name, _ in top3:
        if "LR_L1" in name:
            c_val = float(name.split("_C")[1])
            wf_models[name] = {
                "make": lambda c=c_val: LogisticRegression(penalty="l1", C=c, max_iter=2000, solver="saga"),
                "cols": lean_cols, "needs_scaling": True,
            }
        elif "LR_minimal" in name:
            c_val = 0.05 if "C05" in name else 0.01
            wf_models[name] = {
                "make": lambda c=c_val: LogisticRegression(C=c, max_iter=2000),
                "cols": minimal_cols, "needs_scaling": True,
            }
        elif "LR_lean" in name:
            wf_models[name] = {
                "make": lambda: LogisticRegression(C=0.01, max_iter=2000),
                "cols": lean_cols, "needs_scaling": True,
            }
        elif "FSVM" in name:
            # Extract config from the FSVM model that was trained
            fsvm_model = results[name].get("model")
            if fsvm_model and isinstance(fsvm_model, FuzzySVM):
                _C = fsvm_model.C
                _mem = fsvm_model.membership
                _lam = fsvm_model.lambda_decay
                _alpha = fsvm_model.hybrid_alpha
                _floor = fsvm_model.sigma_floor
            else:
                _C, _mem, _lam, _alpha, _floor = 10.0, "hybrid", 0.001, 0.5, 0.1
            cols = minimal_cols if "_min" in name else lean_cols
            wf_models[name] = {
                "make": lambda c=_C, m=_mem, l=_lam, a=_alpha, f=_floor: FuzzySVM(
                    C=c, kernel='rbf', gamma='scale', membership=m,
                    lambda_decay=l, hybrid_alpha=a, sigma_floor=f),
                "cols": cols, "needs_scaling": True, "is_fsvm": True,
            }
        elif "SVM" in name:
            cols = minimal_cols if "minimal" in name else lean_cols
            wf_models[name] = {
                "make": lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=42),
                "cols": cols, "needs_scaling": True,
            }
        elif "Calibrated" in name:
            wf_models[name] = {
                "make": lambda: CalibratedClassifierCV(
                    XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                        subsample=0.8, eval_metric="logloss", random_state=42, verbosity=0),
                    cv=TimeSeriesSplit(n_splits=3), method="isotonic"),
                "cols": lean_cols, "needs_scaling": False,
            }
        elif "XGB_tuned" in name:
            bp = best_params
            wf_models[name] = {
                "make": lambda p=bp: XGBClassifier(**p, eval_metric="logloss", random_state=42, verbosity=0),
                "cols": lean_cols, "needs_scaling": False,
            }
        elif "XGB" in name:
            cols = minimal_cols if "minimal" in name else lean_cols
            # Extract depth from name like XGB_lean_d10, default 6
            depth = 6
            if "_d" in name:
                try:
                    depth = int(name.split("_d")[-1])
                except ValueError:
                    depth = 6
            n_est = 300 if depth > 6 else 200
            wf_models[name] = {
                "make": lambda d=depth, n=n_est: XGBClassifier(n_estimators=n, max_depth=d, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, verbosity=0),
                "cols": cols, "needs_scaling": False,
            }
        elif "LGBM" in name:
            depth = 6
            if "_d" in name:
                try:
                    depth = int(name.split("_d")[-1])
                except ValueError:
                    depth = 6
            n_est = 300 if depth > 6 else 200
            wf_models[name] = {
                "make": lambda d=depth, n=n_est: LGBMClassifier(n_estimators=n, max_depth=d, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
                "cols": lean_cols, "needs_scaling": False,
            }
        elif "Stacking" in name:
            wf_models[name] = {
                "make": lambda: StackingClassifier(
                    estimators=[
                        ("lr", LogisticRegression(C=0.1, max_iter=1000)),
                        ("xgb", XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.03,
                            subsample=0.8, eval_metric="logloss", random_state=42, verbosity=0)),
                        ("lgbm", LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.03,
                            subsample=0.8, random_state=42, verbose=-1)),
                    ],
                    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
                    cv=3, passthrough=False, n_jobs=1),
                "cols": lean_cols, "needs_scaling": False,
            }
        elif "Voting" in name:
            wf_models[name] = {
                "make": lambda: VotingClassifier(
                    estimators=[
                        ("lr", LogisticRegression(C=0.1, max_iter=1000)),
                        ("xgb", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                            subsample=0.8, eval_metric="logloss", random_state=42, verbosity=0)),
                        ("lgbm", LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.03,
                            subsample=0.8, random_state=42, verbose=-1)),
                    ], voting="soft"),
                "cols": lean_cols, "needs_scaling": False,
            }
        else:
            # Fallback: skip unknown models
            print(f"  Skipping walk-forward for {name} (no constructor)")
            continue

    # Run walk-forward for each model + rank baseline
    rank_col = "dyn_rank_diff" if "dyn_rank_diff" in X.columns else "rank_diff"
    wf_rank_preds, wf_true_labels = [], []

    wf_results = {name: [] for name in wf_models}
    wf_probs = {name: [] for name in wf_models}  # collect probabilities for ensembles
    wf_contribs = {name: [] for name in wf_models}  # SHAP contributions per sample
    model_names = list(wf_models.keys())

    # --- Overfit diagnostics: per-window tracking ---
    wf_window_accs = {name: [] for name in wf_models}   # per-window eval accuracy
    wf_train_accs = {name: [] for name in wf_models}     # per-window train accuracy
    wf_sv_ratios = {name: [] for name in wf_models}      # support vector ratio (SVM only)

    _shap_warned = set()
    def _get_contribs(model, X_eval, feature_names, name):
        """Get per-sample feature contributions."""
        try:
            if "FSVM" in name:
                # Fuzzy SVM wraps SVC — no linear coefficients for RBF kernel
                return None
            if "LR" in name or "SVM" in name:
                # LR: contribution = coef * feature_value (per sample)
                coefs = model.coef_[0]  # shape (n_features,)
                X_arr = np.array(X_eval)
                return X_arr * coefs  # (n_samples, n_features)
            if "LGBM" in name:
                c = model.predict(X_eval, pred_contrib=True)
                return c[:, :-1]
            if "Calibrated" in name:
                cc = model.calibrated_classifiers_[0]
                base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
                if base is None:
                    return None
                booster = base.get_booster()
                dmat = xgb.DMatrix(X_eval, feature_names=list(feature_names))
                c = booster.predict(dmat, pred_contribs=True)
                return c[:, :-1]
            # Plain XGB
            booster = model.get_booster()
            dmat = xgb.DMatrix(X_eval, feature_names=list(feature_names))
            c = booster.predict(dmat, pred_contribs=True)
            return c[:, :-1]
        except Exception as e:
            if name not in _shap_warned:
                print(f"    (SHAP failed for {name}: {e})")
                _shap_warned.add(name)
            return None

    for start in range(wf_start, len(X) - wf_window, wf_window):
        end = start + wf_window
        yt_wf = y[:start]
        ye_wf = y[start:end]

        w = np.array([DECAY ** (len(yt_wf) - i) for i in range(len(yt_wf))])
        w = w / w.sum() * len(w)

        wf_true_labels.extend(ye_wf)
        wf_rank_preds.extend((X[rank_col].iloc[start:end] > 0).astype(int).values)

        for name, cfg in wf_models.items():
            cols = cfg["cols"]
            Xt_wf = X[cols].iloc[:start]
            Xe_wf = X[cols].iloc[start:end]

            if cfg["needs_scaling"]:
                sc_wf = StandardScaler()
                Xt_fit = sc_wf.fit_transform(Xt_wf)
                Xe_fit = sc_wf.transform(Xe_wf)
            else:
                Xt_fit, Xe_fit = Xt_wf, Xe_wf

            model = cfg["make"]()
            if cfg.get("is_fsvm"):
                ts = np.arange(len(yt_wf), dtype=float)
                model.fit(Xt_fit, yt_wf, timestamps=ts)
            else:
                try:
                    model.fit(Xt_fit, yt_wf, sample_weight=w)
                except TypeError:
                    model.fit(Xt_fit, yt_wf)
            preds_window = model.predict(Xe_fit)
            probs_window = model.predict_proba(Xe_fit)[:, 1]
            wf_results[name].extend(preds_window)
            wf_probs[name].extend(probs_window)

            # --- Overfit diagnostics: per-window metrics ---
            # 1. Per-window eval accuracy
            wf_window_accs[name].append(accuracy_score(ye_wf, preds_window))
            # 2. Train accuracy (how well does it fit its own training data?)
            train_preds = model.predict(Xt_fit)
            wf_train_accs[name].append(accuracy_score(yt_wf, train_preds))
            # 3. Support vector ratio (SVM/FSVM only)
            svc = getattr(model, 'svc_', None) or (model if hasattr(model, 'support_') else None)
            if svc is not None and hasattr(svc, 'support_'):
                wf_sv_ratios[name].append(len(svc.support_) / len(yt_wf))

            contribs = _get_contribs(model, Xe_fit, cols, name)
            if contribs is not None:
                wf_contribs[name].append((contribs, cols))
            del model; gc.collect()

    # --- Individual model results ---
    wf_rank_acc = accuracy_score(wf_true_labels, wf_rank_preds)
    print(f"  Walk-forward Rank (dynamic): {wf_rank_acc:.4f}")
    wf_all = {}
    for name in model_names:
        acc = accuracy_score(wf_true_labels, wf_results[name])
        edge = acc - wf_rank_acc
        wf_all[name] = round(acc, 4)
        print(f"  Walk-forward {name}: {acc:.4f} (edge: {edge:+.4f})")

    # --- Overfit Diagnostics ---
    print(f"\n  --- Overfit / Model Complexity Diagnostics ---")
    overfit_diag = {}
    for name in model_names:
        diag = {}
        # 1. Train accuracy gap
        mean_train = np.mean(wf_train_accs[name])
        wf_acc = accuracy_score(wf_true_labels, wf_results[name])
        train_wf_gap = mean_train - wf_acc
        diag["mean_train_acc"] = round(mean_train, 4)
        diag["wf_acc"] = round(wf_acc, 4)
        diag["train_wf_gap"] = round(train_wf_gap, 4)

        # 2. Per-window variance
        window_std = np.std(wf_window_accs[name])
        window_min = np.min(wf_window_accs[name])
        window_max = np.max(wf_window_accs[name])
        diag["window_std"] = round(window_std, 4)
        diag["window_min"] = round(window_min, 4)
        diag["window_max"] = round(window_max, 4)
        diag["window_accs"] = [round(a, 4) for a in wf_window_accs[name]]

        # 3. Support vector ratio (SVM/FSVM only)
        if wf_sv_ratios[name]:
            mean_sv = np.mean(wf_sv_ratios[name])
            diag["sv_ratio_mean"] = round(mean_sv, 4)
            diag["sv_ratio_last"] = round(wf_sv_ratios[name][-1], 4)
        else:
            diag["sv_ratio_mean"] = None

        # 4. Bootstrap 95% CI
        preds_arr = np.array(wf_results[name])
        true_arr = np.array(wf_true_labels)
        n_boot = 10000
        rng = np.random.RandomState(42)
        boot_accs = []
        n = len(preds_arr)
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            boot_accs.append(np.mean(preds_arr[idx] == true_arr[idx]))
        ci_low = np.percentile(boot_accs, 2.5)
        ci_high = np.percentile(boot_accs, 97.5)
        diag["boot_ci_95"] = [round(ci_low, 4), round(ci_high, 4)]
        diag["boot_ci_width"] = round(ci_high - ci_low, 4)

        overfit_diag[name] = diag

        # Print summary
        sv_str = f", SV ratio={diag['sv_ratio_mean']:.2%}" if diag["sv_ratio_mean"] is not None else ""
        print(f"  {name}:")
        print(f"    Train acc: {mean_train:.4f} | WF acc: {wf_acc:.4f} | Gap: {train_wf_gap:+.4f}")
        print(f"    Window std: {window_std:.4f} (range: {window_min:.4f}–{window_max:.4f})")
        print(f"    Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}] (width={ci_high - ci_low:.4f}){sv_str}")

    # Also bootstrap the rank baseline for comparison
    rank_preds_arr = np.array(wf_rank_preds)
    true_arr = np.array(wf_true_labels)
    boot_rank = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(rank_preds_arr), len(rank_preds_arr))
        boot_rank.append(np.mean(rank_preds_arr[idx] == true_arr[idx]))
    rank_ci_low = np.percentile(boot_rank, 2.5)
    rank_ci_high = np.percentile(boot_rank, 97.5)
    overfit_diag["_rank_baseline"] = {
        "wf_acc": round(wf_rank_acc, 4),
        "boot_ci_95": [round(rank_ci_low, 4), round(rank_ci_high, 4)],
        "boot_ci_width": round(rank_ci_high - rank_ci_low, 4),
    }
    print(f"  Rank baseline:")
    print(f"    Bootstrap 95% CI: [{rank_ci_low:.4f}, {rank_ci_high:.4f}] (width={rank_ci_high - rank_ci_low:.4f})")

    # Check if model CIs overlap with rank CI
    print(f"\n  --- CI Overlap Check (model vs rank baseline) ---")
    for name in model_names:
        m_ci = overfit_diag[name]["boot_ci_95"]
        overlaps = m_ci[0] <= rank_ci_high and rank_ci_low <= m_ci[1]
        sep = m_ci[0] - rank_ci_high if m_ci[0] > rank_ci_high else rank_ci_low - m_ci[1] if rank_ci_low > m_ci[1] else 0
        status = "OVERLAPS (not significant)" if overlaps else f"SEPARATED by {sep:.4f} (significant)"
        print(f"  {name}: {status}")

    # --- Ensembles of top-3 walk-forward models ---
    wf_ens_preds = {}  # store ensemble pred arrays for bootstrap CI
    print(f"\n  Walk-forward ensembles (from top 3):")
    probs_matrix = np.array([wf_probs[n] for n in model_names])  # (3, n_samples)
    preds_matrix = np.array([wf_results[n] for n in model_names])  # (3, n_samples)
    true = np.array(wf_true_labels)

    # 1. Majority vote (hard vote of 3 models)
    majority = (preds_matrix.sum(axis=0) >= 2).astype(int)
    acc = accuracy_score(true, majority)
    wf_all["Ens_majority_vote"] = round(acc, 4)
    wf_ens_preds["Ens_majority_vote"] = majority
    print(f"  Ens_majority_vote: {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

    # 2. Soft vote (average probabilities, equal weight)
    avg_prob = probs_matrix.mean(axis=0)
    soft_vote = (avg_prob >= 0.5).astype(int)
    acc = accuracy_score(true, soft_vote)
    wf_all["Ens_soft_vote"] = round(acc, 4)
    wf_ens_preds["Ens_soft_vote"] = soft_vote
    print(f"  Ens_soft_vote: {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

    # 3. Weighted soft vote (weight by test accuracy)
    test_accs = np.array([results[n]["accuracy"] for n in model_names])
    weights = test_accs / test_accs.sum()
    weighted_prob = (probs_matrix * weights[:, None]).sum(axis=0)
    weighted_vote = (weighted_prob >= 0.5).astype(int)
    acc = accuracy_score(true, weighted_vote)
    wf_all["Ens_weighted_soft"] = round(acc, 4)
    wf_ens_preds["Ens_weighted_soft"] = weighted_vote
    print(f"  Ens_weighted_soft: {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

    # 4. Pairs: ensemble each pair of 2
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            pair_prob = (probs_matrix[i] + probs_matrix[j]) / 2.0
            pair_pred = (pair_prob >= 0.5).astype(int)
            acc = accuracy_score(true, pair_pred)
            pair_name = f"Ens_{model_names[i][:8]}+{model_names[j][:8]}"
            wf_all[pair_name] = round(acc, 4)
            wf_ens_preds[pair_name] = pair_pred
            print(f"  {pair_name}: {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

    # 5. Rank-blend: blend best model prob with rank signal
    best_wf_model = max(model_names, key=lambda n: wf_all[n])
    best_probs = np.array(wf_probs[best_wf_model])
    rank_signal = np.array(wf_rank_preds, dtype=float)
    for alpha in [0.1, 0.2, 0.3]:
        blended = (1 - alpha) * best_probs + alpha * rank_signal
        blended_pred = (blended >= 0.5).astype(int)
        acc = accuracy_score(true, blended_pred)
        blend_name = f"Ens_rank_blend_a{alpha}"
        wf_all[blend_name] = round(acc, 4)
        wf_ens_preds[blend_name] = blended_pred
        print(f"  {blend_name}: {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

    # --- Disagreement analysis: which features drive correct vs wrong predictions ---
    print(f"\n  --- Disagreement Feature Analysis ---")
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            nA, nB = model_names[i], model_names[j]
            pA = np.array(wf_results[nA])
            pB = np.array(wf_results[nB])
            disagree = pA != pB
            n_disagree = disagree.sum()
            if n_disagree == 0:
                continue

            # Check both models have SHAP contribs
            if not wf_contribs[nA] or not wf_contribs[nB]:
                print(f"\n  {nA} vs {nB}: {n_disagree} disagreements (no SHAP data)")
                continue

            # Concatenate contribs across windows, map feature indices to names
            def _concat_contribs(contribs_list):
                all_c = []
                for c_arr, cols in contribs_list:
                    all_c.append((c_arr, cols))
                rows = np.vstack([c for c, _ in all_c])
                feat_names = all_c[0][1]  # all windows use same cols
                return rows, feat_names

            cA, fA = _concat_contribs(wf_contribs[nA])
            cB, fB = _concat_contribs(wf_contribs[nB])

            a_right = disagree & (pA == true)
            b_right = disagree & (pB == true)

            print(f"\n  {nA} vs {nB}: {n_disagree} disagreements")

            for label, mask, c_correct, f_correct, c_wrong, f_wrong, n_correct, n_wrong in [
                (f"{nA} correct", a_right, cA, fA, cB, fB, nA, nB),
                (f"{nB} correct", b_right, cB, fB, cA, fA, nB, nA),
            ]:
                n_cases = mask.sum()
                if n_cases == 0:
                    print(f"    {label}: 0 cases")
                    continue

                # For each disagreement sample, find the feature with highest |SHAP|
                correct_top = defaultdict(int)
                wrong_top = defaultdict(int)
                for idx in np.where(mask)[0]:
                    top_c = np.argmax(np.abs(c_correct[idx]))
                    top_w = np.argmax(np.abs(c_wrong[idx]))
                    correct_top[f_correct[top_c]] += 1
                    wrong_top[f_wrong[top_w]] += 1

                top5_correct = sorted(correct_top.items(), key=lambda x: -x[1])[:5]
                top5_wrong = sorted(wrong_top.items(), key=lambda x: -x[1])[:5]

                print(f"    {label} ({n_cases} cases):")
                print(f"      {n_correct} leading features: {', '.join(f'{f}({c})' for f, c in top5_correct)}")
                print(f"      {n_wrong} leading features:  {', '.join(f'{f}({c})' for f, c in top5_wrong)}")

    # --- Confusion matrix & conditional analysis: FSVM vs best XGB ---
    print(f"\n  --- Confusion Matrix & Conditional Analysis ---")
    # Find FSVM and best non-FSVM model
    fsvm_names = [n for n in model_names if "FSVM" in n]
    xgb_names = [n for n in model_names if "XGB" in n and "FSVM" not in n]

    if fsvm_names and xgb_names:
        best_fsvm_name = max(fsvm_names, key=lambda n: wf_all.get(n, 0))
        best_xgb_name = max(xgb_names, key=lambda n: wf_all.get(n, 0))

        pF = np.array(wf_results[best_fsvm_name])
        pX = np.array(wf_results[best_xgb_name])
        probF = np.array(wf_probs[best_fsvm_name])
        probX = np.array(wf_probs[best_xgb_name])

        from sklearn.metrics import confusion_matrix as cm_func

        print(f"\n  {best_fsvm_name} confusion matrix:")
        cm_f = cm_func(true, pF)
        print(f"    TN={cm_f[0,0]} FP={cm_f[0,1]}")
        print(f"    FN={cm_f[1,0]} TP={cm_f[1,1]}")
        prec_f = cm_f[1,1] / (cm_f[1,1] + cm_f[0,1]) if (cm_f[1,1] + cm_f[0,1]) > 0 else 0
        rec_f = cm_f[1,1] / (cm_f[1,1] + cm_f[1,0]) if (cm_f[1,1] + cm_f[1,0]) > 0 else 0
        print(f"    Precision={prec_f:.4f}, Recall={rec_f:.4f}")

        print(f"\n  {best_xgb_name} confusion matrix:")
        cm_x = cm_func(true, pX)
        print(f"    TN={cm_x[0,0]} FP={cm_x[0,1]}")
        print(f"    FN={cm_x[1,0]} TP={cm_x[1,1]}")
        prec_x = cm_x[1,1] / (cm_x[1,1] + cm_x[0,1]) if (cm_x[1,1] + cm_x[0,1]) > 0 else 0
        rec_x = cm_x[1,1] / (cm_x[1,1] + cm_x[1,0]) if (cm_x[1,1] + cm_x[1,0]) > 0 else 0
        print(f"    Precision={prec_x:.4f}, Recall={rec_x:.4f}")

        # Conditional analysis: where does each model win?
        disagree = pF != pX
        f_right = disagree & (pF == true)
        x_right = disagree & (pX == true)
        both_right = (~disagree) & (pF == true)
        both_wrong = (~disagree) & (pF != true)

        print(f"\n  Agreement/Disagreement breakdown ({len(true)} total WF samples):")
        print(f"    Both correct: {both_right.sum()} ({both_right.sum()/len(true)*100:.1f}%)")
        print(f"    Both wrong:   {both_wrong.sum()} ({both_wrong.sum()/len(true)*100:.1f}%)")
        print(f"    FSVM correct, XGB wrong: {f_right.sum()} ({f_right.sum()/len(true)*100:.1f}%)")
        print(f"    XGB correct, FSVM wrong: {x_right.sum()} ({x_right.sum()/len(true)*100:.1f}%)")

        # Analyze by rank difference buckets
        wf_rank_col = "dyn_rank_diff" if "dyn_rank_diff" in X.columns else "rank_diff"
        wf_rank_vals = X[wf_rank_col].iloc[wf_start:wf_start + len(true)].values
        # Also get confidence (probability distance from 0.5)
        confF = np.abs(probF - 0.5)
        confX = np.abs(probX - 0.5)

        # Bucket by rank difference magnitude
        abs_rank = np.abs(wf_rank_vals)
        buckets = [
            ("Close match (|rank_diff|<10)", abs_rank < 10),
            ("Medium gap (10-30)", (abs_rank >= 10) & (abs_rank < 30)),
            ("Large gap (30-60)", (abs_rank >= 30) & (abs_rank < 60)),
            ("Dominant (|rank_diff|>=60)", abs_rank >= 60),
        ]

        print(f"\n  Accuracy by rank difference bucket:")
        print(f"    {'Bucket':<35} {'N':>5}  {'FSVM':>7}  {'XGB':>7}  {'Delta':>7}")
        for label, mask in buckets:
            n = mask.sum()
            if n == 0:
                continue
            acc_f = accuracy_score(true[mask], pF[mask])
            acc_x = accuracy_score(true[mask], pX[mask])
            delta = acc_f - acc_x
            marker = " <--" if abs(delta) > 0.02 else ""
            print(f"    {label:<35} {n:>5}  {acc_f:>6.3f}  {acc_x:>6.3f}  {delta:>+6.3f}{marker}")

        # Bucket by upset vs expected outcome
        is_upset_match = (wf_rank_vals > 0) & (true == 0) | (wf_rank_vals < 0) & (true == 1)
        upset_buckets = [
            ("Upsets (lower-ranked won)", is_upset_match),
            ("Expected (higher-ranked won)", ~is_upset_match),
        ]

        print(f"\n  Accuracy on upsets vs expected:")
        print(f"    {'Category':<35} {'N':>5}  {'FSVM':>7}  {'XGB':>7}  {'Delta':>7}")
        for label, mask in upset_buckets:
            n = mask.sum()
            if n == 0:
                continue
            acc_f = accuracy_score(true[mask], pF[mask])
            acc_x = accuracy_score(true[mask], pX[mask])
            delta = acc_f - acc_x
            marker = " <--" if abs(delta) > 0.02 else ""
            print(f"    {label:<35} {n:>5}  {acc_f:>6.3f}  {acc_x:>6.3f}  {delta:>+6.3f}{marker}")

        # Bucket by BO format
        is_bo1 = X["is_bo1"].iloc[wf_start:wf_start + len(true)].values.astype(bool)
        is_bo3 = X["is_bo3"].iloc[wf_start:wf_start + len(true)].values.astype(bool)
        bo_buckets = [
            ("BO1 matches", is_bo1),
            ("BO3 matches", is_bo3),
            ("BO5 matches", ~is_bo1 & ~is_bo3),
        ]

        print(f"\n  Accuracy by BO format:")
        print(f"    {'Format':<35} {'N':>5}  {'FSVM':>7}  {'XGB':>7}  {'Delta':>7}")
        for label, mask in bo_buckets:
            n = mask.sum()
            if n == 0:
                continue
            acc_f = accuracy_score(true[mask], pF[mask])
            acc_x = accuracy_score(true[mask], pX[mask])
            delta = acc_f - acc_x
            marker = " <--" if abs(delta) > 0.02 else ""
            print(f"    {label:<35} {n:>5}  {acc_f:>6.3f}  {acc_x:>6.3f}  {delta:>+6.3f}{marker}")

        # Bucket by model confidence
        conf_buckets = [
            ("Low conf (prob 0.4-0.6)", (confF < 0.1) | (confX < 0.1)),
            ("Medium conf (0.6-0.75)", ((confF >= 0.1) & (confF < 0.25)) | ((confX >= 0.1) & (confX < 0.25))),
            ("High conf (>0.75)", (confF >= 0.25) | (confX >= 0.25)),
        ]

        print(f"\n  Accuracy by prediction confidence:")
        print(f"    {'Confidence':<35} {'N':>5}  {'FSVM':>7}  {'XGB':>7}  {'Delta':>7}")
        for label, mask in conf_buckets:
            n = mask.sum()
            if n == 0:
                continue
            acc_f = accuracy_score(true[mask], pF[mask])
            acc_x = accuracy_score(true[mask], pX[mask])
            delta = acc_f - acc_x
            marker = " <--" if abs(delta) > 0.02 else ""
            print(f"    {label:<35} {n:>5}  {acc_f:>6.3f}  {acc_x:>6.3f}  {delta:>+6.3f}{marker}")

        # --- Conditional ensemble: use FSVM where it's stronger, XGB where it's stronger ---
        print(f"\n  --- Conditional Ensemble Strategies ---")

        # Strategy 1: Use FSVM when both models agree, use FSVM when they disagree
        # and FSVM is more confident
        cond1 = np.where(confF >= confX, pF, pX)
        acc = accuracy_score(true, cond1)
        wf_all["Ens_conf_routing"] = round(acc, 4)
        wf_ens_preds["Ens_conf_routing"] = cond1
        print(f"  Ens_conf_routing (use more confident): {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

        # Strategy 2: Use FSVM for close matches, XGB for clear gaps
        cond2 = np.where(abs_rank < 20, pF, pX)
        acc = accuracy_score(true, cond2)
        wf_all["Ens_rank_routing_20"] = round(acc, 4)
        wf_ens_preds["Ens_rank_routing_20"] = cond2
        print(f"  Ens_rank_routing (FSVM if |rank_diff|<20): {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

        cond2b = np.where(abs_rank < 30, pF, pX)
        acc = accuracy_score(true, cond2b)
        wf_all["Ens_rank_routing_30"] = round(acc, 4)
        wf_ens_preds["Ens_rank_routing_30"] = cond2b
        print(f"  Ens_rank_routing (FSVM if |rank_diff|<30): {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

        cond2c = np.where(abs_rank < 50, pF, pX)
        acc = accuracy_score(true, cond2c)
        wf_all["Ens_rank_routing_50"] = round(acc, 4)
        wf_ens_preds["Ens_rank_routing_50"] = cond2c
        print(f"  Ens_rank_routing (FSVM if |rank_diff|<50): {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

        # Strategy 3: Weighted blend using rank-dependent alpha
        # Close matches: more FSVM weight; far matches: more XGB weight
        alpha_rank = np.clip(abs_rank / 100.0, 0.0, 0.8)  # 0 to 0.8
        cond3_prob = (1 - alpha_rank) * probF + alpha_rank * probX
        cond3 = (cond3_prob >= 0.5).astype(int)
        acc = accuracy_score(true, cond3)
        wf_all["Ens_rank_adaptive_blend"] = round(acc, 4)
        wf_ens_preds["Ens_rank_adaptive_blend"] = cond3
        print(f"  Ens_rank_adaptive_blend (alpha=|rank|/100): {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

        # Strategy 4: Use FSVM for upsets (where rank predictor would be wrong)
        rank_pred_wf = (wf_rank_vals > 0).astype(int)
        cond4 = np.where(rank_pred_wf != pX, pF, pX)  # When XGB agrees with rank, trust it; when it disagrees, check FSVM
        acc = accuracy_score(true, cond4)
        wf_all["Ens_upset_routing"] = round(acc, 4)
        wf_ens_preds["Ens_upset_routing"] = cond4
        print(f"  Ens_upset_routing (FSVM when XGB!=rank): {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

        # Strategy 5: Blend probabilities with various fixed weights
        for w_f in [0.3, 0.4, 0.5, 0.6, 0.7]:
            blend = w_f * probF + (1 - w_f) * probX
            blend_pred = (blend >= 0.5).astype(int)
            acc = accuracy_score(true, blend_pred)
            bname = f"Ens_FSVM_XGB_w{w_f}"
            wf_all[bname] = round(acc, 4)
            wf_ens_preds[bname] = blend_pred
            print(f"  {bname}: {acc:.4f} (edge: {acc - wf_rank_acc:+.4f})")

    # --- Bootstrap 95% CI for top ensembles ---
    print(f"\n  --- Ensemble Bootstrap 95% CI (top 5) ---")
    ens_sorted = sorted(wf_ens_preds.keys(), key=lambda k: -wf_all.get(k, 0))
    rng_ens = np.random.RandomState(42)
    for ens_name in ens_sorted[:5]:
        ep = np.array(wf_ens_preds[ens_name])
        boot_ens = []
        n = len(ep)
        for _ in range(10000):
            idx = rng_ens.randint(0, n, n)
            boot_ens.append(np.mean(ep[idx] == true[idx]))
        ci_lo = np.percentile(boot_ens, 2.5)
        ci_hi = np.percentile(boot_ens, 97.5)
        overfit_diag[ens_name] = {
            "wf_acc": wf_all[ens_name],
            "boot_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "boot_ci_width": round(ci_hi - ci_lo, 4),
        }
        overlaps_rank = ci_lo <= rank_ci_high and rank_ci_low <= ci_hi
        status = "OVERLAPS rank" if overlaps_rank else "SEPARATED from rank"
        print(f"  {ens_name}: [{ci_lo:.4f}, {ci_hi:.4f}] (width={ci_hi - ci_lo:.4f}) — {status}")

    # --- Upset Rate & Ensemble Accuracy by Team Tier ---
    print(f"\n  --- Upset Rate & Accuracy by Team Rank Tier ---")
    # Get rank columns for the WF evaluation range
    wf_end = wf_start + len(true)
    wf_rank1 = X["rank1"].iloc[wf_start:wf_end].values
    wf_rank2 = X["rank2"].iloc[wf_start:wf_end].values
    wf_avg_rank = (wf_rank1 + wf_rank2) / 2.0
    wf_best_rank = np.minimum(wf_rank1, wf_rank2)  # the higher-ranked team (lower number)
    wf_rank_diff_abs = np.abs(X["rank_diff"].iloc[wf_start:wf_end].values)
    wf_true = np.array(true)

    # Determine which team was favored (rank_diff > 0 means team1 is lower-ranked = team2 favored)
    wf_rank_diff_raw = X["rank_diff"].iloc[wf_start:wf_end].values
    is_upset = ((wf_rank_diff_raw > 0) & (wf_true == 1)) | ((wf_rank_diff_raw < 0) & (wf_true == 0))

    # Best ensemble predictions
    best_ens_name = max(wf_ens_preds.keys(), key=lambda k: wf_all.get(k, 0))
    best_ens_pred = np.array(wf_ens_preds[best_ens_name])

    # --- Tier buckets based on average rank ---
    avg_tier_edges = [0, 15, 30, 50, 75, 102]
    avg_tier_labels = ["Top 15", "15-30", "30-50", "50-75", "75+"]

    # --- Tier buckets based on best (higher-ranked) team ---
    best_tier_edges = [0, 10, 20, 35, 55, 102]
    best_tier_labels = ["Top 10", "10-20", "20-35", "35-55", "55+"]

    def compute_tier_stats(tier_vals, edges, labels):
        upset_rates, accuracies, counts = [], [], []
        for i in range(len(labels)):
            mask = (tier_vals >= edges[i]) & (tier_vals < edges[i + 1])
            n = mask.sum()
            counts.append(n)
            if n == 0:
                upset_rates.append(0)
                accuracies.append(0)
                continue
            upset_rates.append(is_upset[mask].mean())
            accuracies.append(accuracy_score(wf_true[mask], best_ens_pred[mask]))
        return upset_rates, accuracies, counts

    # Compute for average rank tiers
    avg_upsets, avg_accs, avg_counts = compute_tier_stats(wf_avg_rank, avg_tier_edges, avg_tier_labels)
    # Compute for best-team rank tiers
    best_upsets, best_accs, best_counts = compute_tier_stats(wf_best_rank, best_tier_edges, best_tier_labels)

    print(f"\n  By average rank of both teams (best ensemble: {best_ens_name}):")
    print(f"    {'Tier':<12} {'N':>5}  {'Upset%':>7}  {'Acc':>7}")
    for i, label in enumerate(avg_tier_labels):
        print(f"    {label:<12} {avg_counts[i]:>5}  {avg_upsets[i]:>6.1%}  {avg_accs[i]:>6.1%}")

    print(f"\n  By best (higher-ranked) team:")
    print(f"    {'Tier':<12} {'N':>5}  {'Upset%':>7}  {'Acc':>7}")
    for i, label in enumerate(best_tier_labels):
        print(f"    {label:<12} {best_counts[i]:>5}  {best_upsets[i]:>6.1%}  {best_accs[i]:>6.1%}")

    # --- Generate plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Upset Rate & Ensemble Accuracy by Team Rank Tier", fontsize=14, fontweight="bold")

    bar_width = 0.6
    colors_upset = "#e74c3c"
    colors_acc = "#2ecc71"

    # Plot 1: Upset rate by avg rank tier
    ax = axes[0, 0]
    x = np.arange(len(avg_tier_labels))
    bars = ax.bar(x, [u * 100 for u in avg_upsets], bar_width, color=colors_upset, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Average Rank Tier (both teams)")
    ax.set_ylabel("Upset Rate (%)")
    ax.set_title("Upset Rate by Avg Rank Tier")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(avg_tier_labels, avg_counts)])
    ax.set_ylim(0, 55)
    for bar, val in zip(bars, avg_upsets):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val:.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=is_upset.mean() * 100, color="gray", linestyle="--", alpha=0.5, label=f"Overall: {is_upset.mean():.1%}")
    ax.legend(fontsize=9)

    # Plot 2: Ensemble accuracy by avg rank tier
    ax = axes[0, 1]
    bars = ax.bar(x, [a * 100 for a in avg_accs], bar_width, color=colors_acc, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Average Rank Tier (both teams)")
    ax.set_ylabel("Ensemble Accuracy (%)")
    ax.set_title(f"Accuracy by Avg Rank Tier ({best_ens_name})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(avg_tier_labels, avg_counts)])
    ax.set_ylim(50, 80)
    for bar, val in zip(bars, avg_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    overall_acc = accuracy_score(wf_true, best_ens_pred)
    ax.axhline(y=overall_acc * 100, color="gray", linestyle="--", alpha=0.5, label=f"Overall: {overall_acc:.1%}")
    ax.legend(fontsize=9)

    # Plot 3: Upset rate by best-team rank tier
    ax = axes[1, 0]
    x2 = np.arange(len(best_tier_labels))
    bars = ax.bar(x2, [u * 100 for u in best_upsets], bar_width, color=colors_upset, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Best (Higher-Ranked) Team Tier")
    ax.set_ylabel("Upset Rate (%)")
    ax.set_title("Upset Rate by Higher-Ranked Team Tier")
    ax.set_xticks(x2)
    ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(best_tier_labels, best_counts)])
    ax.set_ylim(0, 55)
    for bar, val in zip(bars, best_upsets):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val:.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=is_upset.mean() * 100, color="gray", linestyle="--", alpha=0.5, label=f"Overall: {is_upset.mean():.1%}")
    ax.legend(fontsize=9)

    # Plot 4: Ensemble accuracy by best-team rank tier
    ax = axes[1, 1]
    bars = ax.bar(x2, [a * 100 for a in best_accs], bar_width, color=colors_acc, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Best (Higher-Ranked) Team Tier")
    ax.set_ylabel("Ensemble Accuracy (%)")
    ax.set_title(f"Accuracy by Higher-Ranked Team Tier ({best_ens_name})")
    ax.set_xticks(x2)
    ax.set_xticklabels([f"{l}\n(n={c})" for l, c in zip(best_tier_labels, best_counts)])
    ax.set_ylim(50, 80)
    for bar, val in zip(bars, best_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(y=overall_acc * 100, color="gray", linestyle="--", alpha=0.5, label=f"Overall: {overall_acc:.1%}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "upset_accuracy_by_tier.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    del wf_ens_preds
    del wf_results, wf_probs, wf_contribs, wf_true_labels, wf_rank_preds; gc.collect()

    # =============================================
    # Baselines
    # =============================================
    elo_baseline = accuracy_score(y_test, (X_test["elo_expected"] > 0.5).astype(int))
    rank_baseline = accuracy_score(y_test, (X_test["rank_diff"] > 0).astype(int))
    dyn_rank_baseline = accuracy_score(y_test, (X_test["dyn_rank_diff"] > 0).astype(int))
    weighted_elo_baseline = accuracy_score(y_test, (X_test["elo_diff"] > 0).astype(int))
    print(f"\n  Baselines (test split):")
    print(f"    Elo-only: {elo_baseline:.4f}")
    print(f"    Weighted Elo: {weighted_elo_baseline:.4f}")
    print(f"    Rank-only (static): {rank_baseline:.4f}")
    print(f"    Rank-only (dynamic): {dyn_rank_baseline:.4f}")

    gc.collect()

    # Best model
    scoreable = {k: v for k, v in results.items() if "accuracy" in v}
    best_name = max(scoreable, key=lambda n: scoreable[n]["accuracy"])
    best_acc = scoreable[best_name]["accuracy"]
    best_ll = scoreable[best_name].get("log_loss", 999)
    print(f"\n  Best model: {best_name} (acc={best_acc}, log_loss={best_ll})")

    # Save results
    experiment = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_features_total": len(feature_names),
        "n_lean_features": len(lean_cols),
        "n_minimal_features": len(minimal_cols),
        "lean_features": lean_cols,
        "minimal_features": minimal_cols,
        "cv_results": {k: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)} for k, v in cv_results.items()},
        "xgb_gridsearch_best_params": best_params,
        "xgb_gridsearch_best_cv": round(best_cv_score, 4),
        "test_results": {
            name: {"accuracy": r["accuracy"], "log_loss": r.get("log_loss", 999)}
            for name, r in scoreable.items()
        },
        "best_model": best_name,
        "best_accuracy": best_acc,
        "best_log_loss": best_ll,
        "walk_forward": {
            "rank_accuracy": round(wf_rank_acc, 4),
            "models": {name: {"accuracy": acc, "edge": round(acc - wf_rank_acc, 4)} for name, acc in wf_all.items()},
        },
        "baselines": {
            "elo_only": round(elo_baseline, 4),
            "weighted_elo": round(weighted_elo_baseline, 4),
            "rank_only_static": round(rank_baseline, 4),
            "rank_only_dynamic": round(dyn_rank_baseline, 4),
        },
        "overfit_diagnostics": overfit_diag,
        "player_data_coverage": f"{sum(1 for x in X['t1_avg_rating'] if x != 1.0)}/{len(X)}",
        "target": 0.70,
        "target_met": best_acc >= 0.70,
    }

    exp_path = os.path.join(RESULTS_DIR, "experiments.json")
    if os.path.exists(exp_path):
        with open(exp_path) as f:
            exp_data = json.load(f)
    else:
        exp_data = {"experiments": [], "best_experiment_id": None, "best_accuracy": 0}
    exp_data["experiments"].append(experiment)
    if best_acc > (exp_data.get("best_accuracy") or 0):
        exp_data["best_experiment_id"] = iteration
        exp_data["best_accuracy"] = best_acc
    with open(exp_path, "w") as f:
        json.dump(exp_data, f, indent=2)

    metrics = {"experiments": exp_data["experiments"], "latest": experiment}
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return experiment


if __name__ == "__main__":
    iteration = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    result = run_training(iteration)
    print(f"\n{'='*50}")
    if result["target_met"]:
        print(f"TARGET MET! Best accuracy: {result['best_accuracy']}")
    else:
        print(f"Target NOT met. Best accuracy: {result['best_accuracy']} (target: 0.70)")
