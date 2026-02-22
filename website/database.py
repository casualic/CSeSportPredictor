import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

_client = None


def _get_client():
    global _client
    if _client is None:
        if not _SUPABASE_URL or not _SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set. "
                "Create website/.env with these values."
            )
        _client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
    return _client


def init_db():
    """No-op: schema is managed in Supabase dashboard."""
    pass


def insert_prediction(data):
    """Insert a new prediction. data is a dict matching column names."""
    row = {
        "match_url": data.get("match_url"),
        "team1": data.get("team1"),
        "team2": data.get("team2"),
        "event": data.get("event"),
        "bo_format": data.get("bo_format"),
        "match_time": data.get("match_time"),
        "match_date": data.get("match_date"),
        "predicted_winner": data.get("predicted_winner"),
        "t1_win_prob": data.get("t1_win_prob"),
        "fsvm_prob": data.get("fsvm_prob"),
        "xgb_prob": data.get("xgb_prob"),
        "models_agree": bool(data.get("models_agree")),
        "confidence": data.get("confidence"),
        "odds_t1": data.get("odds_t1"),
        "odds_t2": data.get("odds_t2"),
        "implied_prob_t1": data.get("implied_prob_t1"),
        "implied_prob_t2": data.get("implied_prob_t2"),
        "edge": data.get("edge"),
        "t1_rank": data.get("t1_rank"),
        "t2_rank": data.get("t2_rank"),
    }
    _get_client().table("predictions").upsert(row, on_conflict="match_url").execute()
    return True


def resolve_prediction(pred_id, actual_winner):
    """Resolve a prediction with the actual result. Creates a bet if edge > 0."""
    client = _get_client()
    resp = client.table("predictions").select("*").eq("id", pred_id).single().execute()
    pred = resp.data
    if not pred:
        return False

    correct = actual_winner == pred["predicted_winner"]
    now = datetime.now(timezone.utc).isoformat()
    client.table("predictions").update({
        "actual_winner": actual_winner,
        "prediction_correct": correct,
        "resolved_at": now,
    }).eq("id", pred_id).execute()

    # Create bet record if there was positive edge
    edge = pred["edge"]
    if edge and edge > 0:
        bet_team = pred["predicted_winner"]
        bet_odds = pred["odds_t1"] if bet_team == pred["team1"] else pred["odds_t2"]
        if bet_odds and bet_odds > 0:
            won = actual_winner == bet_team
            pnl = (bet_odds - 1) * 1.0 if won else -1.0
            model_prob = pred["t1_win_prob"] if bet_team == pred["team1"] else 1 - pred["t1_win_prob"]
            client.table("bets").insert({
                "prediction_id": pred_id,
                "bet_team": bet_team,
                "bet_odds": bet_odds,
                "model_prob": model_prob,
                "edge": edge,
                "stake": 1.0,
                "won": won,
                "pnl": pnl,
            }).execute()
    return True


def get_upcoming():
    resp = (
        _get_client()
        .table("predictions")
        .select("*")
        .is_("actual_winner", "null")
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data


def get_resolved():
    resp = (
        _get_client()
        .table("predictions")
        .select("*")
        .not_.is_("actual_winner", "null")
        .order("resolved_at", desc=True)
        .execute()
    )
    return resp.data


def get_prediction(pred_id):
    resp = (
        _get_client()
        .table("predictions")
        .select("*")
        .eq("id", pred_id)
        .single()
        .execute()
    )
    return resp.data


def get_bets():
    resp = (
        _get_client()
        .table("bets")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data


def get_stats():
    """Get aggregate stats for the results page."""
    client = _get_client()
    resolved = (
        client.table("predictions")
        .select("prediction_correct, models_agree")
        .not_.is_("actual_winner", "null")
        .execute()
    ).data

    total = len(resolved)
    correct = sum(1 for r in resolved if r["prediction_correct"])
    agreed = sum(1 for r in resolved if r["models_agree"])
    agreed_correct = sum(1 for r in resolved if r["prediction_correct"] and r["models_agree"])

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "agreed": agreed,
        "agreed_correct": agreed_correct,
        "agreed_accuracy": agreed_correct / agreed if agreed > 0 else 0,
    }


def get_pnl_stats():
    """Get aggregate P&L stats."""
    resp = (
        _get_client()
        .table("bets")
        .select("*")
        .order("created_at")
        .execute()
    )
    rows = resp.data
    if not rows:
        return {"total_bets": 0, "wins": 0, "win_rate": 0, "total_pnl": 0, "roi": 0, "cumulative": []}

    total_bets = len(rows)
    wins = sum(1 for r in rows if r["won"])
    total_pnl = sum(r["pnl"] for r in rows)
    total_staked = sum(r["stake"] for r in rows)
    cumulative = []
    running = 0
    for r in rows:
        running += r["pnl"]
        cumulative.append(round(running, 2))

    return {
        "total_bets": total_bets,
        "wins": wins,
        "win_rate": wins / total_bets if total_bets > 0 else 0,
        "total_pnl": round(total_pnl, 2),
        "roi": round(total_pnl / total_staked * 100, 2) if total_staked > 0 else 0,
        "cumulative": cumulative,
    }
