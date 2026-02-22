"""Flask application for CS2 match predictions."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

from website.config import WEBSITE_DIR, TRACKER_STATE_PATH
from website import database as db
from website.scraper import parse_odds, implied_probability, compute_edge

app = Flask(__name__,
            template_folder=os.path.join(WEBSITE_DIR, "templates"),
            static_folder=os.path.join(WEBSITE_DIR, "static"))
app.secret_key = "cs2-predictor-dev-key"

# Lazy-loaded globals
_bundle = None
_predictor_loaded = False


def get_bundle():
    global _bundle
    if _bundle is None:
        if os.path.exists(TRACKER_STATE_PATH):
            from website.tracker_state import TrackerBundle
            _bundle = TrackerBundle.load()
        else:
            print("WARNING: No tracker state found. Run: python -m website.tracker_state")
    return _bundle


def make_prediction(team1, team2, event="", bo_format="BO3"):
    """Build features and run prediction for a match."""
    bundle = get_bundle()
    if bundle is None:
        return None

    from website.feature_builder import build_match_features
    from website.predictor import predict

    features = build_match_features(
        bundle, team1, team2,
        event=event, bo_format=bo_format,
        match_date=datetime.now(),
    )
    result = predict(features)

    # Map predicted_winner int to team name
    result["predicted_winner"] = team1 if result["predicted_winner"] == 1 else team2
    return result


@app.before_request
def ensure_db():
    db.init_db()


@app.route("/")
def dashboard():
    predictions = db.get_upcoming()
    return render_template("dashboard.html", predictions=predictions)


@app.route("/results")
def results():
    predictions = db.get_resolved()
    stats = db.get_stats()
    return render_template("results.html", predictions=predictions, stats=stats)


@app.route("/pnl")
def pnl():
    pnl_stats = db.get_pnl_stats()
    bets = db.get_bets()
    return render_template("pnl.html", pnl=pnl_stats, bets=bets)


@app.route("/match/<int:pred_id>")
def match_detail(pred_id):
    p = db.get_prediction(pred_id)
    if not p:
        flash("Prediction not found", "danger")
        return redirect(url_for("dashboard"))
    return render_template("match.html", p=p)


@app.route("/api/add-prediction", methods=["POST"])
def add_prediction():
    team1 = request.form.get("team1", "").strip()
    team2 = request.form.get("team2", "").strip()
    event = request.form.get("event", "").strip()
    bo_format = request.form.get("bo_format", "BO3")
    match_url = request.form.get("match_url", "").strip() or f"{team1}-vs-{team2}-{datetime.now().isoformat()}"
    odds_t1 = parse_odds(request.form.get("odds_t1"))
    odds_t2 = parse_odds(request.form.get("odds_t2"))

    if not team1 or not team2:
        flash("Both team names are required", "danger")
        return redirect(url_for("dashboard"))

    # Run prediction
    pred = make_prediction(team1, team2, event, bo_format)
    if pred is None:
        flash("Tracker state not initialized. Run: python -m website.tracker_state", "danger")
        return redirect(url_for("dashboard"))

    # Compute odds/edge
    implied_t1 = implied_probability(odds_t1)
    implied_t2 = implied_probability(odds_t2)
    model_prob_winner = pred["t1_win_prob"] if pred["predicted_winner"] == team1 else 1 - pred["t1_win_prob"]
    implied_winner = implied_t1 if pred["predicted_winner"] == team1 else implied_t2
    edge = compute_edge(model_prob_winner, implied_winner)

    data = {
        "match_url": match_url,
        "team1": team1,
        "team2": team2,
        "event": event,
        "bo_format": bo_format,
        "match_time": request.form.get("match_time", ""),
        "predicted_winner": pred["predicted_winner"],
        "t1_win_prob": pred["t1_win_prob"],
        "fsvm_prob": pred["fsvm_prob"],
        "xgb_prob": pred["xgb_prob"],
        "models_agree": pred["models_agree"],
        "confidence": pred["confidence"],
        "odds_t1": odds_t1,
        "odds_t2": odds_t2,
        "implied_prob_t1": implied_t1,
        "implied_prob_t2": implied_t2,
        "edge": edge,
    }

    db.insert_prediction(data)
    flash(f"Prediction added: {pred['predicted_winner']} to win ({pred['t1_win_prob']*100:.1f}% T1)", "success")
    return redirect(url_for("dashboard"))


@app.route("/api/resolve/<int:pred_id>", methods=["POST"])
def resolve(pred_id):
    winner = request.form.get("winner", "").strip()
    if not winner:
        flash("Winner is required", "danger")
        return redirect(url_for("match_detail", pred_id=pred_id))

    success = db.resolve_prediction(pred_id, winner)
    if success:
        # Update tracker state with result
        bundle = get_bundle()
        p = db.get_prediction(pred_id)
        if bundle and p:
            bundle.update_with_result({
                "team1": p["team1"],
                "team2": p["team2"],
                "winner": winner,
                "event": p["event"] or "",
                "bo_format": p["bo_format"] or "BO3",
            })
            bundle.save()
        flash(f"Resolved: {winner} won!", "success")
    else:
        flash("Failed to resolve prediction", "danger")
    return redirect(url_for("results"))


@app.route("/api/predictions", methods=["GET"])
def api_predictions():
    """JSON API for predictions."""
    upcoming = [dict(r) for r in db.get_upcoming()]
    return jsonify(upcoming)


if __name__ == "__main__":
    db.init_db()
    app.run(debug=True, port=5000)
