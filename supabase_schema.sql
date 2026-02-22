-- Run this in Supabase SQL Editor to set up the database schema

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    match_url TEXT UNIQUE,
    team1 TEXT NOT NULL, team2 TEXT NOT NULL,
    event TEXT, bo_format TEXT, match_time TEXT,
    predicted_winner TEXT,
    t1_win_prob DOUBLE PRECISION, fsvm_prob DOUBLE PRECISION, xgb_prob DOUBLE PRECISION,
    models_agree BOOLEAN DEFAULT FALSE, confidence DOUBLE PRECISION,
    odds_t1 DOUBLE PRECISION, odds_t2 DOUBLE PRECISION,
    implied_prob_t1 DOUBLE PRECISION, implied_prob_t2 DOUBLE PRECISION,
    edge DOUBLE PRECISION,
    actual_winner TEXT, prediction_correct BOOLEAN,
    resolved_at TIMESTAMPTZ, created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE bets (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id),
    bet_team TEXT, bet_odds DOUBLE PRECISION, model_prob DOUBLE PRECISION,
    edge DOUBLE PRECISION, stake DOUBLE PRECISION DEFAULT 1.0,
    won BOOLEAN, pnl DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE scrape_log (
    id SERIAL PRIMARY KEY,
    scrape_type TEXT, timestamp TIMESTAMPTZ DEFAULT now(),
    matches_found INTEGER, status TEXT
);

CREATE INDEX idx_predictions_actual_winner ON predictions(actual_winner);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);

-- Row Level Security (read-only for frontend anon key)
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE bets ENABLE ROW LEVEL SECURITY;
ALTER TABLE scrape_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read" ON predictions FOR SELECT TO anon USING (true);
CREATE POLICY "Public read" ON bets FOR SELECT TO anon USING (true);
CREATE POLICY "Public read" ON scrape_log FOR SELECT TO anon USING (true);
