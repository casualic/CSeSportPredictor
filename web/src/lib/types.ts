export interface Prediction {
  id: number;
  match_url: string | null;
  team1: string;
  team2: string;
  event: string | null;
  bo_format: string | null;
  match_time: string | null;
  match_date: string | null;
  predicted_winner: string | null;
  t1_win_prob: number;
  fsvm_prob: number;
  xgb_prob: number;
  models_agree: boolean;
  confidence: number;
  odds_t1: number | null;
  odds_t2: number | null;
  implied_prob_t1: number | null;
  implied_prob_t2: number | null;
  edge: number | null;
  t1_rank: number | null;
  t2_rank: number | null;
  actual_winner: string | null;
  prediction_correct: boolean | null;
  resolved_at: string | null;
  created_at: string;
}

export interface Bet {
  id: number;
  prediction_id: number;
  bet_team: string;
  bet_odds: number;
  model_prob: number;
  edge: number;
  stake: number;
  won: boolean;
  pnl: number;
  created_at: string;
}
