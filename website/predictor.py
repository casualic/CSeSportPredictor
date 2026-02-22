"""Load saved models and run ensemble prediction."""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_model_fsvm  # noqa: F401 — must be importable for pickle

from website.config import MODELS_DIR, LEAN_COLS, FSVM_WEIGHT, XGB_WEIGHT

_fsvm = None
_xgb = None
_scaler = None


class _ModelUnpickler(pickle.Unpickler):
    """Custom unpickler that redirects __main__.FuzzySVM to train_model_fsvm.FuzzySVM."""

    def find_class(self, module, name):
        if module == "__main__" and name == "FuzzySVM":
            return train_model_fsvm.FuzzySVM
        return super().find_class(module, name)


def _load_pkl(path):
    """Load a pickle file, resolving FuzzySVM from train_model_fsvm."""
    with open(path, "rb") as f:
        return _ModelUnpickler(f).load()


def _load_models():
    """Load saved sklearn/xgb models from disk (once).

    These are our own trained model pickle files — pickle is the standard
    serialization format for scikit-learn estimators.
    """
    global _fsvm, _xgb, _scaler
    if _fsvm is not None:
        return

    # Ensure train_model_fsvm is in sys.modules so pickle can resolve FuzzySVM
    if "train_model_fsvm" not in sys.modules:
        import train_model_fsvm  # noqa: F811

    _fsvm = _load_pkl(os.path.join(MODELS_DIR, "fsvm_winner.pkl"))
    _xgb = _load_pkl(os.path.join(MODELS_DIR, "xgb_winner.pkl"))
    _scaler = _load_pkl(os.path.join(MODELS_DIR, "scaler_winner.pkl"))

    print(f"Models loaded: FSVM={type(_fsvm).__name__}, XGB={type(_xgb).__name__}")


def predict(features_dict):
    """Run ensemble prediction on a single match feature dict.

    Args:
        features_dict: dict with keys matching LEAN_COLS

    Returns:
        dict with: predicted_winner (1=team1, 0=team2), t1_win_prob,
                   fsvm_prob, xgb_prob, models_agree, confidence
    """
    _load_models()

    # Build feature vector in correct order
    x = np.array([[features_dict.get(col, 0.0) for col in LEAN_COLS]])

    # FSVM needs scaled input
    x_scaled = _scaler.transform(x)
    fsvm_proba = _fsvm.predict_proba(x_scaled)[0]
    fsvm_prob_t1 = float(fsvm_proba[1])
    fsvm_pred = 1 if fsvm_prob_t1 > 0.5 else 0

    # XGB uses unscaled input
    xgb_proba = _xgb.predict_proba(x)[0]
    xgb_prob_t1 = float(xgb_proba[1])
    xgb_pred = 1 if xgb_prob_t1 > 0.5 else 0

    # Ensemble logic: agree -> use FSVM; disagree -> weighted blend
    models_agree = fsvm_pred == xgb_pred
    if models_agree:
        t1_win_prob = fsvm_prob_t1
        predicted = fsvm_pred
    else:
        t1_win_prob = FSVM_WEIGHT * fsvm_prob_t1 + XGB_WEIGHT * xgb_prob_t1
        predicted = 1 if t1_win_prob > 0.5 else 0

    confidence = abs(t1_win_prob - 0.5) * 2  # 0-1 scale

    return {
        "predicted_winner": predicted,
        "t1_win_prob": round(t1_win_prob, 4),
        "fsvm_prob": round(fsvm_prob_t1, 4),
        "xgb_prob": round(xgb_prob_t1, 4),
        "models_agree": 1 if models_agree else 0,
        "confidence": round(confidence, 4),
    }
