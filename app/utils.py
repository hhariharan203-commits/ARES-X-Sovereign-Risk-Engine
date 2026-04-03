"""
utils.py — Shared helpers: formatting, scaling, safe calculations
"""

import numpy as np
import pandas as pd


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_pct(val, decimals=2):
    """Format a float as a percentage string."""
    try:
        return f"{float(val):.{decimals}f}%"
    except Exception:
        return "N/A"


def fmt_number(val, decimals=2):
    """Format a float with fixed decimals."""
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return "N/A"


def fmt_signed(val, decimals=2):
    """Format float with explicit sign."""
    try:
        v = float(val)
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.{decimals}f}%"
    except Exception:
        return "N/A"


def fmt_risk_label(score):
    """Map numeric risk score (0–100) to label."""
    if score is None:
        return "Unknown"
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def regime_label(gdp_growth, inflation):
    """Classify macro regime from GDP growth and inflation."""
    try:
        g = float(gdp_growth)
        i = float(inflation)
        if g > 2.5 and i < 4.0:
            return "Expansion"
        elif g > 2.5 and i >= 4.0:
            return "Overheating"
        elif g <= 0:
            return "Recession"
        elif 0 < g <= 2.5 and i >= 4.0:
            return "Stagflation"
        else:
            return "Slowdown"
    except Exception:
        return "Unknown"


# ── Safe Math ─────────────────────────────────────────────────────────────────

def safe_div(num, denom, default=0.0):
    try:
        if denom == 0:
            return default
        return num / denom
    except Exception:
        return default


def safe_mean(series):
    try:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        return float(vals.mean()) if len(vals) > 0 else 0.0
    except Exception:
        return 0.0


def clamp(val, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(val)))
    except Exception:
        return lo


def normalize_series(series):
    """Min-max normalize a pandas Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


# ── Color mapping ─────────────────────────────────────────────────────────────

REGIME_COLORS = {
    "Expansion":   "#00E5FF",
    "Overheating": "#FF9800",
    "Slowdown":    "#FFC107",
    "Stagflation": "#FF5252",
    "Recession":   "#B71C1C",
    "Unknown":     "#78909C",
}

RISK_COLORS = {
    "LOW":    "#00E676",
    "MEDIUM": "#FFC107",
    "HIGH":   "#FF5252",
}

DECISION_COLORS = {
    "BUY":       "#00E676",
    "HOLD":      "#FFC107",
    "DEFENSIVE": "#FF5252",
}


def regime_color(label):
    return REGIME_COLORS.get(label, "#78909C")


def risk_color(label):
    return RISK_COLORS.get(label, "#78909C")


def decision_color(label):
    return DECISION_COLORS.get(label, "#78909C")


# ── Trend helpers ─────────────────────────────────────────────────────────────

def trend_direction(series, window=3):
    """Return 'Rising', 'Falling', or 'Stable' based on recent slope."""
    try:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if len(vals) < window:
            return "Stable"
        recent = vals.iloc[-window:]
        slope = np.polyfit(range(len(recent)), recent.values, 1)[0]
        if slope > 0.05:
            return "Rising"
        elif slope < -0.05:
            return "Falling"
        return "Stable"
    except Exception:
        return "Stable"


def momentum_score(series, window=6):
    """Return a momentum score [-1, 1] from recent trend."""
    try:
        vals = pd.to_numeric(series, errors="coerce").dropna()
        if len(vals) < window:
            return 0.0
        recent = vals.iloc[-window:]
        slope = np.polyfit(range(len(recent)), recent.values, 1)[0]
        std = recent.std() or 1.0
        return float(np.clip(slope / std, -1, 1))
    except Exception:
        return 0.0