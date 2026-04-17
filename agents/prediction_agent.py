"""
prediction_agent.py — ML modelleri ile gişe tahmini yapan agent.

Görev:
  - GradientBoostingRegressor ile gişe tahmini
  - RandomForestClassifier ile başarı sınıfı (Hit/Orta/Riskli)
  - Yönetmen ROI + Oyuncu gücü + Tür başarısı ile kompozit skor
  - Dataset'ten feature engineering yapar
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Feature sütunları (eğitim sırası korunmalı) ──────────────────────────────
FEATURE_COLS = [
    "budget", "runtime", "popularity", "vote_count",
    "vote_average", "genre_count", "director_avg_roi", "cast_power",
]


def predict_movie(
    movie_data: dict,
    df: pd.DataFrame,
    reg_model,
    clf_model,
) -> dict:
    """
    Bir film için gişe tahmini ve başarı skoru hesaplar.

    Args:
        movie_data: search_agent'tan gelen film metadata'sı
        df:         Yüklenmiş TMDB 5000 DataFrame
        reg_model:  Eğitilmiş GradientBoostingRegressor
        clf_model:  Eğitilmiş RandomForestClassifier

    Returns:
        dict: Tahmin sonuçları (gişe, ROI, olasılıklar, kompozit skor)
    """
    budget     = movie_data.get("budget", 0)
    runtime    = movie_data.get("runtime", 100) or 100
    popularity = movie_data.get("popularity", 50)
    vote_count = movie_data.get("vote_count", 0)
    vote_avg   = movie_data.get("vote_average", 5.0)
    genres     = movie_data.get("genres", [])
    directors  = movie_data.get("directors", [])
    cast       = movie_data.get("cast", [])

    genre_count = len(genres) if genres else 1

    # ── Yönetmen Ortalama ROI ─────────────────────────────────────────────────
    dir_roi_med = df["director_avg_roi"].median()
    director    = directors[0] if directors else "Unknown"
    dir_data    = df[df["director"] == director]

    if len(dir_data) > 0:
        dir_roi = dir_data["roi"].mean()
        dir_roi = dir_roi if (not np.isnan(dir_roi) and dir_roi > 0) else dir_roi_med
    else:
        dir_roi = dir_roi_med

    # ── Oyuncu Gişe Gücü ─────────────────────────────────────────────────────
    cast_rev = df.explode("cast_list").groupby("cast_list")["revenue"].mean()
    cast_power_med = df["cast_power"].median()

    if cast:
        values = [cast_rev.get(c, 0) for c in cast]
        cast_power = np.mean(values) if any(v > 0 for v in values) else cast_power_med
    else:
        cast_power = cast_power_med

    # ── Tür Başarı Oranları ───────────────────────────────────────────────────
    genre_hits = {}
    for g in genres:
        g_data = df[df["genres_list"].apply(lambda gl: g in gl)]
        if len(g_data) > 0:
            genre_hits[g] = (g_data["success_class"] == 2).mean()
    avg_genre_hit = np.mean(list(genre_hits.values())) if genre_hits else 0.30

    # ── Bütçe tahmini (TMDB verisi yoksa tür bazlı) ───────────────────────────
    GENRE_BUDGET = {
        "Action": 180e6, "Adventure": 150e6, "Animation": 120e6,
        "Comedy": 60e6,  "Drama": 40e6,      "Horror": 25e6,
        "Science Fiction": 140e6, "Thriller": 55e6,
        "Romance": 35e6, "Fantasy": 120e6,   "Crime": 50e6,
    }
    budget_estimated = budget <= 0
    if budget_estimated:
        first_genre = genres[0] if genres else "Drama"
        budget = GENRE_BUDGET.get(first_genre, 70e6)

    # ── Model Tahmini ─────────────────────────────────────────────────────────
    X = [[budget, runtime, popularity, vote_count,
          vote_avg, genre_count, dir_roi, cast_power]]

    rev_pred   = float(reg_model.predict(X)[0])
    proba      = clf_model.predict_proba(X)[0]

    prob_fail = float(proba[0]) * 100
    prob_mid  = float(proba[1]) * 100
    prob_hit  = float(proba[2]) * 100 if len(proba) == 3 else (100 - prob_fail - prob_mid)

    roi_pred    = rev_pred / budget
    profit_pred = rev_pred - budget

    # ── Kompozit Başarı Skoru ─────────────────────────────────────────────────
    model_sc = proba[2] if len(proba) == 3 else 0.33
    dir_sc   = min(1.0, dir_roi / 5.0)
    cast_q80 = df["cast_power"].quantile(0.80)
    cast_sc  = min(1.0, cast_power / cast_q80) if cast_q80 > 0 else 0.5
    genre_sc = avg_genre_hit

    composite = (
        model_sc * 0.35 +
        dir_sc   * 0.25 +
        cast_sc  * 0.25 +
        genre_sc * 0.15
    )

    # ── Gerçek veriye göre override ───────────────────────────────────────────
    real_revenue = movie_data.get("revenue", 0)
    if real_revenue > 0 and budget > 0:
        real_roi = real_revenue / budget
        if real_roi >= 2.0:
            verdict = "HIT"
        elif real_roi >= 1.0:
            verdict = "ORTA"
        else:
            verdict = "RİSKLİ"
    elif roi_pred >= 2.0 or composite >= 0.45:
        verdict = "HIT"
    elif roi_pred >= 1.0 or composite >= 0.30:
        verdict = "ORTA"
    else:
        verdict = "RİSKLİ"

    return {
        # Tahmin
        "predicted_revenue": rev_pred,
        "predicted_revenue_m": rev_pred / 1e6,
        "roi_pred": roi_pred,
        "profit_pred": profit_pred,
        "verdict": verdict,
        # Olasılıklar
        "prob_hit": round(prob_hit, 1),
        "prob_mid": round(prob_mid, 1),
        "prob_fail": round(prob_fail, 1),
        # Kompozit
        "composite_score": round(composite * 100, 1),
        "model_score": round(model_sc * 100, 1),
        "director_score": round(dir_sc * 100, 1),
        "cast_score": round(cast_sc * 100, 1),
        "genre_score": round(genre_sc * 100, 1),
        # Meta
        "director_avg_roi": round(dir_roi, 2),
        "cast_power_m": round(cast_power / 1e6, 1),
        "avg_genre_hit_rate": round(avg_genre_hit * 100, 1),
        "budget_estimated": budget_estimated,
        "budget_used": budget,
        # Gerçek veri
        "real_revenue": real_revenue,
        "real_roi": real_revenue / budget if (real_revenue > 0 and budget > 0) else None,
    }


def format_prediction_result(pred: dict, movie_title: str) -> str:
    """Tahmin sonucunu okunabilir metin olarak formatlar."""
    real_info = ""
    if pred["real_revenue"] and pred["real_revenue"] > 0:
        real_info = f"\nGerçek Gişe: ${pred['real_revenue']/1e6:.0f}M | Gerçek ROI: x{pred['real_roi']:.2f}"

    budget_note = " (tür bazlı tahmin)" if pred["budget_estimated"] else ""

    return f"""
🎬 {movie_title} — Gişe Analizi
{'='*45}
Tahmini Gişe : ${pred['predicted_revenue_m']:.0f}M
Tahmini ROI  : x{pred['roi_pred']:.2f}
Tahmini Kâr  : ${pred['profit_pred']/1e6:.0f}M
Kullanılan Bütçe: ${pred['budget_used']/1e6:.0f}M{budget_note}
{real_info}

📊 Başarı Olasılıkları
  🔴 Başarısız: %{pred['prob_fail']:.0f}
  🟡 Orta     : %{pred['prob_mid']:.0f}
  🟢 Hit      : %{pred['prob_hit']:.0f}

🧠 Kompozit Başarı Skoru: %{pred['composite_score']:.0f}
  • ML Modeli  : %{pred['model_score']:.0f}
  • Yönetmen   : %{pred['director_score']:.0f} (Ort. ROI: x{pred['director_avg_roi']:.1f})
  • Oyuncu     : %{pred['cast_score']:.0f} (Ort. Gişe: ${pred['cast_power_m']:.0f}M)
  • Tür        : %{pred['genre_score']:.0f} (Hit Oranı: %{pred['avg_genre_hit_rate']:.0f})

Sonuç: {pred['verdict']}
""".strip()
