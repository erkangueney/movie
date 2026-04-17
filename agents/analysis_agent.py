"""
analysis_agent.py — Dataset üzerinde istatistiksel analiz yapan agent.

Görev:
  - Tür bazlı ortalama gişe / ROI / hit oranı
  - Yönetmen & oyuncu istatistikleri
  - Dönem analizi
  - Korelasyon özetleri
  - Kullanıcının doğal dil sorusuna sayısal cevap üretir
"""

import numpy as np
import pandas as pd
from typing import Optional


def analyze_dataset(query_type: str, df: pd.DataFrame, **kwargs) -> dict:
    """
    Dataset üzerinde belirtilen analiz türünü çalıştırır.

    Args:
        query_type: 'genre' | 'director' | 'actor' | 'decade' | 'summary' | 'top_n'
        df:         TMDB 5000 DataFrame
        **kwargs:   Ek parametreler (name, genre, n, metric vb.)

    Returns:
        dict: Analiz sonuçları
    """
    qt = query_type.lower()

    # ── Genel özet ───────────────────────────────────────────────────────────
    if qt == "summary":
        return _summary(df)

    # ── Tür analizi ───────────────────────────────────────────────────────────
    elif qt == "genre":
        genre = kwargs.get("genre")
        return _genre_stats(df, genre)

    # ── Yönetmen analizi ─────────────────────────────────────────────────────
    elif qt == "director":
        name = kwargs.get("name")
        return _director_stats(df, name)

    # ── Oyuncu analizi ───────────────────────────────────────────────────────
    elif qt == "actor":
        name = kwargs.get("name")
        return _actor_stats(df, name)

    # ── Dönem analizi ────────────────────────────────────────────────────────
    elif qt == "decade":
        return _decade_stats(df)

    # ── Top N listesi ────────────────────────────────────────────────────────
    elif qt == "top_n":
        metric = kwargs.get("metric", "revenue")
        n      = kwargs.get("n", 10)
        return _top_n(df, metric, n)

    else:
        return {"error": f"Bilinmeyen analiz türü: {query_type}"}


# ── Yardımcı Fonksiyonlar ─────────────────────────────────────────────────────

def _summary(df: pd.DataFrame) -> dict:
    return {
        "total_films": len(df),
        "total_revenue_b": round(df["revenue"].sum() / 1e9, 1),
        "avg_budget_m": round(df["budget"].mean() / 1e6, 1),
        "avg_revenue_m": round(df["revenue"].mean() / 1e6, 1),
        "avg_roi": round(df["roi"].mean(), 2),
        "hit_rate_pct": round((df["success_class"] == 2).mean() * 100, 1),
        "unique_directors": int(df["director"].nunique()),
        "avg_score": round(df["vote_average"].mean(), 2),
        "top_genre_by_revenue": str(
            df.explode("genres_list").groupby("genres_list")["revenue"].sum().idxmax()
        ),
        "max_revenue_m": round(df["revenue"].max() / 1e6, 0),
        "max_revenue_film": str(df.loc[df["revenue"].idxmax(), "title"]),
    }


def _genre_stats(df: pd.DataFrame, genre: Optional[str]) -> dict:
    df_exp = df.explode("genres_list")

    if genre:
        sub = df_exp[df_exp["genres_list"].str.lower() == genre.lower()]
        if sub.empty:
            return {"error": f"'{genre}' türü bulunamadı."}
        return {
            "genre": genre,
            "film_count": len(sub),
            "avg_revenue_m": round(sub["revenue"].mean() / 1e6, 1),
            "avg_roi": round(sub["roi"].mean(), 2),
            "hit_rate_pct": round((sub["success_class"] == 2).mean() * 100, 1),
            "avg_score": round(sub["vote_average"].mean(), 2),
            "top_film": str(sub.nlargest(1, "revenue")["title"].values[0]),
        }
    else:
        stats = df_exp.groupby("genres_list").agg(
            film_count   = ("title", "count"),
            avg_revenue  = ("revenue", "mean"),
            avg_roi      = ("roi", "mean"),
            hit_rate     = ("success_class", lambda x: (x == 2).mean()),
        ).round(2)

        top5_rev = stats.nlargest(5, "avg_revenue")[["film_count", "avg_revenue", "avg_roi", "hit_rate"]].to_dict()
        return {
            "top5_genres_by_revenue": top5_rev,
            "genres_analyzed": len(stats),
        }


def _director_stats(df: pd.DataFrame, name: Optional[str]) -> dict:
    if name:
        sub = df[df["director"].str.lower() == name.lower()]
        if sub.empty:
            return {"error": f"'{name}' isimli yönetmen bulunamadı."}
        return {
            "director": name,
            "film_count": len(sub),
            "total_revenue_b": round(sub["revenue"].sum() / 1e9, 2),
            "avg_roi": round(sub["roi"].mean(), 2),
            "avg_score": round(sub["vote_average"].mean(), 2),
            "hit_rate_pct": round((sub["success_class"] == 2).mean() * 100, 1),
            "best_film": str(sub.nlargest(1, "revenue")["title"].values[0]),
            "films": sub["title"].tolist()[:10],
        }
    else:
        dir_df = (
            df[df["director"] != "Unknown"]
            .groupby("director")
            .agg(
                film_count  = ("title", "count"),
                total_rev   = ("revenue", "sum"),
                avg_roi     = ("roi", "mean"),
                avg_score   = ("vote_average", "mean"),
            )
            .query("film_count >= 3")
            .nlargest(10, "total_rev")
        )
        return {"top10_directors": dir_df.to_dict()}


def _actor_stats(df: pd.DataFrame, name: Optional[str]) -> dict:
    df_exp = df.explode("cast_list")

    if name:
        sub = df_exp[df_exp["cast_list"].str.lower() == name.lower()]
        if sub.empty:
            return {"error": f"'{name}' isimli oyuncu bulunamadı."}
        return {
            "actor": name,
            "film_count": len(sub),
            "total_revenue_b": round(sub["revenue"].sum() / 1e9, 2),
            "avg_revenue_m": round(sub["revenue"].mean() / 1e6, 1),
            "avg_roi": round(sub["roi"].mean(), 2),
            "hit_rate_pct": round((sub["success_class"] == 2).mean() * 100, 1),
            "best_film": str(sub.nlargest(1, "revenue")["title"].values[0]),
        }
    else:
        act_df = (
            df_exp.groupby("cast_list")
            .agg(
                film_count = ("title", "count"),
                total_rev  = ("revenue", "sum"),
                avg_roi    = ("roi", "mean"),
            )
            .query("film_count >= 5")
            .nlargest(10, "total_rev")
        )
        return {"top10_actors": act_df.to_dict()}


def _decade_stats(df: pd.DataFrame) -> dict:
    decade_df = (
        df[df["decade"] != "NaTs"]
        .groupby("decade")
        .agg(
            film_count  = ("title", "count"),
            total_rev_b = ("revenue", lambda x: round(x.sum() / 1e9, 2)),
            avg_roi     = ("roi", "mean"),
            hit_rate    = ("success_class", lambda x: round((x == 2).mean() * 100, 1)),
        )
        .to_dict()
    )
    return {"decade_analysis": decade_df}


def _top_n(df: pd.DataFrame, metric: str, n: int) -> dict:
    metric_map = {
        "revenue": "revenue",
        "roi": "roi",
        "score": "vote_average",
        "profit": "profit",
        "popularity": "popularity",
    }
    col = metric_map.get(metric.lower(), "revenue")

    top = df.nlargest(n, col)[["title", "director", "release_date", col]].copy()
    top["release_year"] = top["release_date"].astype(str).str[:4]

    return {
        "metric": col,
        "top_n": n,
        "films": top[["title", "director", "release_year", col]].to_dict("records"),
    }


def format_analysis_result(result: dict, query_type: str) -> str:
    """Analiz sonucunu okunabilir metin olarak formatlar."""
    if "error" in result:
        return f"Hata: {result['error']}"

    if query_type == "summary":
        return f"""
📊 Dataset Özeti
{'='*40}
Toplam Film     : {result['total_films']:,}
Toplam Gişe     : ${result['total_revenue_b']:.1f}B
Ort. Bütçe      : ${result['avg_budget_m']:.0f}M
Ort. Gişe       : ${result['avg_revenue_m']:.0f}M
Ort. ROI        : x{result['avg_roi']:.2f}
Hit Oranı       : %{result['hit_rate_pct']:.0f}
Yönetmen Sayısı : {result['unique_directors']:,}
Ort. Puan       : {result['avg_score']:.1f}/10
En Kazançlı Tür : {result['top_genre_by_revenue']}
Rekor Gişe      : ${result['max_revenue_m']:.0f}M ({result['max_revenue_film']})
""".strip()

    return str(result)
