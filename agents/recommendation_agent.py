"""
recommendation_agent.py — Kullanıcı tercihine göre film öneren agent.

Görev:
  - Tür, dönem, puan, başarı filtrelerine göre film öner
  - Kullanıcının ruh haline göre akıllı seçim yap
  - ROI, puan ve popülarite dengesini gözet
  - Rastgele seçimle tekrarlı öneri engelle
"""

import numpy as np
import pandas as pd
import random
from typing import Optional, List


def recommend_movies(
    df: pd.DataFrame,
    genre: Optional[str] = None,
    min_score: float = 7.0,
    decade: Optional[str] = None,
    success_filter: Optional[str] = None,
    sort_by: str = "popularity",
    mood_keywords: Optional[str] = None,
    n: int = 5,
    seed: Optional[int] = None,
) -> dict:
    """
    Filtrelere göre film önerir.

    Args:
        df:             TMDB 5000 DataFrame
        genre:          Film türü (None = hepsi)
        min_score:      Minimum IMDB puanı
        decade:         Dönem filtresi ('1990s', '2000s', vb.)
        success_filter: 'hit' | 'mid' | None (hepsi)
        sort_by:        'popularity' | 'score' | 'revenue' | 'roi'
        mood_keywords:  Kullanıcının ruh hali metni (tür çıkarımı için)
        n:              Önerilecek film sayısı
        seed:           Rastgele seçim seed'i (tekrar önleme)

    Returns:
        dict: Önerilen filmler ve gerekçeleri
    """
    filtered = df.copy()

    # ── Ruh hali → Tür çıkarımı ───────────────────────────────────────────────
    if mood_keywords and not genre:
        genre = _infer_genre_from_mood(mood_keywords)

    # ── Tür filtresi ─────────────────────────────────────────────────────────
    if genre and genre.lower() != "hepsi":
        filtered = filtered[
            filtered["genres_list"].apply(
                lambda gl: any(g.lower() == genre.lower() for g in gl)
            )
        ]

    # ── Puan filtresi ─────────────────────────────────────────────────────────
    filtered = filtered[filtered["vote_average"] >= min_score]

    # ── Dönem filtresi ────────────────────────────────────────────────────────
    if decade and decade.lower() != "hepsi":
        filtered = filtered[filtered["decade"] == decade]

    # ── Başarı filtresi ───────────────────────────────────────────────────────
    if success_filter == "hit":
        filtered = filtered[filtered["success_class"] == 2]
    elif success_filter == "mid":
        filtered = filtered[filtered["success_class"] == 1]

    if filtered.empty:
        return {
            "found": False,
            "message": "Filtrelere uyan film bulunamadı. Kriterleri genişletin.",
            "films": [],
        }

    # ── Sıralama ──────────────────────────────────────────────────────────────
    sort_map = {
        "popularity": "popularity",
        "score": "vote_average",
        "revenue": "revenue",
        "roi": "roi",
    }
    sort_col = sort_map.get(sort_by.lower(), "popularity")

    # Top 50'den rastgele n tane seç (çeşitlilik sağlar)
    top_pool = filtered.nlargest(min(50, len(filtered)), sort_col)
    if seed is not None:
        random.seed(seed)
    sample_size = min(n, len(top_pool))
    selected = top_pool.sample(n=sample_size, random_state=seed)

    films = []
    for _, row in selected.iterrows():
        genres_str = ", ".join(row["genres_list"]) if row["genres_list"] else "—"

        films.append({
            "title": row["title"],
            "director": row.get("director", "—"),
            "release_year": str(row.get("release_date", ""))[:4],
            "score": round(row["vote_average"], 1),
            "popularity": round(row["popularity"], 0),
            "revenue_m": round(row["revenue"] / 1e6, 0),
            "roi": round(row["roi"], 2),
            "genres": genres_str,
            "success": _success_label(row["success_class"]),
        })

    return {
        "found": True,
        "total_matching": len(filtered),
        "genre_used": genre or "Hepsi",
        "sort_by": sort_by,
        "films": films,
        "mood_genre_inferred": genre if mood_keywords else None,
    }


def _success_label(sc: int) -> str:
    return {2: "🟢 Hit", 1: "🟡 Orta", 0: "🔴 Riskli"}.get(sc, "—")


def _infer_genre_from_mood(text: str) -> Optional[str]:
    """Kullanıcının ruh hali metninden tür çıkarır."""
    text = text.lower()
    mood_map = {
        "Action":           ["aksiyon", "heyecan", "nefes kesen", "macera", "savaş", "kavga", "patlama"],
        "Comedy":           ["komedi", "gülmek", "hafif", "eğlenceli", "neşeli", "komik"],
        "Drama":            ["derin", "duygusal", "ağlamak", "drama", "gerçekçi", "etkileyici"],
        "Horror":           ["korku", "gerilim", "ürkütücü", "hayalet"],
        "Science Fiction":  ["bilim kurgu", "uzay", "robot", "gelecek", "teknoloji", "yapay zeka"],
        "Romance":          ["aşk", "romantik", "sevgi", "ilişki"],
        "Animation":        ["animasyon", "çizgi film", "aile", "çocuk"],
        "Thriller":         ["gerilim", "sürpriz", "twist", "gizem"],
        "Fantasy":          ["fantezi", "ejderha", "büyü", "büyücü", "elf"],
        "Crime":            ["suç", "cinayet", "dedektif", "gangster", "polis"],
    }
    for genre, keywords in mood_map.items():
        if any(kw in text for kw in keywords):
            return genre
    return None


def format_recommendation_result(result: dict) -> str:
    """Öneri sonucunu okunabilir metin olarak formatlar."""
    if not result.get("found"):
        return f"⚠️ {result.get('message', 'Film bulunamadı.')}"

    mood_note = ""
    if result.get("mood_genre_inferred"):
        mood_note = f"\n💭 Ruh halinize göre tür: {result['mood_genre_inferred']}"

    films_text = ""
    for i, f in enumerate(result["films"], 1):
        films_text += (
            f"\n{i}. {f['title']} ({f['release_year']})\n"
            f"   🎬 {f['director']} • ⭐ {f['score']}/10 • {f['success']}\n"
            f"   🎭 {f['genres']} • 💰 ${f['revenue_m']:.0f}M Gişe\n"
        )

    return f"""
🍿 Film Önerileri — Tür: {result['genre_used']} | Sıralama: {result['sort_by']}
{mood_note}
Uygun Film: {result['total_matching']:,} adet
{'='*50}
{films_text}
""".strip()
