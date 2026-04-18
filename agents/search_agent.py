"""
search_agent.py — TMDB API'den film bilgisi çeken agent.

Görev:
  - Film adıyla TMDB'de arama yapar
  - Detaylı metadata toplar (yönetmen, oyuncular, türler, bütçe, gişe)
  - Poster URL'si döner
  - prediction_agent için ham veri hazırlar
"""

import requests
from typing import Optional

TMDB_KEY = "9dff4a1400db6ba14b347ce0f29b33a8"
try:
    import streamlit as st
    if "TMDB_API_KEY" in st.secrets:
        TMDB_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    pass

BASE_URL  = "https://api.themoviedb.org/3"
POSTER    = "https://image.tmdb.org/t/p/w500"


def search_movie(query: str, language: str = "tr-TR") -> dict:
    """
    Film adıyla TMDB'de arama yapar ve detaylı bilgi toplar.

    Args:
        query: Film adı (Türkçe veya İngilizce)
        language: TMDB API dil kodu

    Returns:
        dict: Film metadata'sı veya hata mesajı
    """
    try:
        # ── 1. Arama ──────────────────────────────────────────────
        search_res = requests.get(
            f"{BASE_URL}/search/movie",
            params={"api_key": TMDB_KEY, "query": query, "language": language},
            timeout=8,
        ).json()

        results = search_res.get("results", [])
        if not results:
            return {"error": f"'{query}' için film bulunamadı.", "found": False}

        movie = results[0]
        movie_id = movie["id"]

        # ── 2. Detay + Ekip Bilgisi ────────────────────────────────
        detail = requests.get(
            f"{BASE_URL}/movie/{movie_id}",
            params={
                "api_key": TMDB_KEY,
                "language": language,
                "append_to_response": "credits",
            },
            timeout=8,
        ).json()

        crew  = detail.get("credits", {}).get("crew", [])
        cast  = detail.get("credits", {}).get("cast", [])

        directors = [p["name"] for p in crew if p.get("job") == "Director"]
        top_cast  = [p["name"] for p in cast[:5]]
        genres    = [g["name"] for g in detail.get("genres", [])]

        budget  = detail.get("budget", 0)
        revenue = detail.get("revenue", 0)
        runtime = detail.get("runtime") or 100

        poster_path = movie.get("poster_path") or detail.get("poster_path")
        poster_url  = (POSTER + poster_path) if poster_path else None

        return {
            "found": True,
            "id": movie_id,
            "title": movie.get("title", ""),
            "original_title": movie.get("original_title", ""),
            "overview": (movie.get("overview") or detail.get("overview", ""))[:500],
            "release_date": movie.get("release_date", "N/A"),
            "vote_average": movie.get("vote_average", 0),
            "vote_count": movie.get("vote_count", 0),
            "popularity": movie.get("popularity", 0),
            "runtime": runtime,
            "budget": budget,
            "revenue": revenue,
            "genres": genres,
            "directors": directors,
            "cast": top_cast,
            "poster_url": poster_url,
        }

    except requests.exceptions.Timeout:
        return {"error": "TMDB API zaman aşımı.", "found": False}
    except Exception as e:
        return {"error": f"Beklenmeyen hata: {str(e)}", "found": False}


def format_search_result(data: dict) -> str:
    """
    search_movie() sonucunu okunabilir metin olarak formatlar.
    Orchestrator'ın LLM'e göndereceği context için kullanılır.
    """
    if not data.get("found"):
        return f"Hata: {data.get('error', 'Bilinmeyen hata')}"

    budget_str  = f"${data['budget']/1e6:.0f}M"  if data['budget']  > 0 else "Bilinmiyor"
    revenue_str = f"${data['revenue']/1e6:.0f}M" if data['revenue'] > 0 else "Bilinmiyor"

    return f"""
Film: {data['title']} ({data.get('original_title','')})
Tarih: {data['release_date']}
Puan: {data['vote_average']:.1f}/10 ({data['vote_count']:,} oy)
Popülarite: {data['popularity']:.0f}
Süre: {data['runtime']} dk
Bütçe: {budget_str}
Gişe: {revenue_str}
Türler: {', '.join(data['genres'])}
Yönetmen: {', '.join(data['directors']) if data['directors'] else 'Bilinmiyor'}
Başrol: {', '.join(data['cast']) if data['cast'] else 'Bilinmiyor'}
Özet: {data['overview']}
""".strip()
