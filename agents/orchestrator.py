"""
orchestrator.py — LangGraph tabanlı ana koordinatör agent.

Görev:
  - system_prompt.md'yi okuyarak başlar
  - Kullanıcı sorgusunu analiz eder
  - Doğru agent(ları) sırayla çalıştırır
  - Sonuçları birleştirir ve kullanıcıya sunar

Not: LangGraph'ın StateGraph yapısını kullanır.
     LLM olmadan da (rule-based routing) çalışır.
"""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import TypedDict, Optional, List, Literal, Any

# ── LangGraph (pip install langgraph) ─────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠ LangGraph yüklü değil. Rule-based fallback kullanılıyor.")
    print("  Yüklemek için: pip install langgraph")

from agents.search_agent      import search_movie, format_search_result
from agents.prediction_agent  import predict_movie, format_prediction_result
from agents.analysis_agent    import analyze_dataset, format_analysis_result
from agents.recommendation_agent import recommend_movies, format_recommendation_result


# ─────────────────────────────────────────────────────────────────────────────
# State Tanımı
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    user_query:      str
    intent:          Optional[str]          # 'search' | 'predict' | 'analyze' | 'recommend'
    movie_title:     Optional[str]
    search_result:   Optional[dict]
    prediction_result: Optional[dict]
    analysis_result: Optional[dict]
    recommendation_result: Optional[dict]
    final_response:  Optional[str]
    error:           Optional[str]
    # Runtime references (DataFrame + modeller)
    df:              Any
    reg_model:       Any
    clf_model:       Any


# ─────────────────────────────────────────────────────────────────────────────
# Sistem Prompt Yükleme
# ─────────────────────────────────────────────────────────────────────────────
def load_system_prompt(path: str = "system_prompt.md") -> str:
    """
    system_prompt.md'yi okur. Agent her başlatıldığında görevini bu dosyadan öğrenir.
    """
    prompt_path = Path(path)
    if not prompt_path.exists():
        # Alternatif konumlar dene
        alt = Path(__file__).parent.parent / "system_prompt.md"
        if alt.exists():
            prompt_path = alt
        else:
            return "Sistem promptu bulunamadı. Varsayılan davranış uygulanıyor."

    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
# Intent Belirleme (Rule-Based Router)
# ─────────────────────────────────────────────────────────────────────────────
def classify_intent(query: str) -> tuple[str, Optional[str]]:
    """
    Kullanıcı sorgusunun amacını belirler.

    Returns:
        (intent, movie_title)
        intent: 'search_predict' | 'analyze' | 'recommend'
    """
    q_lower = query.lower()

    # ── Öneri sinyalleri ──────────────────────────────────────────────────────
    recommend_signals = [
        "ne izlesem", "öneri", "öner", "tavsiye", "film öner",
        "önerin", "bir şeyler", "akşam", "film seç",
    ]
    if any(s in q_lower for s in recommend_signals):
        return "recommend", None

    # ── Analiz sinyalleri ────────────────────────────────────────────────────
    analyze_signals = [
        "en iyi", "top", "liste", "analiz", "istatistik",
        "ortalama", "kaç film", "toplam", "dataset", "karşılaştır",
        "hangi tür", "hangi yönetmen", "hangi oyuncu",
    ]
    if any(s in q_lower for s in analyze_signals):
        # Spesifik bir film adı var mı?
        title = _extract_movie_title(query)
        if title and _looks_like_movie(q_lower):
            return "search_predict", title
        return "analyze", None

    # ── Film arama/tahmin sinyalleri ─────────────────────────────────────────
    search_signals = [
        "film", "movie", "başarılı", "gişe", "tahmin", "izledim",
        "izleyeceğim", "yönetmen", "oyuncu", "ara", "bul",
    ]
    if any(s in q_lower for s in search_signals):
        title = _extract_movie_title(query)
        if title:
            return "search_predict", title

    # ── Direkt film adı gibi görünen sorgular ─────────────────────────────────
    title = _extract_movie_title(query)
    if title:
        return "search_predict", title

    # ── Varsayılan ────────────────────────────────────────────────────────────
    return "analyze", None


def _extract_movie_title(query: str) -> Optional[str]:
    """Sorgudaki muhtemel film adını çıkarır."""
    # Tırnak içindeki metni al
    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
    if quoted:
        return quoted[0][0] or quoted[0][1]

    # Anahtar kelimelerden sonra gelen metni al
    patterns = [
        r"(?:film(?:ini|i)?|movie)\s+[\"']?(.+?)[\"']?\s*(?:analiz|tahmin|ara|bul|için|hakkında|ne düşün)?$",
        r"(.+?)\s+(?:filmini?|isimli film|adlı film)",
        r"^(.+?)\s+(?:başarılı mı|gişe|tahmin|analiz)",
    ]
    for pat in patterns:
        m = re.search(pat, query, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().strip('"\'')
            if len(candidate) > 2:
                return candidate

    # Kısa ve temiz sorgu — tümünü film adı say
    words = query.strip().split()
    if 1 <= len(words) <= 5 and not any(
        w in query.lower() for w in ["öneri", "analiz", "liste", "ne izle"]
    ):
        return query.strip()

    return None


def _looks_like_movie(text: str) -> bool:
    movie_words = ["film", "movie", "izle", "başarılı", "gişe", "oyuncu", "yönetmen"]
    return any(w in text for w in movie_words)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Düğümleri (Nodes)
# ─────────────────────────────────────────────────────────────────────────────

def node_router(state: AgentState) -> AgentState:
    """Kullanıcı sorgusunu analiz eder ve intent belirler."""
    intent, title = classify_intent(state["user_query"])
    return {**state, "intent": intent, "movie_title": title}


def node_search(state: AgentState) -> AgentState:
    """TMDB'den film bilgisi çeker."""
    title = state.get("movie_title")
    if not title:
        return {**state, "error": "Film adı belirtilmedi."}

    result = search_movie(title)
    return {**state, "search_result": result}


def node_predict(state: AgentState) -> AgentState:
    """Gişe tahmini yapar."""
    search_result = state.get("search_result")
    if not search_result or not search_result.get("found"):
        err = search_result.get("error", "Film verisi yok") if search_result else "Arama yapılmadı"
        return {**state, "error": err}

    result = predict_movie(
        movie_data=search_result,
        df=state["df"],
        reg_model=state["reg_model"],
        clf_model=state["clf_model"],
    )
    return {**state, "prediction_result": result}


def node_analyze(state: AgentState) -> AgentState:
    """Dataset analizi yapar."""
    query = state["user_query"].lower()

    # Basit kural tabanlı analiz türü seçimi
    if "özet" in query or "genel" in query or "toplam" in query:
        qt = "summary"
        kwargs = {}
    elif "yönetmen" in query:
        qt = "director"
        name = _extract_name_from_query(query, ["yönetmen"])
        kwargs = {"name": name}
    elif "oyuncu" in query:
        qt = "actor"
        name = _extract_name_from_query(query, ["oyuncu", "başrol"])
        kwargs = {"name": name}
    elif "dönem" in query or "yıl" in query or "decade" in query:
        qt = "decade"
        kwargs = {}
    elif "tür" in query or "genre" in query:
        qt = "genre"
        kwargs = {}
    else:
        qt = "summary"
        kwargs = {}

    result = analyze_dataset(qt, state["df"], **kwargs)
    return {**state, "analysis_result": result, "_analysis_type": qt}


def node_recommend(state: AgentState) -> AgentState:
    """Film önerir."""
    query = state["user_query"]

    result = recommend_movies(
        df=state["df"],
        mood_keywords=query,
        min_score=7.0,
        sort_by="popularity",
        n=5,
    )
    return {**state, "recommendation_result": result}


def node_respond(state: AgentState) -> AgentState:
    """Tüm sonuçları birleştirip kullanıcıya yanıt üretir."""
    parts = []
    intent = state.get("intent", "")

    # Hata var mı?
    if state.get("error"):
        return {**state, "final_response": f"❌ {state['error']}"}

    if state.get("search_result") and state["search_result"].get("found"):
        parts.append(format_search_result(state["search_result"]))

    if state.get("prediction_result"):
        title = state.get("movie_title", "Film")
        parts.append("\n" + format_prediction_result(state["prediction_result"], title))

    if state.get("analysis_result"):
        qt = state.get("_analysis_type", "summary")
        parts.append(format_analysis_result(state["analysis_result"], qt))

    if state.get("recommendation_result"):
        parts.append(format_recommendation_result(state["recommendation_result"]))

    response = "\n\n".join(parts) if parts else "Yanıt üretilemedi."
    return {**state, "final_response": response}


def _extract_name_from_query(query: str, stop_words: list) -> Optional[str]:
    """Sorgudaki kişi adını çıkarır."""
    for sw in stop_words:
        if sw in query:
            parts = query.split(sw)
            candidate = parts[-1].strip().split()[0:3]
            if candidate:
                return " ".join(candidate).strip("?.,!")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Router Fonksiyonu (Conditional Edge)
# ─────────────────────────────────────────────────────────────────────────────
def route_intent(state: AgentState) -> Literal["search", "analyze", "recommend"]:
    intent = state.get("intent", "analyze")
    if intent == "search_predict":
        return "search"
    elif intent == "recommend":
        return "recommend"
    else:
        return "analyze"


# ─────────────────────────────────────────────────────────────────────────────
# Graf İnşası
# ─────────────────────────────────────────────────────────────────────────────
def build_graph():
    """
    LangGraph StateGraph'ı oluşturur ve derler.

    Akış:
      router → [search → predict] | [analyze] | [recommend] → respond → END
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    g = StateGraph(AgentState)

    g.add_node("router",    node_router)
    g.add_node("search",    node_search)
    g.add_node("predict",   node_predict)
    g.add_node("analyze",   node_analyze)
    g.add_node("recommend", node_recommend)
    g.add_node("respond",   node_respond)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        route_intent,
        {
            "search":    "search",
            "analyze":   "analyze",
            "recommend": "recommend",
        },
    )

    g.add_edge("search",    "predict")
    g.add_edge("predict",   "respond")
    g.add_edge("analyze",   "respond")
    g.add_edge("recommend", "respond")
    g.add_edge("respond",   END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Ana Fonksiyon
# ─────────────────────────────────────────────────────────────────────────────
def run_agent(
    query: str,
    df: pd.DataFrame,
    reg_model,
    clf_model,
    graph=None,
) -> str:
    """
    Kullanıcı sorgusunu çalıştırır ve yanıt döner.

    Args:
        query:     Kullanıcı sorusu
        df:        TMDB DataFrame
        reg_model: GradientBoostingRegressor
        clf_model: RandomForestClassifier
        graph:     Derlenmiş LangGraph (None ise basit fallback)

    Returns:
        str: Kullanıcıya gösterilecek yanıt
    """
    # Sistem promptunu oku
    system_prompt = load_system_prompt()

    initial_state: AgentState = {
        "user_query":            query,
        "intent":                None,
        "movie_title":           None,
        "search_result":         None,
        "prediction_result":     None,
        "analysis_result":       None,
        "recommendation_result": None,
        "final_response":        None,
        "error":                 None,
        "df":                    df,
        "reg_model":             reg_model,
        "clf_model":             clf_model,
    }

    if graph is not None:
        # LangGraph ile çalış
        final = graph.invoke(initial_state)
    else:
        # Fallback: düğümleri sırayla çalıştır
        s = node_router(initial_state)
        if s["intent"] == "search_predict":
            s = node_search(s)
            s = node_predict(s)
        elif s["intent"] == "recommend":
            s = node_recommend(s)
        else:
            s = node_analyze(s)
        final = node_respond(s)

    return final.get("final_response", "Yanıt üretilemedi.")
