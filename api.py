# ─────────────────────────────────────────────────────────────
# api.py  —  FastAPI Gişe Tahmin Servisi
# Çalıştırma: uvicorn api:app --reload --port 8000
# ─────────────────────────────────────────────────────────────
import os, pickle, warnings
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import ast
warnings.filterwarnings("ignore")

# ── Sabitler ──────────────────────────────────────────────────
TMDB_KEY      = "8265bd1679663a7ea12ac168da84d2e8"   # mevcut key
YOUTUBE_KEY   = "AIzaSyDLrfwX38PZWjEVtjzu-i7AM7sfWfVBj9k"
TMDB_BASE     = "https://api.themoviedb.org/3"
YOUTUBE_BASE  = "https://www.googleapis.com/youtube/v3"
MODEL_PATH    = "models/movie_model.pkl"

app = FastAPI(title="RecAI — Gişe Tahmin API", version="1.0")

# CORS: Streamlit'in aynı makineden erişmesine izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Veri Modelleri ────────────────────────────────────────────
class PredictRequest(BaseModel):
    title: str                    # Film adı (YouTube araması için)
    budget: float                 # Bütçe (dolar)
    runtime: float                # Süre (dakika)
    popularity: float = 50.0      # TMDB popularity skoru
    vote_count: float = 0.0       # Oy sayısı (yeni film = 0)
    genre: str = "Action"         # Ana tür
    release_season: str = "summer"# "summer" | "winter" | "spring" | "fall"
    director_avg_revenue: float = 80_000_000  # Yönetmenin ort. gişesi

class PredictResponse(BaseModel):
    predicted_revenue_usd: float
    predicted_revenue_m: float
    roi_estimate: float
    hit_probability: float
    mid_probability: float
    flop_probability: float
    confidence_low: float
    confidence_high: float
    youtube_views: int
    youtube_likes: int
    trailer_sentiment_score: float  # 0-1 arası (views/likes oranı)
    model_used: str

# ── Custom Gradient Descent Modeli ───────────────────────────
class CustomGDRegressor:
    """
    Sıfırdan yazılmış Mini-Batch Gradient Descent regresyon modeli.
    Eğitim sürecini loss_history ile izlenebilir kılar.
    """
    def __init__(self, lr=0.005, epochs=400, batch_size=64, patience=20):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weights = None
        self.bias = 0.0
        self.loss_history = []
        self.best_loss = float("inf")
        self._no_improve = 0

    def _mse(self, y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    def _gradients(self, X, y_true, y_pred):
        n = len(y_true)
        err = y_pred - y_true
        dw = (2 / n) * (X.T @ err)
        db = (2 / n) * float(np.sum(err))
        return dw, db

    def fit(self, X, y):
        n_feat = X.shape[1]
        # Xavier başlatma
        self.weights = np.random.randn(n_feat) * np.sqrt(2.0 / n_feat)
        self.bias = 0.0
        self.loss_history = []
        self._no_improve = 0

        for epoch in range(self.epochs):
            idx = np.random.permutation(len(y))
            Xs, ys = X[idx], y[idx]

            for i in range(0, len(ys), self.batch_size):
                Xb = Xs[i:i + self.batch_size]
                yb = ys[i:i + self.batch_size]
                y_pred = Xb @ self.weights + self.bias
                dw, db = self._gradients(Xb, yb, y_pred)
                self.weights -= self.lr * dw
                self.bias    -= self.lr * db

            full_pred = X @ self.weights + self.bias
            loss = self._mse(full_pred, y)
            self.loss_history.append(loss)

            # Early stopping
            if loss < self.best_loss - 1e-6:
                self.best_loss = loss
                self._no_improve = 0
            else:
                self._no_improve += 1
            if self._no_improve >= self.patience:
                break

        return self

    def predict(self, X):
        return X @ self.weights + self.bias


# ── Model Eğitimi / Yükleme ───────────────────────────────────
def get_season_multiplier(season: str) -> float:
    return {"summer": 1.35, "winter": 1.20, "fall": 1.05, "spring": 0.90}.get(season, 1.0)

def get_genre_multiplier(genre: str) -> float:
    return {
        "Action": 1.25, "Adventure": 1.20, "Animation": 1.15,
        "Comedy": 1.00, "Drama": 0.85, "Horror": 0.95,
        "Science Fiction": 1.20, "Thriller": 1.00, "Romance": 0.80,
    }.get(genre, 1.0)

def load_or_train_models():
    """TMDB 5000 veri setiyle modelleri eğit veya cache'den yükle."""
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Veri seti yoksa basit sentetik veri ile eğit (demo mode)
    np.random.seed(42)
    n = 2000
    budgets    = np.random.uniform(1e6, 3e8, n)
    runtimes   = np.random.uniform(80, 180, n)
    popularity = np.random.uniform(10, 500, n)
    votes      = np.random.uniform(0, 8000, n)
    season_m   = np.random.choice([0.90, 1.05, 1.20, 1.35], n)
    genre_m    = np.random.choice([0.80, 0.95, 1.00, 1.15, 1.25], n)
    dir_rev    = np.random.uniform(2e7, 4e8, n)

    revenue = (budgets * 2.8 * season_m * genre_m
               + popularity * 500_000
               + votes * 10_000
               + dir_rev * 0.3
               + np.random.normal(0, 1e7, n))
    revenue = np.clip(revenue, 1e5, 2e9)

    X = np.column_stack([budgets, runtimes, popularity, votes, dir_rev])
    y_rev = revenue
    y_cls = np.where(revenue / budgets > 3, 2, np.where(revenue / budgets > 1.5, 1, 0))

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # sklearn GBR (production modeli)
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                    max_depth=4, random_state=42)
    gbr.fit(X, y_rev)

    # Random Forest (sınıflandırma)
    rfc = RandomForestClassifier(n_estimators=200, random_state=42)
    rfc.fit(X, y_cls)

    # Custom Gradient Descent (şeffaf/eğitsel model)
    y_norm = y_rev / 1e8
    cgd = CustomGDRegressor(lr=0.003, epochs=500, batch_size=64)
    cgd.fit(X_sc, y_norm)

    bundle = {"gbr": gbr, "rfc": rfc, "cgd": cgd,
              "scaler": scaler, "mode": "demo"}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    return bundle


# Uygulama başlarken modelleri yükle
models = load_or_train_models()


# ── YouTube Trailer Verisi ────────────────────────────────────
def fetch_youtube_stats(title: str) -> dict:
    """Film adıyla YouTube'da trailer ara, istatistikleri getir."""
    try:
        # 1. Arama
        search_url = f"{YOUTUBE_BASE}/search"
        search_params = {
            "part": "snippet",
            "q": f"{title} official trailer",
            "type": "video",
            "maxResults": 1,
            "key": YOUTUBE_KEY,
        }
        sr = requests.get(search_url, params=search_params, timeout=6)
        items = sr.json().get("items", [])
        if not items:
            return {"views": 0, "likes": 0, "sentiment": 0.5}

        video_id = items[0]["id"]["videoId"]

        # 2. İstatistik
        stats_url = f"{YOUTUBE_BASE}/videos"
        stats_params = {
            "part": "statistics",
            "id": video_id,
            "key": YOUTUBE_KEY,
        }
        vr = requests.get(stats_url, params=stats_params, timeout=6)
        stats = vr.json()["items"][0]["statistics"]
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))

        # Basit sentiment skoru: like/view oranı (normalize 0-1)
        sentiment = min(1.0, (likes / max(views, 1)) * 20)
        return {"views": views, "likes": likes, "sentiment": round(sentiment, 3)}

    except Exception:
        return {"views": 0, "likes": 0, "sentiment": 0.5}


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "RecAI Gişe Tahmin API v1.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_mode": models.get("mode", "unknown")}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        season_m = get_season_multiplier(req.release_season)
        genre_m  = get_genre_multiplier(req.genre)

        X_raw = np.array([[
            req.budget,
            req.runtime,
            req.popularity,
            req.vote_count,
            req.director_avg_revenue,
        ]])

        # YouTube trailer verisi
        yt = fetch_youtube_stats(req.title)

        # Trailer görüntüleme bonusu (her 1M görüntüleme ≈ $2M gişe etkisi)
        trailer_bonus = (yt["views"] / 1_000_000) * 2_000_000

        # ── GBR Tahmini ──
        gbr_pred = float(models["gbr"].predict(X_raw)[0])
        gbr_pred = gbr_pred * season_m * genre_m + trailer_bonus
        gbr_pred = max(0, gbr_pred)

        # ── Custom GD Tahmini ──
        X_sc = models["scaler"].transform(X_raw)
        cgd_pred = float(models["cgd"].predict(X_sc)[0]) * 1e8
        cgd_pred = cgd_pred * season_m * genre_m + trailer_bonus
        cgd_pred = max(0, cgd_pred)

        # Ensemble: GBR %70 + CustomGD %30
        final_rev = 0.70 * gbr_pred + 0.30 * cgd_pred

        # ── Sınıflandırma (olasılıklar) ──
        probs = models["rfc"].predict_proba(X_raw)[0]
        # probs sıralaması: [flop, mid, hit]
        if len(probs) == 3:
            prob_flop, prob_mid, prob_hit = probs
        else:
            prob_hit, prob_mid, prob_flop = 0.33, 0.33, 0.34

        roi_est = final_rev / max(req.budget, 1)

        # ± %20 güven aralığı
        conf_low  = final_rev * 0.80
        conf_high = final_rev * 1.20

        return PredictResponse(
            predicted_revenue_usd = round(final_rev, 2),
            predicted_revenue_m   = round(final_rev / 1e6, 2),
            roi_estimate          = round(roi_est, 2),
            hit_probability       = round(float(prob_hit) * 100, 1),
            mid_probability       = round(float(prob_mid) * 100, 1),
            flop_probability      = round(float(prob_flop) * 100, 1),
            confidence_low        = round(conf_low / 1e6, 2),
            confidence_high       = round(conf_high / 1e6, 2),
            youtube_views         = yt["views"],
            youtube_likes         = yt["likes"],
            trailer_sentiment_score = yt["sentiment"],
            model_used            = "GBR(70%) + CustomGD(30%) + YouTube",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/loss_history")
def loss_history():
    """Custom GD eğitim kayıp geçmişi — Streamlit grafiği için."""
    hist = models["cgd"].loss_history if hasattr(models.get("cgd"), "loss_history") else []
    return {"epochs": list(range(len(hist))), "loss": hist}
