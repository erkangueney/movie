import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
import concurrent.futures
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Movie Matrix AI Pro", layout="wide", initial_sidebar_state="collapsed")

BG_URL = "https://images.unsplash.com/photo-1543722530-d2c3201371e7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

/* ── Base ── */
.stApp {{
    background: linear-gradient(135deg,rgba(0,0,0,0.92),rgba(5,5,20,0.95)),
                url("{BG_URL}");
    background-size: cover;
    background-attachment: fixed;
    color: #e0e8ff;
    font-family: 'Rajdhani', sans-serif;
}}
header, #MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 90px !important; }}

/* ── Navbar ── */
.fixed-header {{
    position: fixed; top: 0; left: 0; width: 100%;
    background: rgba(5,5,20,0.96);
    backdrop-filter: blur(16px);
    padding: 12px 30px;
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid rgba(0,243,255,0.3);
    z-index: 9999;
    box-shadow: 0 0 30px rgba(0,243,255,0.1);
}}
.nav-brand {{
    font-family: 'Orbitron', sans-serif;
    font-size: 18px; font-weight: 900;
    color: #00f3ff;
    text-shadow: 0 0 20px rgba(0,243,255,0.6);
    letter-spacing: 2px;
}}
.nav-links {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.nav-links a {{
    color: rgba(0,243,255,0.7);
    text-decoration: none;
    font-family: 'Orbitron', sans-serif;
    font-size: 10px; font-weight: 700;
    padding: 6px 12px;
    border: 1px solid rgba(0,243,255,0.2);
    border-radius: 4px;
    transition: all 0.3s;
    letter-spacing: 1px;
}}
.nav-links a:hover {{
    color: #ff8800;
    border-color: #ff8800;
    text-shadow: 0 0 10px #ff8800;
    box-shadow: 0 0 15px rgba(255,136,0,0.2);
}}

/* ── Section anchor offset ── */
.section-anchor {{ padding-top: 80px; margin-top: -60px; display: block; }}

/* ── Glass Card ── */
.glass-card {{
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,243,255,0.15);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05);
}}

/* ── Section Title ── */
.section-title {{
    font-family: 'Orbitron', sans-serif;
    font-size: 22px; font-weight: 900;
    text-align: center;
    padding: 15px 0 5px;
    letter-spacing: 3px;
}}
.cyan {{ color: #00f3ff; text-shadow: 0 0 20px rgba(0,243,255,0.5); }}
.orange {{ color: #ff8800; text-shadow: 0 0 20px rgba(255,136,0,0.5); }}
.gold {{ color: #ffd700; text-shadow: 0 0 20px rgba(255,215,0,0.5); }}

/* ── KPI Stats ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px; margin: 20px 0;
}}
.kpi-box {{
    text-align: center; padding: 18px 10px;
    border-radius: 12px;
    border: 1px solid;
    transition: transform 0.2s;
}}
.kpi-box:hover {{ transform: translateY(-3px); }}
.kpi-cyan  {{ border-color:#00f3ff; background:rgba(0,243,255,0.05); box-shadow:0 0 20px rgba(0,243,255,0.15); }}
.kpi-orange2    {{ border-color:#ff8800; background:rgba(255,136,0,0.05); box-shadow:0 0 20px rgba(255,136,0,0.15); }}
.kpi-gold  {{ border-color:#ffd700; background:rgba(255,215,0,0.05); box-shadow:0 0 20px rgba(255,215,0,0.15); }}
.kpi-green {{ border-color:#00ff88; background:rgba(0,255,136,0.05); box-shadow:0 0 20px rgba(0,255,136,0.15); }}
.kpi-orange{{ border-color:#ff6b35; background:rgba(255,107,53,0.05); box-shadow:0 0 20px rgba(255,107,53,0.15); }}
.kpi-val   {{ font-family:'Orbitron',sans-serif; font-size:26px; font-weight:900; }}
.kpi-lbl   {{ font-size:11px; letter-spacing:1.5px; opacity:0.7; margin-top:4px; font-weight:600; }}

/* ── Divider ── */
.neon-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, #00f3ff, #ff8800, transparent);
    margin: 20px 0;
    opacity: 0.4;
}}

/* ── Sub-header ── */
.sub-header {{
    font-family: 'Orbitron', sans-serif;
    font-size: 13px;
    letter-spacing: 2px;
    color: rgba(0,243,255,0.8);
    margin-bottom: 10px;
    font-weight: 700;
}}

/* ── Top-10 Table ── */
.top10-row {{
    display: flex; align-items: center;
    padding: 10px 14px; margin: 6px 0;
    border-radius: 8px;
    background: rgba(255,255,255,0.03);
    border-left: 3px solid;
    transition: all 0.2s;
}}
.top10-row:hover {{ background: rgba(255,255,255,0.07); transform: translateX(3px); }}
.top10-rank {{
    font-family: 'Orbitron', sans-serif;
    font-size: 18px; font-weight: 900;
    min-width: 40px;
}}
.top10-info {{ flex: 1; padding: 0 12px; }}
.top10-title {{ font-weight: 700; font-size: 14px; }}
.top10-sub {{ font-size: 11px; opacity: 0.6; }}
.top10-val {{
    font-family: 'Share Tech Mono', monospace;
    font-size: 15px; font-weight: 700;
    text-align: right;
}}

/* ── Pop-Up Tooltips & Images ── */
.tooltip-container {{
    position: relative;
    cursor: pointer;
}}
.tooltip-container .tooltip-content {{
    visibility: hidden;
    position: absolute;
    bottom: 80%;
    left: 50%;
    transform: translateX(-50%) translateY(10px);
    background: rgba(10, 15, 30, 0.98);
    border: 1px solid rgba(0,243,255,0.4);
    border-radius: 12px;
    padding: 16px;
    width: max-content;
    max-width: 250px;
    color: #fff;
    box-shadow: 0 10px 30px rgba(0,0,0,0.8), 0 0 20px rgba(0,243,255,0.2);
    opacity: 0;
    transition: all 0.3s cubic-bezier(0.18, 0.89, 0.32, 1.28);
    z-index: 99999;
    text-align: center;
    pointer-events: none;
}}
.tooltip-container:hover .tooltip-content {{
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}}
.tooltip-content img {{
    border-radius: 8px;
    margin-bottom: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    max-height: 120px;
    object-fit: cover;
}}
.tooltip-title {{
    font-family: 'Orbitron', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #ff8800;
    margin-bottom: 6px;
}}
.tooltip-sub {{
    font-size: 12px;
    opacity: 0.8;
}}
.avatar-img {{
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 12px;
    border: 2px solid rgba(0,243,255,0.3);
    transition: all 0.3s;
}}
.tooltip-container:hover .avatar-img {{
    border-color: #ff8800;
    transform: scale(1.1);
    box-shadow: 0 0 15px rgba(255,136,0,0.4);
}}

/* ── Prediction Box ── */
.pred-result {{
    background: rgba(0,243,255,0.07);
    border: 1px solid rgba(0,243,255,0.4);
    border-radius: 16px;
    padding: 30px; text-align: center;
    box-shadow: 0 0 30px rgba(0,243,255,0.1);
}}
.pred-big {{ font-family:'Orbitron',sans-serif; font-size:48px; font-weight:900; }}
.pred-label {{ font-size:13px; letter-spacing:2px; opacity:0.7; margin-top:4px; }}

/* ── AI Search Result ── */
.movie-card-live {{
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,136,0,0.3);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 0 20px rgba(255,136,0,0.1);
}}

/* ── Feature Importance Bar ── */
.feat-bar-wrap {{ margin: 8px 0; }}
.feat-bar-label {{ font-size:12px; font-weight:600; margin-bottom:4px; display:flex; justify-content:space-between; }}
.feat-bar-bg {{ background:rgba(255,255,255,0.07); border-radius:4px; height:10px; }}
.feat-bar-fill {{ height:10px; border-radius:4px; }}

/* ── Person Card ── */
.person-card {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,243,255,0.2);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    transition: all 0.2s;
}}
.person-card:hover {{
    border-color: #ff8800;
    box-shadow: 0 0 20px rgba(255,136,0,0.2);
    transform: translateY(-2px);
}}
.person-name {{ font-weight: 700; font-size: 13px; margin-top: 8px; }}
.person-meta {{ font-size: 11px; opacity: 0.6; }}
.person-stat {{ font-family:'Share Tech Mono',monospace; font-size:14px; color:#00f3ff; margin-top:6px; font-weight:700; }}

/* ── Stagger animation ── */
@keyframes fadeSlideIn {{
    from {{ opacity:0; transform:translateY(15px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
.anim-in {{ animation: fadeSlideIn 0.5s ease forwards; }}

/* ── Streamlit widget overrides ── */
.stSlider > div {{ color: #00f3ff !important; }}
.stButton > button {{
    background: linear-gradient(135deg, rgba(0,243,255,0.15), rgba(255,136,0,0.15));
    color: #fff;
    border: 1px solid rgba(0,243,255,0.5);
    border-radius: 8px;
    font-family: 'Orbitron', sans-serif;
    font-size: 12px;
    letter-spacing: 2px;
    padding: 10px 24px;
    font-weight: 700;
    transition: all 0.3s;
    width: 100%;
}}
.stButton > button:hover {{
    border-color: #ff8800;
    box-shadow: 0 0 20px rgba(255,136,0,0.3);
}}
.stTextInput input, .stTextArea textarea, .stNumberInput input {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0,243,255,0.3) !important;
    color: #fff !important;
    border-radius: 8px !important;
}}
.stSelectbox > div > div {{
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0,243,255,0.3) !important;
    color: #fff !important;
}}
</style>

<div class="fixed-header">
  <div class="nav-brand">🎬 MOVIE MATRIX AI</div>
  <div class="nav-links">
    <a href="#dashboard">📊 DASHBOARD</a>
    <a href="#eda">📈 ETKİ ANALİZİ</a>
    <a href="#top10">🏆 TOP 10</a>
    <a href="#people">🌟 YÖNETMEN & OYUNCU</a>
    <a href="#prediction">🔮 GİŞE TAHMİNİ</a>
    <a href="#live-ai">🔍 CANLI AI</a>
    <a href="#mood-watch">🍿 NE İZLESEM?</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
API_KEY   = "9dff4a1400db6ba14b347ce0f29b33a8"
BASE_URL  = "https://api.themoviedb.org/3"
POSTER_URL = "https://image.tmdb.org/t/p/w500"

DARK_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Rajdhani', color='#e0e8ff'),
)

@st.cache_data(show_spinner=False)
def fetch_tmdb_image(query, doc_type="movie"):
    try:
        url = f"{BASE_URL}/search/{doc_type}?api_key={API_KEY}&query={query}&language=tr-TR"
        res = requests.get(url, timeout=3).json()
        if res.get('results'):
            path = res['results'][0].get('poster_path' if doc_type == 'movie' else 'profile_path')
            if path:
                return POSTER_URL + path
    except:
        pass
    return f"https://ui-avatars.com/api/?name={query.replace(' ', '+')}&background=random&color=fff"

def prefetch_images(queries_dict, doc_type="movie"):
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_q = {executor.submit(fetch_tmdb_image, q, doc_type): q for q in queries_dict.values()}
        for future in concurrent.futures.as_completed(future_to_q):
            q = future_to_q[future]
            results[q] = future.result()
    return results

# ─────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    m = pd.read_csv('tmdb_5000_movies.csv')
    c = pd.read_csv('tmdb_5000_credits.csv')
    df = m.merge(c, on='title')
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].reset_index(drop=True)

    def parse_json_col(obj):
        try: return [i['name'] for i in ast.literal_eval(obj)]
        except: return []

    def parse_crew_role(obj, job_filter):
        try:
            return [i['name'] for i in ast.literal_eval(obj) if i.get('job') == job_filter]
        except: return []

    def parse_cast(obj, limit=5):
        try: return [i['name'] for i in ast.literal_eval(obj)][:limit]
        except: return []

    df['genres_list']     = df['genres'].apply(parse_json_col)
    df['keywords_list']   = df['keywords'].apply(parse_json_col)
    df['companies_list']  = df['production_companies'].apply(parse_json_col)
    df['countries_list']  = df['production_countries'].apply(parse_json_col)
    df['cast_list']       = df['cast'].apply(lambda x: parse_cast(x, 5))
    df['director_list']   = df['crew'].apply(lambda x: parse_crew_role(x, 'Director'))
    df['director']        = df['director_list'].apply(lambda x: x[0] if x else 'Unknown')

    df['roi']            = df['revenue'] / df['budget']
    df['profit']         = df['revenue'] - df['budget']
    df['success_class']  = df['roi'].apply(lambda x: 2 if x > 3 else (1 if x > 1.5 else 0))
    df['decade']         = (pd.to_datetime(df['release_date'], errors='coerce').dt.year // 10 * 10).astype('Int64').astype(str) + 's'

    return df

df = load_data()

# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(data):
    X   = data[['budget', 'runtime', 'popularity', 'vote_count']].fillna(0)
    reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42).fit(X, data['revenue'])
    clf = RandomForestClassifier(n_estimators=150, random_state=42).fit(X, data['success_class'])
    return reg, clf

reg_mod, clf_mod = train_models(df)

# Helper: chart update
def apply_dark(fig, height=380):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        template='plotly_dark',
        font=dict(family='Rajdhani', color='#e0e8ff', size=12),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.1)')
    return fig

# ═══════════════════════════════════════════════════════
# SECTION 0: DASHBOARD — KPI + Hero
# ═══════════════════════════════════════════════════════
st.markdown('<span id="dashboard" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<div class='section-title cyan'>🛰️ MOVIE MATRIX AI PRO — DASHBOARD</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;opacity:0.5;font-size:13px;letter-spacing:2px;'>TMDB 5000 FİLM VERİSETİ • MAKİNE ÖĞRENMESİ + CANLI API ANALİZİ</p>", unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

total_films   = len(df)
avg_roi       = df['roi'].mean()
avg_budget    = df['budget'].mean() / 1e6
avg_revenue   = df['revenue'].mean() / 1e6
hit_rate      = (df['success_class'] == 2).mean() * 100
total_revenue = df['revenue'].sum() / 1e9
unique_dir    = df['director'].nunique()
avg_score     = df['vote_average'].mean()
top_genre     = df.explode('genres_list').groupby('genres_list')['revenue'].sum().idxmax()
max_revenue   = df['revenue'].max() / 1e6

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-box kpi-cyan">
    <div class="kpi-val" style="color:#00f3ff">{total_films:,}</div>
    <div class="kpi-lbl">FİLM ANALİZİ</div>
  </div>
  <div class="kpi-box kpi-orange2">
    <div class="kpi-val" style="color:#ff8800">${avg_budget:.0f}M</div>
    <div class="kpi-lbl">ORT. BÜTÇE</div>
  </div>
  <div class="kpi-box kpi-gold">
    <div class="kpi-val" style="color:#ffd700">${avg_revenue:.0f}M</div>
    <div class="kpi-lbl">ORT. GİŞE</div>
  </div>
  <div class="kpi-box kpi-green">
    <div class="kpi-val" style="color:#00ff88">%{avg_roi*100:.0f}</div>
    <div class="kpi-lbl">ORT. ROI</div>
  </div>
  <div class="kpi-box kpi-orange">
    <div class="kpi-val" style="color:#ff6b35">%{hit_rate:.0f}</div>
    <div class="kpi-lbl">HİT ORANI</div>
  </div>
  <div class="kpi-box kpi-cyan">
    <div class="kpi-val" style="color:#00f3ff">${total_revenue:.1f}B</div>
    <div class="kpi-lbl">TOPLAM GİŞE</div>
  </div>
  <div class="kpi-box kpi-orange2">
    <div class="kpi-val" style="color:#ff8800">{unique_dir}</div>
    <div class="kpi-lbl">YÖNETMEN</div>
  </div>
  <div class="kpi-box kpi-gold">
    <div class="kpi-val" style="color:#ffd700">{avg_score:.1f}</div>
    <div class="kpi-lbl">ORT. PUAN</div>
  </div>
  <div class="kpi-box kpi-green">
    <div class="kpi-val" style="color:#00ff88" title="{top_genre}">{top_genre[:8]}</div>
    <div class="kpi-lbl">EN KAZANLI TÜR</div>
  </div>
  <div class="kpi-box kpi-orange">
    <div class="kpi-val" style="color:#ff6b35">${max_revenue:.0f}M</div>
    <div class="kpi-lbl">EN YÜKSEK GİŞE</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 1: EDA — TÜM ETKİ GRAFİKLERİ
# ═══════════════════════════════════════════════════════
st.markdown('<span id="eda" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<div class='section-title cyan'>📈 TÜM ETKENLERİN GİŞEYE ETKİSİ</div>", unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# Row 1: Bütçe vs Gelir + ROI Dağılımı
r1c1, r1c2 = st.columns([2, 1])
with r1c1:
    st.markdown("<div class='sub-header'>💰 BÜTÇE → GİŞE İLİŞKİSİ (Boyut=Popülarite | Renk=Puan)</div>", unsafe_allow_html=True)
    fig = px.scatter(df, x='budget', y='revenue', size='popularity', color='vote_average',
                     hover_name='title', color_continuous_scale='Plasma',
                     labels={'budget':'Bütçe ($)','revenue':'Gişe ($)','vote_average':'Puan'})
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0)))
    apply_dark(fig, 400)
    st.plotly_chart(fig, use_container_width=True)

with r1c2:
    st.markdown("<div class='sub-header'>📊 GİŞE DAĞILIMI (Log Ölçek)</div>", unsafe_allow_html=True)
    fig2 = px.histogram(df, x='revenue', nbins=50, color_discrete_sequence=['#00f3ff'],
                        log_x=True, labels={'revenue':'Gişe ($)'})
    fig2.update_traces(marker_line_width=0, opacity=0.8)
    apply_dark(fig2, 400)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# Row 2: Tür bazlı + Oy / Popülarite
r2c1, r2c2, r2c3 = st.columns(3)
with r2c1:
    st.markdown("<div class='sub-header'>🎭 TÜR BAZLI ORTALAMA GİŞE</div>", unsafe_allow_html=True)
    genre_rev = df.explode('genres_list').groupby('genres_list')['revenue'].mean().sort_values(ascending=True).tail(15)
    fig3 = px.bar(genre_rev, orientation='h', color=genre_rev.values,
                  color_continuous_scale='Magma', labels={'value':'Ort. Gişe ($)', 'genres_list':'Tür'})
    fig3.update_layout(showlegend=False, coloraxis_showscale=False)
    apply_dark(fig3, 420)
    st.plotly_chart(fig3, use_container_width=True)

with r2c2:
    st.markdown("<div class='sub-header'>🎭 TÜR BAZLI ORTALAMA ROI</div>", unsafe_allow_html=True)
    genre_roi = df.explode('genres_list').groupby('genres_list')['roi'].mean().sort_values(ascending=True).tail(15)
    fig4 = px.bar(genre_roi, orientation='h', color=genre_roi.values,
                  color_continuous_scale='Viridis', labels={'value':'Ort. ROI', 'genres_list':'Tür'})
    fig4.update_layout(showlegend=False, coloraxis_showscale=False)
    apply_dark(fig4, 420)
    st.plotly_chart(fig4, use_container_width=True)

with r2c3:
    st.markdown("<div class='sub-header'>⭐ PUAN → GİŞE KORELASYONU</div>", unsafe_allow_html=True)
    fig5 = px.scatter(df, x='vote_average', y='revenue', color='success_class',
                      color_continuous_scale='RdYlGn', size='vote_count',
                      labels={'vote_average':'Ort. Puan','revenue':'Gişe ($)'})
    fig5.update_traces(marker=dict(opacity=0.6, line=dict(width=0)))
    apply_dark(fig5, 420)
    st.plotly_chart(fig5, use_container_width=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# Row 3: Popülarite + Runtime + Dekad
r3c1, r3c2, r3c3 = st.columns(3)
with r3c1:
    st.markdown("<div class='sub-header'>🔥 POPÜLARİTE → GİŞE ETKİSİ</div>", unsafe_allow_html=True)
    fig6 = px.scatter(df[df['popularity'] < 500], x='popularity', y='revenue',
                      trendline='ols', color_discrete_sequence=['#ff8800'],
                      labels={'popularity':'Popülarite','revenue':'Gişe ($)'})
    apply_dark(fig6, 380)
    st.plotly_chart(fig6, use_container_width=True)

with r3c2:
    st.markdown("<div class='sub-header'>⏱️ FİLM SÜRESİ → GİŞE ETKİSİ</div>", unsafe_allow_html=True)
    df_rt = df[(df['runtime'] > 60) & (df['runtime'] < 220)].copy()
    df_rt['runtime_bin'] = pd.cut(df_rt['runtime'], bins=10)
    rt_data = df_rt.groupby('runtime_bin')['revenue'].mean().reset_index()
    rt_data['runtime_bin'] = rt_data['runtime_bin'].astype(str)
    fig7 = px.bar(rt_data, x='runtime_bin', y='revenue', color='revenue',
                  color_continuous_scale='Turbo', labels={'runtime_bin':'Süre (dk)','revenue':'Ort. Gişe ($)'})
    fig7.update_layout(showlegend=False, coloraxis_showscale=False, xaxis_tickangle=-45)
    apply_dark(fig7, 380)
    st.plotly_chart(fig7, use_container_width=True)

with r3c3:
    st.markdown("<div class='sub-header'>📅 DEKAD BAZLI TOPLAM GİŞE</div>", unsafe_allow_html=True)
    decade_data = df[df['decade'] != 'NaTs'].groupby('decade')['revenue'].sum().reset_index()
    fig8 = px.bar(decade_data, x='decade', y='revenue', color='revenue',
                  color_continuous_scale='Plasma', labels={'decade':'Dönem','revenue':'Toplam Gişe ($)'})
    fig8.update_layout(showlegend=False, coloraxis_showscale=False)
    apply_dark(fig8, 380)
    st.plotly_chart(fig8, use_container_width=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# Row 4: Korelasyon matrisi + Başarı Sınıfı dağılımı + Oy sayısı etkisi
r4c1, r4c2, r4c3 = st.columns(3)

with r4c1:
    st.markdown("<div class='sub-header'>🔗 KORELASYON MATRİSİ</div>", unsafe_allow_html=True)
    corr_cols = ['budget','revenue','popularity','vote_average','vote_count','runtime','roi']
    corr = df[corr_cols].corr()
    fig9 = px.imshow(corr, color_continuous_scale='RdBu_r', aspect='auto',
                     text_auto='.2f', labels=dict(color='Korelasyon'))
    apply_dark(fig9, 400)
    st.plotly_chart(fig9, use_container_width=True)

with r4c2:
    st.markdown("<div class='sub-header'>🏅 BAŞARI SINIFI DAĞILIMI</div>", unsafe_allow_html=True)
    success_map = {0: 'Başarısız (ROI<1.5)', 1: 'Orta (ROI 1.5-3)', 2: 'Hit (ROI>3)'}
    sc_data = df['success_class'].map(success_map).value_counts().reset_index()
    sc_data.columns = ['Sınıf', 'Sayı']
    fig10 = px.pie(sc_data, names='Sınıf', values='Sayı',
                   color_discrete_sequence=['#ff4444','#ffd700','#00ff88'],
                   hole=0.45)
    fig10.update_traces(textinfo='percent+label', textfont_size=12)
    apply_dark(fig10, 400)
    st.plotly_chart(fig10, use_container_width=True)

with r4c3:
    st.markdown("<div class='sub-header'>🗳️ OY SAYISI → GİŞE ETKİSİ</div>", unsafe_allow_html=True)
    fig11 = px.scatter(df, x='vote_count', y='revenue', color='vote_average',
                       color_continuous_scale='Inferno', size='budget',
                       hover_name='title',
                       labels={'vote_count':'Oy Sayısı','revenue':'Gişe ($)'})
    fig11.update_traces(marker=dict(opacity=0.6, line=dict(width=0)))
    apply_dark(fig11, 400)
    st.plotly_chart(fig11, use_container_width=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# Row 5: Box plot tür/gelir + Violin + ROI heatmap
r5c1, r5c2 = st.columns([2,1])
with r5c1:
    st.markdown("<div class='sub-header'>📦 TÜR BAZLI GİŞE DAĞILIMI (Box Plot)</div>", unsafe_allow_html=True)
    df_exp = df.explode('genres_list')
    top_genres = df_exp['genres_list'].value_counts().head(10).index
    df_box = df_exp[df_exp['genres_list'].isin(top_genres)]
    fig12 = px.box(df_box, x='genres_list', y='revenue', color='genres_list',
                   color_discrete_sequence=px.colors.qualitative.Dark24,
                   labels={'genres_list':'Tür','revenue':'Gişe ($)'})
    fig12.update_layout(showlegend=False)
    apply_dark(fig12, 400)
    st.plotly_chart(fig12, use_container_width=True)

with r5c2:
    st.markdown("<div class='sub-header'>🎯 BÜTÇE-PUAN ROI ISISI</div>", unsafe_allow_html=True)
    df['budget_bin'] = pd.cut(df['budget'], bins=5, labels=['Çok Düşük','Düşük','Orta','Yüksek','Blok Buster'])
    df['score_bin']  = pd.cut(df['vote_average'], bins=5, labels=['1-2','3-4','5-6','7-8','9-10'])
    heat_data = df.groupby(['budget_bin','score_bin'])['roi'].mean().reset_index()
    heat_pivot = heat_data.pivot(index='budget_bin', columns='score_bin', values='roi')
    fig13 = px.imshow(heat_pivot, color_continuous_scale='Hot', aspect='auto',
                      text_auto='.1f', labels=dict(color='ROI'))
    apply_dark(fig13, 400)
    st.plotly_chart(fig13, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 2: TOP 10 LİSTELERİ
# ═══════════════════════════════════════════════════════
st.markdown('<span id="top10" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<div class='section-title gold'>🏆 TOP 10 LİSTELERİ</div>", unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

top10_tab1, top10_tab2, top10_tab3, top10_tab4 = st.tabs([
    "💰 En Yüksek Gişe", "📈 En Yüksek ROI", "⭐ En Yüksek Puan", "💵 En Kârlı Filmler"
])

def render_top10(data, val_col, val_format, label, color):
    colors = ['#ffd700','#c0c0c0','#cd7f32'] + ['#00f3ff'] * 7
    html = ""
    queries = {row['title']: row['title'] for _, row in data.iterrows()}
    img_dict = prefetch_images(queries, "movie")
    for rank, (_, row) in enumerate(data.iterrows(), 1):
        val = row[val_col]
        if '%' in val_format:
            val_str = val_format % (val * 100) if val < 100 else val_format % val
        else:
            val_str = val_format % val
        img_url = img_dict.get(row['title'], '')
        html += f"""
        <div class="top10-row tooltip-container" style="border-color:{colors[rank-1]}">
          <div class="tooltip-content">
            <img src="{img_url}" width="100%">
            <div class="tooltip-title">{row['title']}</div>
            <div class="tooltip-sub">🎬 {row.get('director','—')}</div>
          </div>
          <div class="top10-rank" style="color:{colors[rank-1]}">#{rank}</div>
          <img src="{img_url}" class="avatar-img">
          <div class="top10-info">
            <div class="top10-title">{row['title']}</div>
            <div class="top10-sub">{row.get('director','—')} • {str(row.get('release_date',''))[:4]}</div>
          </div>
          <div class="top10-val" style="color:{color}">{val_str}</div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)

with top10_tab1:
    top_rev = df.nlargest(10, 'revenue')[['title','revenue','director','release_date']]
    render_top10(top_rev, 'revenue', '$%.0fM', 'Gişe', '#00f3ff')
    # make revenue in millions for display
    top_rev_disp = df.nlargest(10, 'revenue').copy()
    top_rev_disp['revenue_m'] = top_rev_disp['revenue'] / 1e6
    fig_t1 = px.bar(top_rev_disp, x='revenue_m', y='title', orientation='h',
                    color='revenue_m', color_continuous_scale='Plasma',
                    labels={'revenue_m':'Gişe ($M)','title':'Film'})
    fig_t1.update_layout(showlegend=False, coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
    apply_dark(fig_t1, 350)
    st.plotly_chart(fig_t1, use_container_width=True)

with top10_tab2:
    top_roi = df[df['roi'] < 200].nlargest(10, 'roi')[['title','roi','director','release_date','budget']]
    for _, row in top_roi.iterrows():
        pass  # just use render_top10
    html_roi = ""
    queries = {row['title']: row['title'] for _, row in top_roi.iterrows()}
    img_dict = prefetch_images(queries, "movie")
    colors2 = ['#ffd700','#c0c0c0','#cd7f32'] + ['#00ff88'] * 7
    for rank, (_, row) in enumerate(top_roi.iterrows(), 1):
        img_url = img_dict.get(row['title'], '')
        html_roi += f"""
        <div class="top10-row tooltip-container" style="border-color:{colors2[rank-1]}">
          <div class="tooltip-content">
            <img src="{img_url}" width="100%">
            <div class="tooltip-title">{row['title']}</div>
            <div class="tooltip-sub">🎬 {row.get('director','—')}</div>
          </div>
          <div class="top10-rank" style="color:{colors2[rank-1]}">#{rank}</div>
          <img src="{img_url}" class="avatar-img">
          <div class="top10-info">
            <div class="top10-title">{row['title']}</div>
            <div class="top10-sub">{row.get('director','—')} • Bütçe: ${row['budget']/1e6:.1f}M</div>
          </div>
          <div class="top10-val" style="color:#00ff88">x{row['roi']:.1f}</div>
        </div>"""
    st.markdown(html_roi, unsafe_allow_html=True)
    top_roi2 = top_roi.copy(); top_roi2['roi_x'] = top_roi2['roi']
    fig_t2 = px.bar(top_roi2, x='roi_x', y='title', orientation='h',
                    color='roi_x', color_continuous_scale='Viridis',
                    labels={'roi_x':'ROI Çarpanı','title':'Film'})
    fig_t2.update_layout(showlegend=False, coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
    apply_dark(fig_t2, 350)
    st.plotly_chart(fig_t2, use_container_width=True)

with top10_tab3:
    top_score = df[df['vote_count'] > 500].nlargest(10, 'vote_average')[['title','vote_average','director','release_date','vote_count']]
    html_sc = ""
    queries = {row['title']: row['title'] for _, row in top_score.iterrows()}
    img_dict = prefetch_images(queries, "movie")
    for rank, (_, row) in enumerate(top_score.iterrows(), 1):
        c = colors2[rank-1]
        img_url = img_dict.get(row['title'], '')
        html_sc += f"""
        <div class="top10-row tooltip-container" style="border-color:{c}">
          <div class="tooltip-content">
            <img src="{img_url}" width="100%">
            <div class="tooltip-title">{row['title']}</div>
            <div class="tooltip-sub">🎬 {row.get('director','—')}</div>
          </div>
          <div class="top10-rank" style="color:{c}">#{rank}</div>
          <img src="{img_url}" class="avatar-img">
          <div class="top10-info">
            <div class="top10-title">{row['title']}</div>
            <div class="top10-sub">{row.get('director','—')} • {row['vote_count']:,.0f} oy</div>
          </div>
          <div class="top10-val" style="color:#ffd700">⭐ {row['vote_average']:.1f}</div>
        </div>"""
    st.markdown(html_sc, unsafe_allow_html=True)
    fig_t3 = px.bar(top_score, x='vote_average', y='title', orientation='h',
                    color='vote_average', color_continuous_scale='Sunsetdark',
                    labels={'vote_average':'Puan','title':'Film'})
    fig_t3.update_layout(showlegend=False, coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
    apply_dark(fig_t3, 350)
    st.plotly_chart(fig_t3, use_container_width=True)

with top10_tab4:
    top_profit = df.nlargest(10, 'profit')[['title','profit','budget','revenue','director','release_date']]
    html_pr = ""
    queries = {row['title']: row['title'] for _, row in top_profit.iterrows()}
    img_dict = prefetch_images(queries, "movie")
    for rank, (_, row) in enumerate(top_profit.iterrows(), 1):
        c = colors2[rank-1]
        img_url = img_dict.get(row['title'], '')
        html_pr += f"""
        <div class="top10-row tooltip-container" style="border-color:{c}">
          <div class="tooltip-content">
            <img src="{img_url}" width="100%">
            <div class="tooltip-title">{row['title']}</div>
            <div class="tooltip-sub">🎬 {row.get('director','—')}</div>
          </div>
          <div class="top10-rank" style="color:{c}">#{rank}</div>
          <img src="{img_url}" class="avatar-img">
          <div class="top10-info">
            <div class="top10-title">{row['title']}</div>
            <div class="top10-sub">{row.get('director','—')} • Bütçe: ${row['budget']/1e6:.0f}M</div>
          </div>
          <div class="top10-val" style="color:#ff6b35">${row['profit']/1e6:.0f}M</div>
        </div>"""
    st.markdown(html_pr, unsafe_allow_html=True)
    top_profit2 = top_profit.copy(); top_profit2['profit_m'] = top_profit2['profit']/1e6
    fig_t4 = px.bar(top_profit2, x='profit_m', y='title', orientation='h',
                    color='profit_m', color_continuous_scale='Hot',
                    labels={'profit_m':'Kâr ($M)','title':'Film'})
    fig_t4.update_layout(showlegend=False, coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'})
    apply_dark(fig_t4, 350)
    st.plotly_chart(fig_t4, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 3: EN İYİ YÖNETMENLER & OYUNCULAR
# ═══════════════════════════════════════════════════════
st.markdown('<span id="people" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<div class='section-title orange'>🌟 EN İYİ YÖNETMENLER & OYUNCULAR</div>", unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

people_tab1, people_tab2 = st.tabs(["🎬 En İyi Yönetmenler", "🎭 En İyi Oyuncular"])

with people_tab1:
    dir_metric = st.selectbox("Sıralama Kriteri:", 
                               ['Toplam Gişe','Ortalama ROI','Film Sayısı','Ortalama Puan'],
                               key='dir_metric')
    
    dir_df = df[df['director'] != 'Unknown'].groupby('director').agg(
        film_count   = ('title','count'),
        total_rev    = ('revenue','sum'),
        avg_roi      = ('roi','mean'),
        avg_score    = ('vote_average','mean'),
        avg_budget   = ('budget','mean')
    ).reset_index()
    dir_df = dir_df[dir_df['film_count'] >= 2]
    
    metric_map = {
        'Toplam Gişe': 'total_rev',
        'Ortalama ROI': 'avg_roi',
        'Film Sayısı': 'film_count',
        'Ortalama Puan': 'avg_score'
    }
    sort_col = metric_map[dir_metric]
    top_dirs = dir_df.nlargest(20, sort_col)
    
    # Chart
    pc1, pc2 = st.columns([2, 1])
    with pc1:
        if dir_metric == 'Toplam Gişe':
            top_dirs['disp'] = top_dirs['total_rev'] / 1e6
            ylabel = 'Toplam Gişe ($M)'
        elif dir_metric == 'Ortalama ROI':
            top_dirs['disp'] = top_dirs['avg_roi']
            ylabel = 'Ort. ROI'
        elif dir_metric == 'Film Sayısı':
            top_dirs['disp'] = top_dirs['film_count']
            ylabel = 'Film Sayısı'
        else:
            top_dirs['disp'] = top_dirs['avg_score']
            ylabel = 'Ort. Puan'
        
        fig_dir = px.bar(top_dirs.sort_values('disp', ascending=True), 
                         x='disp', y='director', orientation='h',
                         color='disp', color_continuous_scale='Magma',
                         labels={'disp': ylabel, 'director': 'Yönetmen'})
        fig_dir.update_layout(showlegend=False, coloraxis_showscale=False)
        apply_dark(fig_dir, 500)
        st.plotly_chart(fig_dir, use_container_width=True)

    with pc2:
        st.markdown("<div class='sub-header'>🏅 TOP 10 YÖNETMEN KARTI</div>", unsafe_allow_html=True)
        top10_dirs = dir_df.nlargest(10, sort_col)
        html_dirs = ""
        queries = {row['director']: row['director'] for _, row in top10_dirs.iterrows()}
        img_dict = prefetch_images(queries, "person")
        for rank, (_, row) in enumerate(top10_dirs.iterrows(), 1):
            c = ['#ffd700','#c0c0c0','#cd7f32'] + ['#ff8800']*7
            if dir_metric == 'Toplam Gişe':
                val_str = f"${row['total_rev']/1e9:.2f}B"
            elif dir_metric == 'Ortalama ROI':
                val_str = f"x{row['avg_roi']:.1f}"
            elif dir_metric == 'Film Sayısı':
                val_str = f"{row['film_count']} film"
            else:
                val_str = f"⭐{row['avg_score']:.2f}"
            img_url = img_dict.get(row['director'], '')
            html_dirs += f"""
            <div class="top10-row tooltip-container" style="border-color:{c[rank-1]}">
              <div class="tooltip-content">
                <img src="{img_url}" width="100%">
                <div class="tooltip-title">{row['director']}</div>
                <div class="tooltip-sub">🎬 Yönetmen</div>
              </div>
              <div class="top10-rank" style="color:{c[rank-1]};font-size:14px">#{rank}</div>
              <img src="{img_url}" class="avatar-img">
              <div class="top10-info">
                <div class="top10-title" style="font-size:12px">{row['director']}</div>
                <div class="top10-sub">{row['film_count']} film • ⭐{row['avg_score']:.1f}</div>
              </div>
              <div class="top10-val" style="color:#ff8800;font-size:12px">{val_str}</div>
            </div>"""
        st.markdown(html_dirs, unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    
    # Scatter: Yönetmen Bütçe vs Gişe
    st.markdown("<div class='sub-header'>💎 YÖNETMEN BÜTÇE vs GİŞE DAĞILIMI (Min 3 Film)</div>", unsafe_allow_html=True)
    dir_scatter = dir_df[dir_df['film_count'] >= 3].copy()
    dir_scatter['total_rev_m'] = dir_scatter['total_rev'] / 1e6
    dir_scatter['avg_budget_m'] = dir_scatter['avg_budget'] / 1e6
    fig_ds = px.scatter(dir_scatter, x='avg_budget_m', y='total_rev_m', size='film_count',
                        color='avg_roi', color_continuous_scale='Turbo',
                        hover_name='director', size_max=40,
                        labels={'avg_budget_m':'Ort. Bütçe ($M)','total_rev_m':'Toplam Gişe ($M)','avg_roi':'Ort. ROI'})
    fig_ds.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.2)')))
    apply_dark(fig_ds, 450)
    st.plotly_chart(fig_ds, use_container_width=True)

with people_tab2:
    act_metric = st.selectbox("Sıralama Kriteri:", 
                               ['Toplam Gişe','Film Sayısı','Ortalama Puan'],
                               key='act_metric')
    
    act_df = df.explode('cast_list').groupby('cast_list').agg(
        film_count = ('title','count'),
        total_rev  = ('revenue','sum'),
        avg_score  = ('vote_average','mean'),
        avg_roi    = ('roi','mean')
    ).reset_index()
    act_df = act_df[act_df['film_count'] >= 3].rename(columns={'cast_list': 'actor'})
    
    act_metric_map = {'Toplam Gişe':'total_rev','Film Sayısı':'film_count','Ortalama Puan':'avg_score'}
    act_sort = act_metric_map[act_metric]
    top_acts = act_df.nlargest(20, act_sort)
    
    ac1, ac2 = st.columns([2, 1])
    with ac1:
        if act_metric == 'Toplam Gişe':
            top_acts['disp'] = top_acts['total_rev'] / 1e6
            ylabel2 = 'Toplam Gişe ($M)'
        elif act_metric == 'Film Sayısı':
            top_acts['disp'] = top_acts['film_count']
            ylabel2 = 'Film Sayısı'
        else:
            top_acts['disp'] = top_acts['avg_score']
            ylabel2 = 'Ort. Puan'
        
        fig_act = px.bar(top_acts.sort_values('disp', ascending=True),
                         x='disp', y='actor', orientation='h',
                         color='disp', color_continuous_scale='Plasma',
                         labels={'disp': ylabel2, 'actor': 'Oyuncu'})
        fig_act.update_layout(showlegend=False, coloraxis_showscale=False)
        apply_dark(fig_act, 500)
        st.plotly_chart(fig_act, use_container_width=True)

    with ac2:
        st.markdown("<div class='sub-header'>🏅 TOP 10 OYUNCU KARTI</div>", unsafe_allow_html=True)
        top10_acts = act_df.nlargest(10, act_sort)
        html_acts = ""
        queries = {row['actor']: row['actor'] for _, row in top10_acts.iterrows()}
        img_dict = prefetch_images(queries, "person")
        for rank, (_, row) in enumerate(top10_acts.iterrows(), 1):
            c = ['#ffd700','#c0c0c0','#cd7f32'] + ['#00f3ff']*7
            if act_metric == 'Toplam Gişe':
                val_str = f"${row['total_rev']/1e9:.2f}B"
            elif act_metric == 'Film Sayısı':
                val_str = f"{row['film_count']} film"
            else:
                val_str = f"⭐{row['avg_score']:.2f}"
            img_url = img_dict.get(row['actor'], '')
            html_acts += f"""
            <div class="top10-row tooltip-container" style="border-color:{c[rank-1]}">
              <div class="tooltip-content">
                <img src="{img_url}" width="100%">
                <div class="tooltip-title">{row['actor']}</div>
                <div class="tooltip-sub">🌟 Oyuncu</div>
              </div>
              <div class="top10-rank" style="color:{c[rank-1]};font-size:14px">#{rank}</div>
              <img src="{img_url}" class="avatar-img">
              <div class="top10-info">
                <div class="top10-title" style="font-size:12px">{row['actor']}</div>
                <div class="top10-sub">{row['film_count']} film • ROI x{row['avg_roi']:.1f}</div>
              </div>
              <div class="top10-val" style="color:#00f3ff;font-size:12px">{val_str}</div>
            </div>"""
        st.markdown(html_acts, unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>💡 OYUNCU BAŞARI SINIFI DAĞILIMI (Min 5 Film)</div>", unsafe_allow_html=True)
    
    act_success = df.explode('cast_list').groupby('cast_list').agg(
        film_count  = ('title','count'),
        hit_rate    = ('success_class', lambda x: (x == 2).mean() * 100),
        avg_rev_m   = ('revenue', lambda x: x.mean()/1e6)
    ).reset_index()
    act_success = act_success[act_success['film_count'] >= 5].nlargest(30, 'hit_rate')
    
    fig_ah = px.scatter(act_success, x='avg_rev_m', y='hit_rate', size='film_count',
                        hover_name='cast_list', color='hit_rate',
                        color_continuous_scale='RdYlGn', size_max=30,
                        labels={'avg_rev_m':'Ort. Gişe ($M)','hit_rate':'Hit Oranı (%)','cast_list':'Oyuncu'})
    fig_ah.update_traces(marker=dict(opacity=0.8))
    apply_dark(fig_ah, 420)
    st.plotly_chart(fig_ah, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 4: GİŞE TAHMİNİ
# ═══════════════════════════════════════════════════════
st.markdown('<span id="prediction" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<div class='section-title orange'>🔮 GİŞE TAHMİN PROJEKSİYONU</div>", unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

pred_c1, pred_c2 = st.columns([1, 1])

with pred_c1:
    st.markdown("<div class='sub-header'>📝 FİLM PARAMETRELERİ</div>", unsafe_allow_html=True)
    u_budget  = st.number_input("💰 Yapım Bütçesi ($)", min_value=100_000, max_value=500_000_000, value=80_000_000, step=1_000_000)
    u_runtime = st.slider("⏱️ Film Süresi (Dakika)", 60, 220, 115)
    u_pop     = st.slider("🔥 Hedef Popülarite Skoru", 0, 500, 80)
    u_vote    = st.number_input("🗳️ Beklenen Oy Sayısı", min_value=100, max_value=20000, value=2000, step=100)
    
    predict_btn = st.button("🚀 ANALİZİ BAŞLAT")

with pred_c2:
    if predict_btn:
        X_input    = [[u_budget, u_runtime, u_pop, u_vote]]
        rev_pred   = reg_mod.predict(X_input)[0]
        proba      = clf_mod.predict_proba(X_input)[0]
        prob_hit   = proba[2] * 100
        prob_mid   = proba[1] * 100
        prob_fail  = proba[0] * 100
        roi_pred   = rev_pred / u_budget
        profit_pred = rev_pred - u_budget

        success_label = "🟢 HİT" if prob_hit > 50 else ("🟡 ORTA" if prob_mid > prob_fail else "🔴 RİSKLİ")
        
        st.markdown(f"""
        <div class="pred-result">
          <div style="font-size:13px;letter-spacing:2px;opacity:0.6;margin-bottom:10px">TAHMİNİ GİŞE HASILATI</div>
          <div class="pred-big" style="color:#00f3ff">${rev_pred/1e6:.1f}M</div>
          <div style="margin:16px 0;padding:12px;background:rgba(0,0,0,0.3);border-radius:10px">
            <span style="color:#00ff88;font-size:13px;font-weight:700">ROI: x{roi_pred:.2f}</span> &nbsp;|&nbsp;
            <span style="color:#ff6b35;font-size:13px;font-weight:700">KÂR: ${profit_pred/1e6:.1f}M</span>
          </div>
          <div style="font-size:24px;font-weight:900;letter-spacing:2px">{success_label}</div>
          <div style="margin-top:16px;display:grid;grid-template-columns:repeat(3,1fr);gap:8px">
            <div style="background:rgba(255,68,68,0.15);border:1px solid #ff4444;border-radius:8px;padding:10px">
              <div style="color:#ff4444;font-size:18px;font-weight:900">{prob_fail:.0f}%</div>
              <div style="font-size:10px;opacity:0.6">BAŞARISIZ</div>
            </div>
            <div style="background:rgba(255,215,0,0.15);border:1px solid #ffd700;border-radius:8px;padding:10px">
              <div style="color:#ffd700;font-size:18px;font-weight:900">{prob_mid:.0f}%</div>
              <div style="font-size:10px;opacity:0.6">ORTA</div>
            </div>
            <div style="background:rgba(0,255,136,0.15);border:1px solid #00ff88;border-radius:8px;padding:10px">
              <div style="color:#00ff88;font-size:18px;font-weight:900">{prob_hit:.0f}%</div>
              <div style="font-size:10px;opacity:0.6">HİT</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance viz
        st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>🧠 MODEL FAKTÖR AĞIRLIKLARI</div>", unsafe_allow_html=True)
        feat_names = ['Bütçe', 'Süre', 'Popülarite', 'Oy Sayısı']
        importances = reg_mod.feature_importances_
        max_imp = importances.max()
        colors_feat = ['#00f3ff','#ff8800','#ffd700','#00ff88']
        html_feat = ""
        for i, (fn, imp) in enumerate(zip(feat_names, importances)):
            pct = imp / max_imp * 100
            html_feat += f"""
            <div class="feat-bar-wrap">
              <div class="feat-bar-label">
                <span>{fn}</span><span style="color:{colors_feat[i]}">{imp:.3f}</span>
              </div>
              <div class="feat-bar-bg">
                <div class="feat-bar-fill" style="width:{pct}%;background:{colors_feat[i]}"></div>
              </div>
            </div>"""
        st.markdown(html_feat, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 5: CANLI AI ANALİZ (TMDB + AI Yorum)
# ═══════════════════════════════════════════════════════
st.markdown('<span id="live-ai" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<div class='section-title cyan'>🔍 CANLI AI ANALİZ — TMDB + GİŞE TAHMİNİ</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;opacity:0.5;font-size:12px;letter-spacing:1px'>Dataset dışındaki herhangi bir film için TMDB API + AI modeli ile anlık analiz yapın</p>", unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

c1, c2 = st.columns([3, 1])
with c1:
    search_query = st.text_input("🎬 Film Ara (Türkçe veya İngilizce):", placeholder="Örn: Inception, Interstellar, Joker...")
with c2:
    st.write("") # Boşluk bırakarak butonu hizalamak için
    st.write("")
    search_btn = st.button("🔍 AI ile Ara")

if search_btn and search_query:
    try:
        res = requests.get(
            f"{BASE_URL}/search/movie?api_key={API_KEY}&query={search_query}&language=tr-TR",
            timeout=8
        ).json()
        
        if res.get('results'):
            movie = res['results'][0]
            movie_id = movie['id']
            
            # Fetch detailed data
            detail_res = requests.get(
                f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=tr-TR&append_to_response=credits",
                timeout=8
            ).json()
            
            budget_live   = detail_res.get('budget', 0)
            revenue_live  = detail_res.get('revenue', 0)
            runtime_live  = detail_res.get('runtime', 100) or 100
            pop_live      = movie.get('popularity', 50)
            score_live    = movie.get('vote_average', 0)
            vote_cnt_live = movie.get('vote_count', 0)
            
            # Director from credits
            crew_live = detail_res.get('credits', {}).get('crew', [])
            directors_live = [p['name'] for p in crew_live if p.get('job') == 'Director']
            cast_live = [p['name'] for p in detail_res.get('credits', {}).get('cast', [])[:5]]
            genres_live = [g['name'] for g in detail_res.get('genres', [])]
            
            la, lb = st.columns([1, 2])
            with la:
                poster_path = movie.get('poster_path')
                if poster_path:
                    st.image(POSTER_URL + poster_path, use_container_width=True)
                else:
                    st.markdown("<div style='height:300px;background:rgba(255,255,255,0.05);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:48px'>🎬</div>", unsafe_allow_html=True)
            
            with lb:
                st.markdown(f"<h2 style='color:#00f3ff;font-family:Orbitron;margin:0'>{movie.get('title','')}</h2>", unsafe_allow_html=True)
                orig_title = movie.get('original_title','')
                if orig_title != movie.get('title',''):
                    st.markdown(f"<p style='opacity:0.5;font-size:12px;margin:2px 0'>{orig_title}</p>", unsafe_allow_html=True)
                
                release_date = movie.get('release_date', 'N/A')
                st.markdown(f"""
                <div style='display:flex;gap:10px;flex-wrap:wrap;margin:10px 0'>
                  <span style='background:rgba(0,243,255,0.1);border:1px solid rgba(0,243,255,0.3);padding:4px 10px;border-radius:20px;font-size:12px'>📅 {release_date}</span>
                  <span style='background:rgba(255,215,0,0.1);border:1px solid rgba(255,215,0,0.3);padding:4px 10px;border-radius:20px;font-size:12px'>⭐ {score_live:.1f}/10</span>
                  <span style='background:rgba(255,136,0,0.1);border:1px solid rgba(255,136,0,0.3);padding:4px 10px;border-radius:20px;font-size:12px'>⏱️ {runtime_live} dk</span>
                  <span style='background:rgba(0,255,136,0.1);border:1px solid rgba(0,255,136,0.3);padding:4px 10px;border-radius:20px;font-size:12px'>🔥 {pop_live:.0f} Popülarite</span>
                </div>
                """, unsafe_allow_html=True)
                
                if genres_live:
                    genres_str = " • ".join(genres_live)
                    st.markdown(f"<p style='opacity:0.6;font-size:12px'>🎭 {genres_str}</p>", unsafe_allow_html=True)
                if directors_live:
                    st.markdown(f"<p style='font-size:13px'><b>🎬 Yönetmen:</b> {', '.join(directors_live)}</p>", unsafe_allow_html=True)
                if cast_live:
                    st.markdown(f"<p style='font-size:13px'><b>🌟 Başrol:</b> {', '.join(cast_live)}</p>", unsafe_allow_html=True)
                
                overview = movie.get('overview','') or detail_res.get('overview','')
                if overview:
                    st.markdown(f"<p style='opacity:0.7;font-size:13px;line-height:1.6'>{overview[:400]}{'...' if len(overview)>400 else ''}</p>", unsafe_allow_html=True)
                
                # AI model prediction on this movie
                if budget_live > 0:
                    rev_ai   = reg_mod.predict([[budget_live, runtime_live, pop_live, vote_cnt_live]])[0]
                    proba_ai = clf_mod.predict_proba([[budget_live, runtime_live, pop_live, vote_cnt_live]])[0]
                    roi_ai   = rev_ai / budget_live
                    
                    success_ai = "🟢 HİT" if proba_ai[2] > 0.5 else ("🟡 ORTA" if proba_ai[1] > proba_ai[0] else "🔴 RİSKLİ")
                    
                    st.markdown(f"""
                    <div style='background:rgba(0,243,255,0.07);border:1px solid rgba(0,243,255,0.3);border-radius:12px;padding:16px;margin-top:12px'>
                      <div style='font-size:11px;letter-spacing:2px;opacity:0.6;margin-bottom:8px'>🤖 AI MODEL TAHMİNİ</div>
                      <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px'>
                        <div style='text-align:center'>
                          <div style='color:#00f3ff;font-size:16px;font-weight:900'>${rev_ai/1e6:.0f}M</div>
                          <div style='font-size:10px;opacity:0.5'>TAHMİNİ GİŞE</div>
                        </div>
                        <div style='text-align:center'>
                          <div style='color:#ff8800;font-size:16px;font-weight:900'>x{roi_ai:.1f}</div>
                          <div style='font-size:10px;opacity:0.5'>TAHMİNİ ROI</div>
                        </div>
                        <div style='text-align:center'>
                          <div style='color:#00ff88;font-size:16px;font-weight:900'>{proba_ai[2]*100:.0f}%</div>
                          <div style='font-size:10px;opacity:0.5'>HİT ŞANSI</div>
                        </div>
                        <div style='text-align:center'>
                          <div style='font-size:16px;font-weight:900'>{success_ai}</div>
                          <div style='font-size:10px;opacity:0.5'>SONUÇ</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if revenue_live > 0:
                        real_roi = revenue_live / budget_live
                        st.markdown(f"""
                        <div style='background:rgba(255,215,0,0.07);border:1px solid rgba(255,215,0,0.3);border-radius:12px;padding:12px;margin-top:8px'>
                          <div style='font-size:11px;letter-spacing:2px;opacity:0.6;margin-bottom:6px'>✅ GERÇEK VERİ (TMDB)</div>
                          <span style='color:#ffd700;font-size:14px;font-weight:700'>${revenue_live/1e6:.0f}M Gerçek Gişe</span> &nbsp;|&nbsp;
                          <span style='color:#ffd700;font-size:14px;font-weight:700'>x{real_roi:.1f} Gerçek ROI</span> &nbsp;|&nbsp;
                          <span style='color:#ffd700;font-size:14px;font-weight:700'>${budget_live/1e6:.0f}M Bütçe</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Bu film için bütçe verisi TMDB'de mevcut değil. Tahmin bütçe bilgisi gerektirir.")
        else:
            st.warning("Film bulunamadı. Farklı bir arama deneyin.")
    except Exception as e:
        st.error(f"API bağlantı hatası: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SECTION 6: BU AKŞAM NE İZLESEM?
# ═══════════════════════════════════════════════════════
st.markdown('<span id="mood-watch" class="section-anchor"></span>', unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("<div class='section-title gold'>🍿 AI FİLM TERAPİSTİ — BU AKŞAM NE İZLESEM?</div>", unsafe_allow_html=True)
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

mw_c1, mw_c2 = st.columns([1, 2])
with mw_c1:
    st.markdown("<div class='sub-header'>🎛️ FİLTRELER</div>", unsafe_allow_html=True)
    mood_genre = st.selectbox("🎭 Tür:", ['Hepsi'] + sorted(df.explode('genres_list')['genres_list'].dropna().unique().tolist()))
    mood_min_score = st.slider("⭐ Minimum Puan:", 0.0, 10.0, 7.0, 0.1)
    mood_decade    = st.selectbox("📅 Dönem:", ['Hepsi'] + sorted(df['decade'].dropna().unique().tolist()))
    mood_success   = st.selectbox("🏆 Başarı Filtresi:", ['Hepsi', 'Hit Filmler (ROI>3)', 'Orta Bütçe Hikayeleri'])
    mood_sort      = st.selectbox("🔢 Sırala:", ['Popülarite','Puan','Gişe','ROI'])

with mw_c2:
    st.markdown("<div class='sub-header'>💭 RUH HALİNİ ANLAT (Opsiyonel)</div>", unsafe_allow_html=True)
    user_mood = st.text_area("", placeholder="Örn: Aksiyon dolu, sürükleyici bir şey istiyorum... veya Duygusal, derin bir drama...", height=100)

if st.button("✨ SİHİRLİ ÖNERME BAŞLAT"):
    filtered = df.copy()
    
    if mood_genre != 'Hepsi':
        filtered = filtered[filtered['genres_list'].apply(lambda x: mood_genre in x)]
    if mood_decade != 'Hepsi':
        filtered = filtered[filtered['decade'] == mood_decade]
    if mood_success == 'Hit Filmler (ROI>3)':
        filtered = filtered[filtered['success_class'] == 2]
    elif mood_success == 'Orta Bütçe Hikayeleri':
        filtered = filtered[(filtered['budget'] < 30_000_000) & (filtered['vote_average'] >= mood_min_score)]
    
    filtered = filtered[filtered['vote_average'] >= mood_min_score]
    
    sort_col_mood = {'Popülarite':'popularity','Puan':'vote_average','Gişe':'revenue','ROI':'roi'}[mood_sort]
    filtered = filtered.nlargest(100, sort_col_mood)
    
    if len(filtered) == 0:
        st.warning("Bu filtrelere uygun film bulunamadı. Filtreleri gevşetin.")
    else:
        recs = filtered.sample(min(8, len(filtered)))
        st.markdown("<div class='sub-header' style='margin-top:20px'>🎬 SENIN İÇİN AI SEÇİMLERİ</div>", unsafe_allow_html=True)
        
        cols_recs = st.columns(4)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols_recs[i % 4]:
                try:
                    api_r = requests.get(
                        f"{BASE_URL}/search/movie?api_key={API_KEY}&query={row['title']}&language=tr-TR",
                        timeout=5
                    ).json()
                    img_url = ""
                    if api_r.get('results') and api_r['results'][0].get('poster_path'):
                        img_url = POSTER_URL + api_r['results'][0]['poster_path']
                except:
                    img_url = ""
                
                genres_str = ", ".join(row['genres_list'][:2]) if row['genres_list'] else ""
                
                if img_url:
                    st.image(img_url, use_container_width=True)
                else:
                    st.markdown("<div style='height:200px;background:rgba(255,255,255,0.05);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:36px'>🎬</div>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='text-align:center;padding:8px 0'>
                  <div style='font-weight:700;font-size:13px;line-height:1.3'>{row['title']}</div>
                  <div style='opacity:0.5;font-size:11px'>{genres_str}</div>
                  <div style='margin-top:6px'>
                    <span style='color:#ffd700;font-size:13px'>⭐ {row['vote_average']:.1f}</span>
                    <span style='margin:0 6px;opacity:0.3'>|</span>
                    <span style='color:#00f3ff;font-size:11px'>${row['revenue']/1e6:.0f}M</span>
                  </div>
                  <div style='color:#ff8800;font-size:11px'>ROI x{row['roi']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center;padding:30px;opacity:0.3;font-family:Orbitron,sans-serif;font-size:10px;letter-spacing:3px'>
MOVIE MATRIX AI PRO v4.0 • TMDB 5000 DATASET • ML POWERED • © 2025
</div>
""", unsafe_allow_html=True)