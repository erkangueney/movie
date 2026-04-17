import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 1. SAYFA AYARLARI
st.set_page_config(page_title="AI Movie Matrix Pro", layout="wide")

# 2. VERİ YÜKLEME VE ÖN İŞLEME
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    df = movies.merge(credits, left_on='id', right_on='movie_id')
    
    # JSON sütunlarını ayrıştırma
    def parse_json(x, key):
        try:
            items = json.loads(x)
            return [i[key] for i in items]
        except: return []

    df['genres'] = df['genres'].apply(lambda x: parse_json(x, 'name'))
    df['cast'] = df['cast'].apply(lambda x: parse_json(x, 'name'))
    df['crew'] = df['crew'].apply(lambda x: parse_json(x, 'name'))
    
    # ROI ve Başarı Etiketi (Target Engineering)
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    df['roi'] = df['revenue'] / df['budget']
    df['is_hit'] = (df['roi'] >= 3).astype(int) # ROI >= 3 ise başarılı (Hit)
    
    return df

df = load_data()

# 3. FEATURE ENGINEERING (Yarışma Kazandıracak Metrikler)
def prepare_features(df):
    # Türleri One-Hot Encoding yapma
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genres'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index)
    
    # Sayısal özellikler
    features = pd.concat([df[['budget', 'popularity', 'runtime']], genre_df], axis=1)
    return features, df['revenue'], df['is_hit']

# 4. MODEL EĞİTİMİ (Gradient Descent Tabanlı)
X, y_rev, y_hit = prepare_features(df)
X_train, X_test, y_train_hit, y_test_hit = train_test_split(X, y_hit, test_size=0.2, random_state=42)

# Sınıflandırma Modeli (Hit Olasılığı için)
# Bu model arka planda Gradient Descent (Gradyan İniş) kullanarak hata minimize eder
hit_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05, # Gradient Descent adım büyüklüğü
    max_depth=5,
    random_state=42
)
hit_model.fit(X_train, y_train_hit)

# 5. STREAMLIT ARAYÜZÜ
st.title("🎬 AI Film Analiz ve Gişe Tahmin Platformu")
st.markdown("---")

# Yan Panel - Filtreler ve Girişler
st.sidebar.header("🎥 Film Parametreleri")
input_budget = st.sidebar.number_input("Bütçe ($)", min_value=100000, max_value=500000000, value=50000000)
input_genres = st.sidebar.multiselect("Türler", options=X.columns[3:].tolist(), default=["Action", "Adventure"])
input_runtime = st.sidebar.slider("Süre (Dakika)", 60, 240, 120)

# 6. GİŞE OLASILIĞI TAHMİNİ (Yeni Feature)
st.header("📈 Gişe Başarı Analizi")

# Tahmin Girişi Hazırlama
input_data = pd.DataFrame(0, index=[0], columns=X.columns)
input_data['budget'] = input_budget
input_data['runtime'] = input_runtime
input_data['popularity'] = df['popularity'].mean() # Ortalama popülerlik varsayımı
for g in input_genres:
    input_data[g] = 1

# Olasılık Tahmini
prob = hit_model.predict_proba(input_data)[0]
hit_chance = prob[1] * 100

col1, col2 = st.columns(2)

with col1:
    # Gösterge (Gauge Chart)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = hit_chance,
        title = {'text': "Başarı (HIT) Olasılığı"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00f3ff"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🤖 AI Strateji Notu")
    if hit_chance > 70:
        st.success("Bu proje yüksek ticari potansiyele sahip! Yeşil ışık yakılabilir.")
    elif hit_chance > 40:
        st.warning("Riskli bölge. Oyuncu kadrosu veya pazarlama bütçesi optimize edilmeli.")
    else:
        st.error("Düşük başarı ihtimali. Senaryo veya tür kombinasyonu tekrar gözden geçirilmeli.")
    
    st.info(f"**Gradyan İniş Notu:** Bu tahmin, {hit_model.n_estimators} ağaçlı bir Gradient Boosting mimarisi tarafından, kayıp fonksiyonunu her adımda minimize ederek hesaplanmıştır.")

# 7. VERİ ANALİZİ PLATFORMU (Görselleştirme)
st.markdown("---")
st.header("📊 Pazar Analizi")
tab1, tab2 = st.tabs(["Tür Analizi", "Bütçe vs Hasılat"])

with tab1:
    genre_impact = df.explode('genres').groupby('genres')['roi'].mean().sort_values(ascending=False).reset_index()
    fig_genre = px.bar(genre_impact, x='genres', y='roi', title="Hangi Tür Daha Çok Kazandırıyor? (ROI)", color='roi')
    st.plotly_chart(fig_genre, use_container_width=True)

with tab2:
    fig_scatter = px.scatter(df, x='budget', y='revenue', color='is_hit', 
                             hover_name='title', title="Bütçe ve Hasılat İlişkisi",
                             labels={'is_hit': 'Başarılı (Hit)'})
    st.plotly_chart(fig_scatter, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Developed by Data Science Expert 🚀")