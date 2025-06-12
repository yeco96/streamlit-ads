import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --- Dataset simplificado para demostración (usa el tuyo completo en producción) ---
data = {
    "busqueda": [
        "accesorios para mascotas",
        "bicicletas de montaña",
        "bicicleta estatica",
        "coches electricos",
        "consolas de nueva generacion",
        "documentales sobre salud",
        "juegos de estrategia",
        "decoracion de interiores"
    ],
    "tema_interes": [
        "mascotas",
        "deportes al aire libre",
        "fitness",
        "movilidad sostenible",
        "videojuegos",
        "salud",
        "entretenimiento",
        "hogar"
    ],
    "anuncio_sugerido": [
        "Arnés Julius-K9 IDC Power para perros",
        "Bicicleta Trek Marlin 7 con suspensión delantera",
        "Bicicleta Estática Schwinn IC4 con Bluetooth",
        "Tesla Model 3 Long Range AWD 2024",
        "PlayStation 5 Edición Digital",
        "Colección de documentales 'Heal' en Blu-ray",
        "Juego de mesa Catan Edición 25 Aniversario",
        "Set de cuadros minimalistas Nordic para sala"
    ]
}
df_busquedas = pd.DataFrame(data)

# --- Entrenamiento del modelo TF-IDF + KNN ---
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df_busquedas["busqueda"])

knn_model = NearestNeighbors(n_neighbors=3, metric="cosine")
knn_model.fit(X)

# --- Función mejorada de sugerencia ---
def sugerir_anuncios_tfidf(user_input):
    vec = tfidf.transform([user_input])
    _, indices = knn_model.kneighbors(vec)
    sugerencias = df_busquedas.iloc[indices[0]]["anuncio_sugerido"].unique()
    return list(sugerencias)

# --- Interfaz Streamlit ---
st.set_page_config(page_title="🔍 Anuncios Inteligentes", layout="centered")

st.title("🧠 Sugeridor de Anuncios Inteligente")
st.markdown("Introduce una búsqueda y el sistema sugerirá anuncios similares con IA (TF-IDF + KNN).")

user_input = st.text_input("¿Qué estás buscando?", placeholder="Ej: bicicleta para montaña, alimentos para gatos")

if user_input:
    st.subheader("Anuncios Sugeridos para ti:")
    sugerencias = sugerir_anuncios_tfidf(user_input)
    if sugerencias:
        for anuncio in sugerencias:
            st.success(f"👉 {anuncio}")
    else:
        st.warning("No se encontraron sugerencias para esta búsqueda.")

st.markdown("---")
st.caption("Este sistema usa técnicas de vectorización TF-IDF y búsqueda por similitud con KNN.")
