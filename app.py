# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import DistanceMetric

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('resumes_train.csv')

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def compute_embeddings(resumes):
    model = load_model()
    return model.encode(resumes)

df = load_data()
model = load_model()
embedding_arr = compute_embeddings(df['resume'].tolist())
pca = PCA(n_components=2).fit(embedding_arr)

# ── INTERFACE ─────────────────────────────────────────────────────────────────
st.title("🔍 Recherche sémantique de CVs RH")
st.markdown("Décris le profil recherché, le moteur trouve les CVs les plus proches.")

query = st.text_input("Votre recherche", "Expert en analyse de données RH et masse salariale")
top_k = st.slider("Nombre de résultats", min_value=1, max_value=10, value=5)

if query:
    # embed query
    query_embedding = model.encode(query)

    # distances
    dist = DistanceMetric.get_metric('euclidean')
    dist_arr = dist.pairwise(embedding_arr, query_embedding.reshape(1, -1)).flatten()
    idist_sorted = np.argsort(dist_arr)

    # ── GRAPHIQUE PCA ─────────────────────────────────────────────────────────
    st.subheader("📊 Carte sémantique des CVs")
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = mpl.colormaps['jet']
    roles = df['role'].unique()

    for i, role in enumerate(roles):
        idx = np.where(df['role'] == role)
        coords = pca.transform(embedding_arr)
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=[cmap(i / len(roles))] * len(idx[0]), label=role, alpha=0.7)

    query_pca = pca.transform(query_embedding.reshape(1, -1))[0]
    ax.scatter(query_pca[0], query_pca[1], c='black', marker='*', s=500, label='Votre recherche', zorder=5)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f'"{query}"')
    ax.grid(True)
    st.pyplot(fig)

    # ── RÉSULTATS ─────────────────────────────────────────────────────────────
    st.subheader(f"🏆 Top {top_k} profils correspondants")
    for rank, idx in enumerate(idist_sorted[:top_k]):
        role = df['role'].iloc[idx]
        score = 1 / (1 + dist_arr[idx])  # similarité 0→1
        with st.expander(f"#{rank+1} — {role}  |  Score : {score:.2f}"):
            st.write(df['resume'].iloc[idx])
