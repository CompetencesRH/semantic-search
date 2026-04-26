# Recherche Sémantique de CVs RH

Application Streamlit de matching de CVs par recherche sémantique — trouvez les profils les plus proches d'une description de poste, sans mots-clés exacts.

---

## Démonstration

1. Décrivez le profil recherché en langage naturel
2. L'application encode votre requête et la compare à tous les CVs
3. Les profils les plus proches sont retournés avec un score de similarité
4. Une carte sémantique PCA visualise la distribution des profils

---

## Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| Modèle d'embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Réduction dimensionnelle | PCA (scikit-learn) |
| Distance | Euclidienne (scikit-learn) |
| Visualisation | Matplotlib |
| Data | Pandas / NumPy |

---

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/CompetencesRH/semantic-search
cd semantic-search
```

### 2. Installer les dépendances

```bash
pip install streamlit pandas numpy matplotlib scikit-learn sentence-transformers
```

### 3. Préparer les données

Le fichier `resumes_train.csv` doit être à la racine du projet avec ces colonnes :

| Colonne | Description |
|---|---|
| `resume` | Texte complet du CV |
| `role` | Intitulé de poste (utilisé pour la coloration de la carte) |

---

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

---

## Fonctionnement

### Embeddings

Chaque CV est encodé en vecteur dense de 384 dimensions via `all-MiniLM-L6-v2`. La requête utilisateur est encodée de la même façon à chaque recherche.

### Similarité

La distance euclidienne est calculée entre le vecteur requête et tous les vecteurs CVs. Le score affiché est converti en similarité 0→1 :

```python
score = 1 / (1 + distance_euclidienne)
```

Un score proche de 1 indique un profil très proche de la recherche.

### Carte sémantique PCA

Une réduction PCA à 2 dimensions projette tous les CVs sur un plan. Chaque couleur correspond à un rôle. L'étoile noire représente votre requête — plus elle est proche d'un cluster, plus les profils de ce cluster correspondent.

---

## Interface

- **Champ texte** : décrivez librement le profil recherché
- **Slider** : choisissez le nombre de résultats (1 à 10)
- **Carte sémantique** : visualisation PCA colorée par rôle
- **Résultats** : top K profils avec score et texte complet en accordéon

---

## Performance

Le chargement du modèle et le calcul des embeddings sont mis en cache via `@st.cache_resource` et `@st.cache_data` — le premier lancement est plus lent, les suivants sont instantanés.

---

## Structure du projet

```
.
├── app.py                  # Application Streamlit
├── resumes_train.csv       # Dataset CVs (resume, role)
└── README.md
```

---

## Améliorations possibles

- Remplacer la distance euclidienne par la similarité cosinus (plus standard en NLP)
- Ajouter un filtre par rôle avant la recherche
- Permettre l'upload d'un CV au format PDF pour le comparer au dataset
- Passer à un modèle multilingue (`paraphrase-multilingual-MiniLM-L12-v2`) pour les CVs en français
- Indexer les embeddings avec FAISS pour scaler sur des datasets larges

---

## Licence

MIT — [CompétencesRH](https://competencesrh.fr)
