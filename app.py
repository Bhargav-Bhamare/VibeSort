import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Music Playlist Clustering", page_icon="🎵", layout="wide")
st.title("🎵 Music Playlist Clustering")
st.caption("Unsupervised machine learning with K-Means")
st.write("This app groups songs using tempo, energy, danceability and acousticness.")

FEATURES = ["tempo", "energy", "danceability", "acousticness"]
REQUIRED = ["track_name", "artist"] + FEATURES

@st.cache_data
def load_data():
    return pd.read_csv("songs.csv")

try:
    raw_df = load_data()
except Exception as exc:
    st.error(f"Could not load songs.csv: {exc}")
    st.stop()

missing_columns = [c for c in REQUIRED if c not in raw_df.columns]
if missing_columns:
    st.error("songs.csv is missing required columns: " + ", ".join(missing_columns))
    st.stop()

st.subheader("Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Rows in input dataset", f"{len(raw_df):,}")
c2.metric("Duplicate track/artist rows", int(raw_df.duplicated(subset=["track_name", "artist"]).sum()))
c3.metric("Missing feature values", int(raw_df[FEATURES].isna().sum().sum()))
st.dataframe(raw_df.head(10), use_container_width=True)

# Preserve the original project's preprocessing and clustering approach.
df = raw_df.drop_duplicates(subset=["track_name", "artist"]).copy()
df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
X = StandardScaler().fit_transform(df[FEATURES])

ks = list(range(2, 11))
inertia = []
silhouette_scores = []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

K = 4
final_model = KMeans(n_clusters=K, n_init=10, random_state=42)
df["cluster"] = final_model.fit_predict(X)

cluster_means = df.groupby("cluster")[FEATURES].mean()
names, left = {}, set(cluster_means.index)
for label, score in [
    ("High-Energy Workout", cluster_means.energy + cluster_means.tempo / cluster_means.tempo.max()),
    ("Acoustic Lounge", cluster_means.acousticness),
    ("Party / Dance", cluster_means.danceability),
    ("Chill Study", -cluster_means.energy),
]:
    best = score[list(left)].idxmax()
    names[best] = label
    left.discard(best)
df["playlist"] = df["cluster"].map(names)
df = df.sort_values("playlist").reset_index(drop=True)

st.subheader("Generated playlists")
counts = df["playlist"].value_counts()
cols = st.columns(4)
for col, playlist in zip(cols, ["High-Energy Workout", "Acoustic Lounge", "Party / Dance", "Chill Study"]):
    col.metric(playlist, int(counts.get(playlist, 0)) )
selected_playlist = st.selectbox("Choose a playlist to explore", ["All playlists"] + sorted(df["playlist"].unique().tolist()))
shown = df if selected_playlist == "All playlists" else df[df["playlist"] == selected_playlist]
st.dataframe(shown, use_container_width=True, hide_index=True)
st.download_button(
    "⬇️ Download playlist results as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="playlists_output.csv",
    mime="text/csv",
)

st.subheader("Clustering evaluation")
best_k = ks[silhouette_scores.index(max(silhouette_scores))]
m1, m2 = st.columns(2)
m1.metric("Chosen K for playlists", K)
m2.metric("Best silhouette score in K=2…10", f"K={best_k} · {max(silhouette_scores):.3f}")
st.caption("The project deliberately uses K=4 to produce four interpretable playlist types. The silhouette score is a diagnostic measure and may prefer a different K.")

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(ks, inertia, "o-")
ax1.set_xlabel("Number of clusters (K)")
ax1.set_ylabel("Inertia (WCSS)")
ax1.set_title("Elbow Method")
ax1.grid(alpha=0.3)
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="energy", y="danceability", hue="playlist", palette="Set2", s=60, ax=ax2)
ax2.set_title("Song Clusters: Energy vs Danceability")
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

st.subheader("Average audio features by playlist")
profile = df.groupby("playlist")[FEATURES].mean().round(3)
st.dataframe(profile, use_container_width=True)
st.info("This is a demonstration using the supplied sample dataset. Playlist labels are interpretations of the clusters, not verified music genres.")
