import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES = ["tempo", "energy", "danceability", "acousticness"]
df = pd.read_csv("songs.csv")                                    # 1. LOAD
print("Missing values before:\n", df[FEATURES].isna().sum())

df = df.drop_duplicates(subset=["track_name", "artist"])         # 2. PREPROCESS
df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())        #    fill NaN with median
X = StandardScaler().fit_transform(df[FEATURES])                 #    feature scaling

ks = range(2, 11)                                                # 3. ELBOW METHOD
inertia = [KMeans(k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]
sil = [silhouette_score(X, KMeans(k, n_init=10, random_state=42).fit_predict(X)) for k in ks]
plt.figure(figsize=(7, 4)); plt.plot(ks, inertia, "o-")
plt.xlabel("Number of clusters (k)"); plt.ylabel("Inertia (WCSS)"); plt.title("Elbow Method")
plt.grid(alpha=.3); plt.savefig("elbow_plot.png", dpi=150, bbox_inches="tight")

K = 4                                                            # chosen from elbow + silhouette
df["cluster"] = KMeans(K, n_init=10, random_state=42).fit_predict(X)   # 4. K-MEANS

c = df.groupby("cluster")[FEATURES].mean()                       # 5. NAME THE PLAYLISTS
names, left = {}, set(c.index)
for label, score in [("High-Energy Workout", c.energy + c.tempo / c.tempo.max()),
                     ("Acoustic Lounge", c.acousticness), ("Party / Dance", c.danceability),
                     ("Chill Study", -c.energy)]:
    best = score[list(left)].idxmax(); names[best] = label; left.discard(best)
df["playlist"] = df["cluster"].map(names)

plt.figure(figsize=(8, 6))                                       # 6. SCATTER PLOT
sns.scatterplot(data=df, x="energy", y="danceability", hue="playlist", palette="Set2", s=60)
plt.title("Song Clusters: Energy vs Danceability"); plt.savefig("cluster_scatter.png", dpi=150, bbox_inches="tight")

df.sort_values("playlist").to_csv("playlists_output.csv", index=False)
print("\nBest k by silhouette:", list(ks)[sil.index(max(sil))], "| silhouette scores:", [round(s, 2) for s in sil])
print("\nCluster profiles (mean values):\n", c.rename(index=names).round(2))
print("\nSongs per playlist:\n", df.playlist.value_counts())
for p, g in df.groupby("playlist"): print(f"\n{p}: ", ", ".join(g.track_name.head(5)), "...")
