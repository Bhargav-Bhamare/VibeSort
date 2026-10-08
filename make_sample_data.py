"""Generates a sample songs.csv (use only if you don't have a Kaggle CSV).
Mimics Spotify audio features. Includes a few missing values on purpose
so the preprocessing step has something to do."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
# (tempo, energy, danceability, acousticness) means per hidden "mood"
moods = [(78, .35, .55, .45), (150, .88, .65, .08), (120, .72, .88, .20), (105, .25, .30, .93)]
rows = []
for i in range(320):
    t, e, d, a = moods[i % 4]
    rows.append([f"Song {i+1:03d}", f"Artist {rng.integers(1, 60)}",
                 rng.normal(t, 10), np.clip(rng.normal(e, .09), 0, 1),
                 np.clip(rng.normal(d, .09), 0, 1), np.clip(rng.normal(a, .10), 0, 1)])
df = pd.DataFrame(rows, columns=["track_name", "artist", "tempo", "energy", "danceability", "acousticness"])
for col in ["tempo", "energy", "danceability", "acousticness"]:      # inject ~3% missing
    df.loc[rng.choice(len(df), 10, replace=False), col] = np.nan
df.sample(frac=1, random_state=1).to_csv("songs.csv", index=False)
print("songs.csv created:", df.shape)
