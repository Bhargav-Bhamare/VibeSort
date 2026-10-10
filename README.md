# Music Playlist Clustering (Streamlit)

A simple Streamlit web app that applies the project's existing K-Means workflow to `songs.csv`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Create a public GitHub repository and upload all files in this folder.
2. Open https://share.streamlit.io/ and sign in with GitHub.
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit will install the packages listed in `requirements.txt`.

The app keeps the original four audio features, median imputation, StandardScaler, K-Means settings (`n_init=10`, `random_state=42`), K=4 playlist assignment, and the existing playlist naming logic.
