import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.metrics import davies_bouldin_score
import pandas as pd
import matplotlib.pyplot as plt


# ===== Helper: Arrow-Listen -> 2D NumPy =====
def listarray_to_2d_numpy(arr: pa.Array, dtype=np.float32) -> np.ndarray:
    t = arr.type
    if pa.types.is_fixed_size_list(t):
        dim = t.list_size
        flat = arr.values
        out = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=dtype)
        return out.reshape(len(arr), dim)
    elif pa.types.is_list(t):
        py = arr.to_pylist()
        first = next((x for x in py if x is not None), None)
        if first is None:
            return np.empty((len(py), 0), dtype=dtype)
        dim = len(first)
        return np.array(py, dtype=dtype).reshape(len(py), dim)
    else:
        raise TypeError(f"column embeddings must be List/FixedSizeList: {t}")


# ===== Elbow Plot =====
def plot_elbow(X: np.ndarray, k_values, out_path: Path):

    db_scores = []

    print("Computing inertia for elbow plot...")
    for k in k_values:
        print(f"  k = {k} ...")

        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        db_index = davies_bouldin_score(X, labels)
        inertia = kmeans.inertia_
        db_scores.append(db_index)
       

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, db_scores, marker="o")
    plt.title("Elbow Plot: KMeans Davies–Bouldin Index vs. k")
    plt.xlabel("k (#clusters)")
    plt.ylabel("Davies–Bouldin Index")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved elbow plot to {out_path}")



def main():
    # === Paths ===
    PCA_PARQUET = Path("/dtu/blackhole/1a/222266/embeddings_umap15.parquet")
    OUT_CSV = Path("/dtu/blackhole/1a/222266/clustered_reviews_umap15_final.csv")
    PLOT_DIR = Path("/zhome/3d/c/222266/ComputationalTools/AmazonReview/data")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_ELBOW = PLOT_DIR / "elbow_plot_kmeans_umap15_final.png"

    # === Columns ===
    ID_COL = "row_id"
    RATING_COL = "review/score"
    SENT_COL = "sentiment_score"
    EMBEDDINGS_COL = "embedding_umap_15"

    # === k options for elbow plot ===
    K_VALUES = [2, 3, 4, 5, 6, 7, 8, 10]


    print(f"Loading PCA embeddings from {PCA_PARQUET}")
    pf = pq.ParquetFile(PCA_PARQUET)

    all_ids, all_ratings, all_sents, all_X = [], [], [], []

    # ===== Read all PCA embeddings =====
    for rb in pf.iter_batches(
        columns=[ID_COL, RATING_COL, SENT_COL, EMBEDDINGS_COL],
        batch_size=65536
    ):
        ids = rb.column(0).to_pylist()
        ratings = rb.column(1).to_pylist()
        sents = rb.column(2).to_pylist()
        emb_arr = rb.column(3)

        X_part = listarray_to_2d_numpy(emb_arr, dtype=np.float32)
        if X_part.size == 0:
            continue

        all_ids.extend(ids)
        all_ratings.extend(ratings)
        all_sents.extend(sents)
        all_X.append(X_part)

    if not all_X:
        print("No data found.")
        return

    X_pca = np.vstack(all_X)
    print("PCA embeddings shape:", X_pca.shape)

    # ===== 1) Elbow Plot =====
    plot_elbow(X_pca, K_VALUES, OUT_ELBOW)

    
    # ===== 2) Choose your preferred k after seeing elbow plot =====
    K = 7   # <--- HIER kannst du k ändern nach Elbow-Plot
    print(f"Running final KMeans with k={K} ...")

    kmeans = KMeans(n_clusters=K, n_init="auto", random_state=42)
    labels_kmeans = kmeans.fit_predict(X_pca)

    print("Finished KMeans.")


    # ===== 3) Evaluate KMeans with Davies–Bouldin Index =====
    try:
        dbi_kmeans = davies_bouldin_score(X_pca, labels_kmeans)
        print(f"Davies–Bouldin Index (KMeans, k={K}): {dbi_kmeans:.4f}")
    except Exception as e:
        print("Could not compute DBI for KMeans:", e)


    # ===== 4) DBSCAN =====
    print("Running DBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=500,
        min_samples=10,
        metric='euclidean'
    )

    labels_db = clusterer.fit_predict(X_pca)

    print("Finished DBSCAN.")


    # ===== 5) Evaluate DBSCAN with DBI (excluding noise) =====
    try:
        mask = labels_db != -1
        if mask.sum() > 1 and len(set(labels_db[mask])) > 1:
            dbi_db = davies_bouldin_score(X_pca[mask], labels_db[mask])
            print(f"Davies–Bouldin Index (DBSCAN, w/o noise): {dbi_db:.4f}")
            #dbi with noise points included would be:
            dbi_db_full = davies_bouldin_score(X_pca, labels_db)
            print(f"Davies–Bouldin Index (DBSCAN, with noise): {dbi_db_full:.4f}")
        else:
            print("DBSCAN created too few clusters (or only noise) for a DBI score.")
    except Exception as e:
        print("Could not compute DBI for DBSCAN:", e)


    # ===== 6) Save cluster results =====
    df = pd.DataFrame({
        "row_id": all_ids,
        "rating": all_ratings,
        "sentiment_score": all_sents,
        "cluster_kmeans": labels_kmeans,
        "cluster_dbscan": labels_db,
    })

    print("KMeans cluster distribution:")
    print(df["cluster_kmeans"].value_counts().sort_index())

    print("DBSCAN cluster distribution:")
    print(df["cluster_dbscan"].value_counts().sort_index())

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved clustering results to {OUT_CSV}")
    

if __name__ == "__main__":
    main()
