import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import DBSCAN
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ===== Helper: Arrow-Listen to 2D NumPy =====
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
        raise TypeError(f"column embeddings must be List/FFixedSizeList: {t}")
    

# ===== Helper: Embeddings + Meta load =====
def extract_embeddings_and_meta(fp: str, embeddings_col: str, id_col: str, rating_col: str):
    """read embeddings, ID and rating from a Parquet file."""
    pf = pq.ParquetFile(fp)
    ids, ratings, mats = [], [], []

    for rb in pf.iter_batches(columns=[id_col, rating_col, embeddings_col], batch_size=65536):
        ids.append(rb.column(0).to_pylist())  # ID
        ratings.append(rb.column(1).to_pylist())  # Rating
        arr = rb.column(2)  # Embedding
        X = listarray_to_2d_numpy(arr, dtype=np.float32)
        if X.size > 0:
            mats.append(X)

    if not mats:
        return np.empty((0, 0)), np.array([]), np.array([])

    X = np.vstack(mats)
    ids = np.concatenate(ids)
    ratings = np.concatenate(ratings)
    return X, ids, ratings


# ===== Main =====
def main():
    # ===== Parameters =====
    NUM_FILES = 297
    EMBEDDINGS_COL = "embedding"
    ID_COL = "row_id"     
    RATING_COL = "review/score"    
    K = 5
    

    # ===== Load data =====
    from pathlib import Path
    import glob
    path_book_ratings = "/dtu/blackhole/1a/222266/Books_rating_embeddings"   
    in_dir = Path(path_book_ratings)
    files = sorted(glob.glob(str(in_dir / "*.parquet")))[:NUM_FILES]
    all_embeddings, all_ids, all_ratings = [], [], []

    for fp in tqdm(files):
        X, ids, ratings = extract_embeddings_and_meta(fp, EMBEDDINGS_COL, ID_COL, RATING_COL)
        all_embeddings.append(X)
        all_ids.append(ids)
        all_ratings.append(ratings)

    X_all = np.vstack(all_embeddings)
    ids_all = np.concatenate(all_ids)
    ratings_all = np.concatenate(all_ratings)

    print(f"Loaded all embeddings: shape = {X_all.shape}")

    # ===== PCA Dimensionality Reduction =====
    print(f"Running PCA to reduce to dimension...")
    pca = PCA(n_components=100, random_state=42)
    X_reduced = pca.fit_transform(X_all)
    explained_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA done. Retained {explained_var:.2f}% variance.")

    # ===== KMeans Clustering =====
    kmeans = KMeans(n_clusters=K, n_init="auto", random_state=42)
    labels_k_means = kmeans.fit_predict(X_reduced)

    print('Finished KMeans clustering.', flush=True)
    
    # ===== DBSCAN Clustering =====
    dbscan = DBSCAN(
        eps=2.6,        # max distance between samples in a cluster, tune this
        min_samples=20, # minimum number of samples to form a cluster
        metric='euclidean',
        n_jobs=-1
    )
    labels_db = dbscan.fit_predict(X_reduced)
    print('Finished DBSCAN clustering.', flush=True)

    # ===== Evaluate Clustering =====
    db_index_dbscan = davies_bouldin_score(X_reduced, labels_db)
    db_index_kmeans = davies_bouldin_score(X_reduced, labels_k_means)
    print(f"Davies–Bouldin Index (DBSCAN): {db_index_dbscan:.4f}")
    print(f"Davies–Bouldin Index (KMeans): {db_index_kmeans:.4f}")

    # ===== Save Results =====
    df_results = pd.DataFrame({
        "review_id": ids_all,
        "rating": ratings_all,
        "cluster_kmeans": labels_k_means,
        "cluster_db": labels_db
    })

    print("KMeans Cluster Distribution:")
    print(df_results["cluster_kmeans"].value_counts().sort_index())
    print("DBSCAN Cluster Distribution:")
    print(df_results["cluster_db"].value_counts().sort_index())

    csv_path = "/dtu/blackhole/1a/222266/clustered_reviews_pca.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")

    # ===== UMAP Visualization =====
    print("Running UMAP visualization...")
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=42, init='random')
    embedding_2d = reducer.fit_transform(X_reduced)

    # --- Visualization: KMeans ---
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels_k_means,
        cmap='tab10',
        s=15,
        alpha=0.8
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title("UMAP Projection of PCA-reduced Embeddings (KMeans)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig("/dtu/blackhole/1a/222266/plots/umap_clusters_kmeans_pca.png", dpi=300)
    plt.close()

    # --- Visualization: DBSCAN ---
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels_db,
        cmap='tab10',
        s=15,
        alpha=0.8
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title("UMAP Projection of PCA-reduced Embeddings (DBSCAN)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig("/dtu/blackhole/1a/222266/plots/umap_clusters_db_pca.png", dpi=300)
    plt.close()

    print("All done!")


if __name__ == "__main__":
    main()
