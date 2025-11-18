import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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


def main():
    PCA_PARQUET = Path("/dtu/blackhole/1a/222266/embeddings_subset_pca.parquet")
    CLUSTER_CSV = Path("/dtu/blackhole/1a/222266/clustered_reviews_umap15_final.csv")
    OUT_DIR = Path("/zhome/3d/c/222266/ComputationalTools/AmazonReview/data/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ID_COL = "row_id"
    EMBEDDINGS_COL = "embedding_pca"

    # ===== 1) PCA-Embeddings laden =====
    print(f"Loading PCA embeddings from {PCA_PARQUET}")
    pf = pq.ParquetFile(PCA_PARQUET)

    all_ids = []
    all_X = []

    for rb in pf.iter_batches(columns=[ID_COL, EMBEDDINGS_COL], batch_size=65536):
        ids = rb.column(0).to_pylist()
        emb_arr = rb.column(1)
        X = listarray_to_2d_numpy(emb_arr, dtype=np.float32)
        if X.size == 0:
            continue
        all_ids.extend(ids)
        all_X.append(X)

    if not all_X:
        print("No data found.")
        return

    X_pca = np.vstack(all_X)
    print("PCA embeddings shape:", X_pca.shape)

    # ===== 2) Cluster-Labels laden & mergen =====
    print(f"Loading clusters from {CLUSTER_CSV}")
    df_clusters = pd.read_csv(CLUSTER_CSV, dtype={"row_id": str})



    labels_km = df_clusters["cluster_kmeans"].to_numpy()
    labels_db = df_clusters["cluster_dbscan"].to_numpy()

    # ===== 3) UMAP (2D) für Visualisierung =====
    print("Running UMAP (2D) for visualization...")
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.3,
        n_components=2,
        random_state=42,
        init="random"
    )
    X_umap2d = reducer.fit_transform(X_pca)

    # ===== 4) Plot: KMeans-Cluster =====
    base_cmap = plt.get_cmap("tab10")
    cnao = ListedColormap(base_cmap.colors[:len(np.unique(labels_km))]) 
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        X_umap2d[:, 0],
        X_umap2d[:, 1],
        c=labels_km,
        s=5,
        alpha=0.7,
        cmap=cnao
    )
    plt.colorbar(sc, label="cluster_kmeans")
    plt.title("UMAP of PCA embeddings colored by KMeans cluster")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    out_path = OUT_DIR / "umap_pca_clusters_kmeans.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

    # ===== 5) Plot: DBSCAN-Cluster =====
    n_clusters = len(np.unique(labels_db))
    cmap_db = plt.cm.get_cmap("hsv", n_clusters)
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        X_umap2d[:, 0],
        X_umap2d[:, 1],
        c=labels_db,
        s=5,
        alpha=0.7,
        cmap=cmap_db
    )
    plt.colorbar(sc, label="cluster_dbscan")
    plt.title("UMAP of PCA embeddings colored by DBSCAN cluster")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    out_path = OUT_DIR / "umap_pca_clusters_dbscan.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()