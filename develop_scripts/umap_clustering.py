import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap , BoundaryNorm
from matplotlib.lines import Line2D
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
    unique_km = np.unique(labels_km)
    n_km = len(unique_km)

    label2idx_km = {lab: i for i, lab in enumerate(unique_km)}
    idx_km = np.array([label2idx_km[lab] for lab in labels_km])

    cmap_km = plt.cm.get_cmap("tab20", n_km)

    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        X_umap2d[:, 0],
        X_umap2d[:, 1],
        c=idx_km,
        s=5,
        alpha=0.7,
        cmap=cmap_km,
    )

    # Legend statt Colorbar
    handles = [
        Line2D([0], [0], marker='o', linestyle='',
            color=cmap_km(i), label=str(lab))
        for i, lab in enumerate(unique_km)
    ]
    plt.legend(handles=handles, title="cluster_kmeans",
            bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.title("UMAP of PCA embeddings colored by KMeans cluster")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    out_path = OUT_DIR / "umap_pca_clusters_kmeans2.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    # ===== 5) Plot: DBSCAN-Cluster =====
    # convert labels to pandas for counting
    labels = np.array(labels_db)

    # Maske für Noise
    noise_mask = labels == -1
    non_noise_mask = ~noise_mask

    # eindeutige Nicht-Noise-Cluster
    unique_non_noise = np.unique(labels[non_noise_mask])
    n_clusters = len(unique_non_noise)

    # Mappe Nicht-Noise-Label -> 0..n_clusters-1
    label2idx = {lab: i for i, lab in enumerate(unique_non_noise)}
    idx_non_noise = np.array([label2idx[lab] for lab in labels[non_noise_mask]])

    # Bunte Farben NUR für Nicht-Noise-Cluster
    cmap_clusters = plt.cm.get_cmap("hsv", n_clusters)  # schön bunt, kein Grau
    colors_clusters = cmap_clusters(range(n_clusters))
    cmap_clusters = ListedColormap(colors_clusters)

    plt.figure(figsize=(10, 7))

    # 1) Noise separat, klein + sehr transparent, eindeutig grau
    plt.scatter(
        X_umap2d[noise_mask, 0],
        X_umap2d[noise_mask, 1],
        c=[(0.5, 0.5, 0.5, 0.2)],  # nur Noise = grau, alpha 0.2
        s=3,
        marker='o',
        linewidths=0,
    )

    # 2) Alle echten Cluster bunt darüber
    sc = plt.scatter(
        X_umap2d[non_noise_mask, 0],
        X_umap2d[non_noise_mask, 1],
        c=idx_non_noise,
        s=5,
        alpha=0.9,
        cmap=cmap_clusters,
    )

    # -------- Legende: Noise + Top-K Cluster + "..." --------
    labels_series = pd.Series(labels[non_noise_mask])
    cluster_sizes = labels_series.value_counts().sort_values(ascending=False)

    top_k = 10
    top_labels = cluster_sizes.index[:top_k].tolist()

    handles = []

    # Noise
    handles.append(
        Line2D([0], [0], marker='o', linestyle='', color=(0.5, 0.5, 0.5, 0.8),
            label="noise (-1)")
    )

    # Top-K Cluster (ohne n, wie du wolltest)
    for lab in top_labels:
        i = label2idx[lab]
        handles.append(
            Line2D([0], [0], marker='o', linestyle='',
                color=colors_clusters[i],
                label=f"cluster {lab}")
        )

    # "… weitere Cluster"
    remaining = len(unique_non_noise) - len(top_labels)
    if remaining > 0:
        handles.append(
            Line2D([0], [0], marker='', linestyle='', label=f"… 92 further clusters")
        )

    plt.legend(handles=handles, title="HDBSCAN clusters",
            bbox_to_anchor=(1.05, 1), loc="upper left")

    #plt.title("UMAP of PCA embeddings colored by HDBSCAN clusters")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "umap_hdbscan.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()