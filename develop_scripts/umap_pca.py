
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import umap
import pandas as pd


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
    OUT_DIR = Path("/dtu/blackhole/1a/222266/")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ID_COL = "row_id"
    RATING_COL = "review/score"
    SENT_COL = "sentiment_score"
    EMBEDDINGS_COL = "embedding_pca"

    print(f"Loading PCA embeddings from {PCA_PARQUET}")
    pf = pq.ParquetFile(PCA_PARQUET)

    all_ids = []
    all_ratings = []
    all_sents = []
    all_X = []

    for rb in pf.iter_batches(
        columns=[ID_COL, RATING_COL, SENT_COL, EMBEDDINGS_COL],
        batch_size=65536
    ):
        ids = rb.column(0).to_pylist()
        ratings = rb.column(1).to_pylist()
        sents = rb.column(2).to_pylist()
        emb_arr = rb.column(3)

        X = listarray_to_2d_numpy(emb_arr, dtype=np.float32)
        if X.size == 0:
            continue

        all_ids.extend(ids)
        all_ratings.extend(ratings)
        all_sents.extend(sents)
        all_X.append(X)

    if not all_X:
        print("No data found.")
        return

    X_all = np.vstack(all_X)
    print("Shape of PCA embeddings:", X_all.shape)

    # convert ratings/sentiments into arrays for coloring
    ratings = np.array(all_ratings)
    sentiments = np.array(all_sents, dtype=float)

    print("Running UMAP (15D) for clustering features...")
    reducer_15d = umap.UMAP(
        n_neighbors=30,
        min_dist=0.0,
        n_components=15,
        metric="cosine",
        random_state=42,
        init="random",
    )
    X_umap15 = reducer_15d.fit_transform(X_all)
    print("UMAP 15D shape:", X_umap15.shape)

    # In neues Parquet mit UMAP-Features schreiben
    # Wir speichern: row_id, rating, sentiment_score, embedding_umap_15
    print("Saving UMAP 15D embeddings to Parquet...")

    id_arr = pa.array(all_ids)
    rating_arr = pa.array(all_ratings)
    sent_arr = pa.array(all_sents)

    # X_umap15: (N, 15) -> Flat + FixedSizeListArray
    umap15_flat = pa.array(X_umap15.astype(np.float32).reshape(-1))
    umap15_list = pa.FixedSizeListArray.from_arrays(umap15_flat, 15)

    table = pa.Table.from_arrays(
        [id_arr, rating_arr, sent_arr, umap15_list],
        names=[ID_COL, RATING_COL, SENT_COL, "embedding_umap_15"],
    )

    out_parquet_umap = OUT_DIR / "embeddings_umap15.parquet"
    pq.write_table(table, out_parquet_umap)
    print(f"Saved UMAP 15D embeddings to {out_parquet_umap}")

    # UMAP 2D
    print("Running UMAP (2D) for visualization...")
    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.3,
        n_components=2,
        random_state=42,
        init="random"
    )
    X_umap2d = reducer.fit_transform(X_all)


    # Plot nach Sentiment
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        X_umap2d[:, 0],
        X_umap2d[:, 1],
        c=sentiments,
        s=5,
        alpha=0.6,
        cmap="coolwarm"
    )
    plt.colorbar(sc, label="sentiment_score")
    plt.title("UMAP of PCA embeddings colored by sentiment_score")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    out_path = OUT_DIR / "umap_pca_by_sentiment.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()