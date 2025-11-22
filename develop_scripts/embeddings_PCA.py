import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from pathlib import Path
import glob

from sklearn.decomposition import PCA


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


def main():
    # === Pfade & Spaltennamen anpassen ===
    IN_DIR = Path("/dtu/blackhole/1a/222266/embeddings_subset")
    OUT_PARQUET = Path("/dtu/blackhole/1a/222266/embeddings_subset_pca.parquet")

    EMBEDDINGS_COL = "embedding"           # wie in deinen Embedding-Parquets
    ID_COL = "row_id"
    RATING_COL = "review/score"
    SENTIMENT_COL = "sentiment_score"      # aus deinem letzten Script

    N_COMPONENTS = 50                      # Ziel-Dimension der PCA

    files = sorted(glob.glob(str(IN_DIR / "*.parquet")))
    print(f"Found {len(files)} parquet files.")

    all_embeddings = []
    all_ids = []
    all_ratings = []
    all_sents = []

    # === 1) Alles einlesen ===
    for fp in tqdm(files, desc="Reading parquet files"):
        pf = pq.ParquetFile(fp)

        for rb in pf.iter_batches(
            columns=[ID_COL, RATING_COL, SENTIMENT_COL, EMBEDDINGS_COL],
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
            all_embeddings.append(X)

    if not all_embeddings:
        print("No embeddings found.")
        return

    X_all = np.vstack(all_embeddings)
    print("Total embeddings shape:", X_all.shape)
    print("Total ids:", len(all_ids))

    # === 2) PCA fit + transform ===
    print("Fitting PCA...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X_all).astype(np.float32)
    print("PCA done. Reduced shape:", X_pca.shape)

    # === 3) Arrow-Table mit PCA-Embeddings bauen ===
    arr_ids = pa.array(all_ids)
    arr_ratings = pa.array(all_ratings)
    arr_sents = pa.array(all_sents)

    list_of_lists = [row.tolist() for row in X_pca]
    arr_pca = pa.array(list_of_lists, type=pa.list_(pa.float32()))

    table = pa.Table.from_arrays(
        [arr_ids, arr_ratings, arr_sents, arr_pca],
        names=["row_id", "review/score", "sentiment_score", "embedding_pca"]
    )

    # === 4) Eine große Parquet-Datei schreiben ===
    pq.write_table(table, OUT_PARQUET, compression="snappy")
    print(f"Wrote PCA parquet to {OUT_PARQUET}")


if __name__ == "__main__":
    main()
