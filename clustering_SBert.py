import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from sklearn.cluster import KMeans
import pandas as pd


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

    # ===== load data =====
    from pathlib import Path
    import glob
    path_book_ratings = "data/Books_rating_embeddings"   
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

    kmeans = KMeans(n_clusters=K, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X_all)

    df_results = pd.DataFrame({
    "review_id": ids_all,
    "rating": ratings_all,
    "cluster": labels
    })


    csv_path = "data/clustered_reviews_Kmeans_1.csv"
    df_results.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()