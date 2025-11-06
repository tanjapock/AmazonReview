#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--text-col", default="cleanText")
    p.add_argument("--id-col", default="Id")
    p.add_argument("--score-col", default="review/score")
    p.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                   help="SBERT-Modell-ID")
    p.add_argument("--chunksize", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sep", default=",", help="CSV-Separator (default ,)")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compression", default="snappy",
                   choices=["snappy", "gzip", "brotli", "zstd", "none"],
                   help="Parquet-Kompression (default snappy)")
    p.add_argument("--log-file", default=None, help="Pfad zur Logdatei (optional)")
    return p.parse_args()


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    logger = logging.getLogger("embeddings")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def as_arrow_table(ids, texts, score, embs):
    arr_ids = pa.array(ids)
    arr_texts = pa.array(texts, type=pa.string())
    arr_scores = pa.array(score)
    list_of_lists = [emb.tolist() for emb in embs]
    arr_embs = pa.array(list_of_lists, type=pa.list_(pa.float32()))
    return pa.Table.from_arrays(
        [arr_ids, arr_scores, arr_texts, arr_embs],
        names=["row_id", "review/score", "cleanText", "embedding"]
    )


def run(args, logger):
    os.makedirs(args.out_dir, exist_ok=True)
    compression = None if args.compression == "none" else args.compression

    logger.info("Start: in=%s out_dir=%s model=%s chunksize=%d batch_size=%d compression=%s resume=%s",
                args.in_path, args.out_dir, args.model, args.chunksize, args.batch_size, args.compression, args.resume)

    # Modell laden
    try:
        model = SentenceTransformer(args.model)
        logger.info("Model loaded.")
    except Exception:
        logger.exception("Could not load model.")
        sys.exit(2)

    # CSV-Reader
    try:
        reader = pd.read_csv(
            args.in_path, sep=args.sep, encoding=args.encoding,
            chunksize=args.chunksize, dtype=str, keep_default_na=False,
            on_bad_lines="skip"
        )
    except Exception:
        logger.exception("Input CSV could not be opened.")
        sys.exit(3)

    part_index = 0
    if args.resume:
        existing = sorted([f for f in os.listdir(args.out_dir)
                           if f.startswith("part-") and f.endswith(".parquet")])
        if existing:
            try:
                last = existing[-1]
                part_index = int(last.split("-")[1].split(".")[0]) + 1
                logger.info("Resume active - starting at part-%05d.", part_index)
            except Exception:
                logger.warning("Could not parse last part index; starting at 0.")

    total_rows = total_written = failed_chunks = 0
    running_index = 0

    for i, chunk in enumerate(tqdm(reader, desc="Chunks"), start=1):
        try:
            if args.text_col not in chunk.columns:
                raise KeyError(f"column '{args.text_col}' missing in CSV.")
            if args.score_col not in chunk.columns:
                raise KeyError(f"column '{args.score_col}' missing in CSV.")

            texts = chunk[args.text_col].astype(str).tolist()
            score = chunk[args.score_col].astype(str).tolist()

            if args.id_col and args.id_col in chunk.columns:
                ids = chunk[args.id_col].astype(str).tolist()
            else:
                ids = [str(j + running_index) for j in range(len(chunk))]
                running_index += len(chunk)

            # Embeddings
            embs = model.encode(
                texts,
                batch_size=args.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            ).astype(np.float32)

            table = as_arrow_table(ids, texts, score, embs)

            part_name = f"part-{part_index:05d}.parquet"
            target = os.path.join(args.out_dir, part_name)
            if args.resume and os.path.exists(target):
                logger.info("%s exists - skipping (resume).", part_name)
                part_index += 1
                continue

            tmp = target + ".tmp"
            if os.path.exists(tmp):
                os.remove(tmp)

            pq.write_table(table, tmp, compression=compression)
            os.replace(tmp, target)

            try:
                with open(os.path.join(args.out_dir, f"{part_name}.meta"), "w", encoding="utf-8") as m:
                    m.write(f"rows={len(texts)}\ncreated={datetime.utcnow().isoformat()}Z\nsource_chunk={i}\n")
            except Exception:
                logger.warning("Could not write meta file for %s", part_name)

            total_rows += len(texts)
            total_written += len(texts)
            part_index += 1
            logger.info("Chunk %d -> %s: %d rows", i, part_name, len(texts))

        except KeyboardInterrupt:
            logger.warning("User interrupted processing.")
            break
        except Exception:
            failed_chunks += 1
            logger.exception("Error in chunk %d", i)
            

    logger.info("Done: total_rows=%d, written=%d, failed_chunks=%d",
                total_rows, total_written, failed_chunks)

    try:
        parts = [f for f in os.listdir(args.out_dir)
                 if f.startswith("part-") and f.endswith(".parquet")]
        logger.info("Parts in output: %d", len(parts))
    except Exception:
        logger.warning("Could not list parts in output directory.")


if __name__ == "__main__":
    args = parse_args()

    # log file
    if args.log_file:
        log_path = args.log_file
    else:
        base = args.out_dir if os.path.isdir(args.out_dir) else (os.path.dirname(args.out_dir) or ".")
        log_path = os.path.join(base, "embeddings.log")

   
    logger = setup_logger(log_path)

    # Uncaught exceptions 
    def _excepthook(exc_type, exc, tb):
        logging.getLogger("embeddings").exception("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.exit(1)
    sys.excepthook = _excepthook

    run(args, logger)
