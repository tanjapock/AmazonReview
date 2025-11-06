import os
import argparse
from collections import Counter
import pandas as pd
import fasttext
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/Books_rating_cleaned_only_eng.csv")
    ap.add_argument("--text-col", default="cleanText")
    ap.add_argument("--model", default="data/lid.176.ftz")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--sep", default=",")
    ap.add_argument("--min-tokens", type=int, default=1,
                    help="min amount of words")
    return ap.parse_args()


def safe_predict(ft_model, texts, k=1):
    """predict languages with error handling - only use highest prob label"""
    try:
        labels, _ = ft_model.predict(texts, k=k)
        langs = []
        for lab in labels:
            if not lab:
                langs.append("unknown")
            else:
                langs.append(lab[0].replace("__label__", ""))
        return langs
    except Exception:
        return ["unknown"] * len(texts)


def main():
    args = parse_args()
    assert os.path.exists(args.input), f"Input not found: {args.input}"
    assert os.path.exists(args.model), f"Model not found: {args.model}"

    ft = fasttext.load_model(args.model)

    total = en_count = unknown_count = 0
    lang_counter = Counter()

    tqdm.pandas()

    # read header
    with open(args.input, "r", encoding="utf-8") as f:
        header_line = f.readline()
    header_cols = header_line.strip().split(args.sep)

    # Check if text column exists
    if args.text_col not in header_cols:
        raise KeyError(f"Column '{args.text_col}' not found in CSV header: {header_cols}")

    header_written = False

    for chunk in tqdm(pd.read_csv(args.input, chunksize=args.chunk, sep=args.sep)):
        total += len(chunk)
        texts = chunk[args.text_col].fillna("").astype(str)

        # Filter short texts
        if args.min_tokens > 0:
            token_counts = texts.str.split().map(len)
            short_mask = token_counts < args.min_tokens
        else:
            short_mask = pd.Series(False, index=texts.index)

        langs = safe_predict(ft, texts.tolist(), k=1)
        chunk["lang"] = langs
        chunk.loc[short_mask, "lang"] = "unknown"

        lang_counter.update(chunk["lang"])

        en_mask = chunk["lang"].eq("en")
        unk_mask = chunk["lang"].eq("unknown")
        en_count += int(en_mask.sum())
        unknown_count += int(unk_mask.sum())

        en_chunk = chunk[en_mask]

        # Append to output file
        en_chunk.to_csv(args.output, mode="a", index=False, header=not header_written)
        header_written = True

    non_en = total - en_count - unknown_count

    #print(" Summary")
    #print(f"Total reviews processed : {total:,}")
    #print(f"English (en)            : {en_count:,}")
    #print(f"Unknown                 : {unknown_count:,}")
    #print(f"Non-English (others)    : {non_en:,}")

    # save language statistics
    stats_df = pd.DataFrame(
        [(lang, cnt, cnt / total * 100 if total else 0.0) for lang, cnt in lang_counter.items()],
        columns=["language", "count", "percent"]
    ).sort_values("count", ascending=False)
    stats_df.to_csv("data/language_counts-2.csv", index=False)

    print("English-only file written to:", args.output)
    print("Language statistics written to: data/language_counts-2.csv")


if __name__ == "__main__":
    main()