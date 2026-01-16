import math
import re
from pathlib import Path

import polars as pl

LEXICON_PATH = 'data/NRC-VAD-Lexicon-v2.1.txt'

PATH_TRAIN_1 = 'data/train_subtask1.csv'
PATH_TRAIN_2A = 'data/train_subtask2a.csv'
PATH_TRAIN_2B = 'data/train_subtask2b.csv'

OUT_PATH_1 = 'output/baseline_subtask1.csv'
OUT_PATH_2A = 'output/baseline_subtask2a.csv'
OUT_PATH_2B = 'output/baseline_subtask2b.csv'

LOWER_BOUND = -0.1
UPPER_BOUND = 0.1

def prepare_lexicon():
    df = pl.read_csv(LEXICON_PATH, separator='\t')
    df.drop_in_place('dominance')
    
    df_scaled = df.with_columns([
        (pl.col("valence") * 2).alias("valence"),
        (pl.col("arousal") * 2).alias("arousal")
    ])
    
    df_valence = df_scaled.drop('arousal').filter(
            (pl.col('valence') < LOWER_BOUND) | (pl.col('valence') > UPPER_BOUND)
    )
    df_arousal = df_scaled.drop('valence').filter(
            (pl.col('arousal') < LOWER_BOUND) | (pl.col('arousal') > UPPER_BOUND)
    )
    
    key_valence = df_valence['term'].to_list()
    key_arousal = df_arousal['term'].to_list()
    val_valence = df_valence['valence'].to_list()
    val_arousal = df_arousal['arousal'].to_list()
    
    valence_dict = dict(zip(key_valence, val_valence))
    arousal_dict = dict(zip(key_arousal, val_arousal))
    return valence_dict, arousal_dict

def load_dataset(path: str):
    print(f"Loading {path}...")
    df = pl.read_csv(path)
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").str.to_datetime())
    return df

def apply_lexicon_scoring(dataset: pl.DataFrame, lexicon: dict, col_name: str = 'valence'):
    print(f"Applying lexicon for {col_name}...")
    col_baseline = []
    
    target_col_exists = col_name in dataset.columns
    mse = 0.0
    
    def scale_score(raw_avg, c_name):
        if c_name == 'valence': return raw_avg * 2
        else: return raw_avg + 1.0

    for row in dataset.iter_rows(named=True):
        text = row.get('text', "")
        if text is None: text = ""
        text = text.lower()
        
        lexemes = re.sub(r'[^A-Za-z ]', '', text).split(' ')
        count = 0
        s = 0.0
        for lexeme in lexemes:
            if lexeme:
                score = lexicon.get(lexeme)
                if score is not None:
                    count += 1
                    s += score
        
        if count == 0: avg = 0.0
        else: avg = s / count
            
        final_avg = scale_score(avg, col_name)
        col_baseline.append(final_avg)
        
        if target_col_exists and row[col_name] is not None:
            mse += (row[col_name] - final_avg) ** 2
            
    if target_col_exists and dataset.shape[0] > 0:
        mse = mse / dataset.shape[0]
        print(f"Subtask 1: MSE {col_name}: {mse:.4f} (RMSE: {math.sqrt(mse):.4f})")

    return dataset.with_columns(pl.Series(f'{col_name}_baseline', col_baseline, dtype=pl.Float64))


def main():
    print("Preparing Lexicon")
    val_dict, ar_dict = prepare_lexicon()
    
    print("Processing Subtask 1")
    if Path(PATH_TRAIN_1).exists():
        df_1 = load_dataset(PATH_TRAIN_1)
        df_1 = apply_lexicon_scoring(df_1, val_dict, 'valence')
        df_1 = apply_lexicon_scoring(df_1, ar_dict, 'arousal')
        Path(OUT_PATH_1).parent.mkdir(parents=True, exist_ok=True)
        df_1.write_csv(OUT_PATH_1)
       
    print("All Done.")

if __name__ == "__main__":
    main()

