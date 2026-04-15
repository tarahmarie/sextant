#!/usr/bin/env python3
"""
Score the Shelley-Lovelace corpus using frozen ELTeC ground truth weights.

Fully self-contained -- no dependency on model.py or ELTeC result files.
All parameters hardcoded from the ELTeC-100 validation run (Feb 22, 2026 rerun
with CHAPTER_PATTERN regex fix applied):
  - 5,956,426 pairs, ROC AUC 0.782 (10-fold CV), 0.781 (held-out test)
  - All 8 anchor cases above 99th percentile
  - SHAP: Hapax 86.2%, SVM 12.2%, Alignment 1.6%

Run from dh-trace root:
    python3 score_shelley_lovelace.py
"""
import sqlite3
import numpy as np
import pandas as pd

PROJECT = 'shelley-lovelace'

# ======================================================================
# ELTeC GROUND TRUTH PARAMETERS (frozen -- do not modify)
# Rerun Feb 22, 2026 with SVM chapter extraction bug fix applied.
# Previous values (pre-fix) superseded.
# ======================================================================

INTERCEPT = -4.204492292354901

COEFS = {
    'hap': -1.232514911232166,
    'al':  -0.15808776795659207,
    'svm':  0.1699869995259586,
}

# Z-score normalization (fitted on ELTeC 80% training split only)
MEANS = {
    'hap': 0.949792266894945,
    'al':  0.9999707384075235,
    'svm': 0.3244306975787624,
}
STDS = {
    'hap': 0.009913605118458133,
    'al':  0.0002607027907101452,
    'svm': 0.25803071899309576,
}


def load_and_score():
    main_conn = sqlite3.connect(f'./projects/{PROJECT}/db/{PROJECT}.db')
    svm_conn = sqlite3.connect(f'./projects/{PROJECT}/db/svm.db')

    df = pd.read_sql_query("""
        SELECT cj.*,
               t1.source_filename as source_name,
               t2.source_filename as target_name,
               t1.short_name_for_svm as source_svm_name,
               t2.short_name_for_svm as target_svm_name,
               t1.chapter_num as source_chapter,
               t2.chapter_num as target_chapter
        FROM combined_jaccard cj
        JOIN all_texts t1 ON cj.source_text = t1.text_id
        JOIN all_texts t2 ON cj.target_text = t2.text_id
    """, main_conn)

    print(f"Total pairs: {len(df):,}")

    # SVM lookup: "How confident is the SVM that the TARGET text
    # was written by the SOURCE author?" This is the influence signal.
    chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
    svm_conn.close()

    # Vectorized SVM join: melt wide table to long, then merge
    id_cols = ['novel', 'number']
    score_cols = [c for c in chapter_df.columns if c not in id_cols]
    chapter_long = chapter_df.melt(
        id_vars=id_cols,
        value_vars=score_cols,
        var_name='source_svm_name',
        value_name='svm_score'
    )
    chapter_long['number'] = chapter_long['number'].astype(str)
    df['target_chapter'] = df['target_chapter'].astype(str)

    df = df.merge(
        chapter_long,
        left_on=['target_svm_name', 'target_chapter', 'source_svm_name'],
        right_on=['novel', 'number', 'source_svm_name'],
        how='left'
    ).drop(columns=['novel', 'number'])

    n_with = df['svm_score'].notna().sum()
    n_without = df['svm_score'].isna().sum()
    print(f"Pairs with SVM scores: {n_with:,}")
    print(f"Pairs without SVM scores: {n_without:,}")
    df = df.dropna(subset=['svm_score'])

    # Z-score normalize using ELTeC training parameters
    df['hap_z'] = (df['hap_jac_dis'] - MEANS['hap']) / STDS['hap']
    df['al_z']  = (df['al_jac_dis']  - MEANS['al'])  / STDS['al']
    df['svm_z'] = (df['svm_score']   - MEANS['svm']) / STDS['svm']

    # Apply frozen ELTeC coefficients
    df['logit'] = (INTERCEPT
                   + COEFS['hap'] * df['hap_z']
                   + COEFS['al']  * df['al_z']
                   + COEFS['svm'] * df['svm_z'])
    df['prob'] = 1 / (1 + np.exp(-df['logit']))

    # Rank by probability (highest = most likely influence)
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    N = len(df)

    main_conn.close()
    return df, N


if __name__ == '__main__':
    df, N = load_and_score()

    print(f"\n{'='*80}")
    print(f"=== TOP 25 PAIRS BY INFLUENCE PROBABILITY ===")
    print(f"{'='*80}\n")
    for _, row in df.head(25).iterrows():
        print(f"  {row['rank']:5d}. p={row['prob']:.4f}  {row['source_name']} -> {row['target_name']}")

    # Shelley -> Lovelace
    sl = df[df['source_name'].str.contains('Shelley') & df['target_name'].str.contains('Lovelace')]
    print(f"\n{'='*80}")
    print(f"=== SHELLEY -> LOVELACE PAIRS ({len(sl)} pairs) ===")
    print(f"{'='*80}\n")
    for _, row in sl.iterrows():
        print(f"  rank {row['rank']:5d}/{N}  p={row['prob']:.4f}  "
              f"hap={row['hap_jac_dis']:.6f}  al={row['al_jac_dis']:.6f}  "
              f"svm={row['svm_score']:.4f}  {row['source_name']} -> {row['target_name']}")

    if len(sl) > 0:
        best = sl['rank'].min()
        print(f"\n  Best rank: {best} of {N} (top {best/N*100:.1f}%)")
        print(f"  Percentile: {(1 - best/N)*100:.1f}th")

    # All authors -> Lovelace (top 20)
    al_df = df[df['target_name'].str.contains('Lovelace')]
    print(f"\n{'='*80}")
    print(f"=== ALL AUTHORS -> LOVELACE (top 20) ===")
    print(f"{'='*80}\n")
    for _, row in al_df.head(20).iterrows():
        print(f"  rank {row['rank']:5d}/{N}  p={row['prob']:.4f}  "
              f"{row['source_name']} -> {row['target_name']}")

    # Somerville -> Lovelace
    som = df[df['source_name'].str.contains('Somerville') & df['target_name'].str.contains('Lovelace')]
    if len(som) > 0:
        print(f"\n{'='*80}")
        print(f"=== SOMERVILLE -> LOVELACE (top 10) ===")
        print(f"{'='*80}\n")
        for _, row in som.head(10).iterrows():
            print(f"  rank {row['rank']:5d}/{N}  p={row['prob']:.4f}  "
                  f"{row['source_name']} -> {row['target_name']}")

    # Babbage -> Lovelace
    bab = df[df['source_name'].str.contains('Babbage') & df['target_name'].str.contains('Lovelace')]
    if len(bab) > 0:
        print(f"\n{'='*80}")
        print(f"=== BABBAGE -> LOVELACE (top 10) ===")
        print(f"{'='*80}\n")
        for _, row in bab.head(10).iterrows():
            print(f"  rank {row['rank']:5d}/{N}  p={row['prob']:.4f}  "
                  f"{row['source_name']} -> {row['target_name']}")
