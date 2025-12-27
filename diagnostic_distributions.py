#!/usr/bin/env python3
"""
diagnostic_distributions.py

Examines the actual score distributions for key author pairs to understand
why mean percentile is low even when max percentile is high.

Usage:
    python diagnostic_distributions.py
"""

import sqlite3
import pandas as pd
import numpy as np
import os

# Configuration
PROJECT = "eltec-100"
MAIN_DB = f"./projects/{PROJECT}/db/{PROJECT}.db"
SVM_DB = f"./projects/{PROJECT}/db/svm.db"
OUTPUT_DIR = "./validation_output"


def load_data():
    """Load data (abbreviated version)."""
    print("Loading data...")
    
    main_conn = sqlite3.connect(MAIN_DB)
    
    query = """
    SELECT 
        cj.source_auth,
        cj.target_auth,
        cj.hap_jac_dis,
        cj.al_jac_dis,
        auth1.author_name as source_author_name,
        auth2.author_name as target_author_name
    FROM combined_jaccard cj
    JOIN all_texts a1 ON cj.source_text = a1.text_id
    JOIN all_texts a2 ON cj.target_text = a2.text_id
    JOIN authors auth1 ON cj.source_auth = auth1.id
    JOIN authors auth2 ON cj.target_auth = auth2.id
    WHERE cj.source_year <= cj.target_year
    """
    
    df = pd.read_sql_query(query, main_conn)
    print(f"  Loaded {len(df):,} pairs")
    
    df['same_author'] = (df['source_auth'] == df['target_auth']).astype(int)
    
    # Load SVM scores
    svm_conn = sqlite3.connect(SVM_DB)
    chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
    svm_conn.close()
    
    text_query = "SELECT text_id, source_filename, chapter_num FROM all_texts"
    text_df = pd.read_sql_query(text_query, main_conn)
    
    def extract_novel_name(filename):
        parts = filename.split('-chapter')[0]
        parts = parts.split('-')[1] if '-' in parts else parts
        return parts
    
    text_df['novel'] = text_df['source_filename'].apply(extract_novel_name)
    text_df['number'] = text_df['chapter_num'].astype(str)
    
    dirs_query = "SELECT id, dir FROM dirs"
    dirs_df = pd.read_sql_query(dirs_query, main_conn)
    novels_dict = {}
    for _, row in dirs_df.iterrows():
        dir_name = row['dir']
        novel_name = dir_name.split('-')[1] if '-' in dir_name else dir_name
        novels_dict[row['id']] = novel_name
    
    main_conn.close()
    
    # Merge SVM scores (abbreviated)
    text_lookup_df = text_df.set_index('text_id')[['novel', 'number']]
    
    # For speed, we'll just use hapax for this diagnostic
    # Z-score normalize hapax
    df['hap_jac_dis'] = (df['hap_jac_dis'] - df['hap_jac_dis'].mean()) / df['hap_jac_dis'].std()
    df['al_jac_dis'] = (df['al_jac_dis'] - df['al_jac_dis'].mean()) / df['al_jac_dis'].std()
    
    # Use just hapax for this diagnostic (79.2% of signal)
    df['influence_score'] = df['hap_jac_dis']
    
    print(f"  Final: {len(df):,} pairs")
    return df


def analyze_pair(df, cross_author, src_name, tgt_name, all_scores):
    """Analyze distribution for a specific author pair."""
    
    mask = (
        (df['source_author_name'] == src_name) & 
        (df['target_author_name'] == tgt_name) &
        (df['same_author'] == 0)
    )
    
    pair_df = df[mask]
    n = len(pair_df)
    
    if n == 0:
        print(f"\n{src_name} → {tgt_name}: NO PAIRS FOUND")
        return
    
    scores = pair_df['influence_score'].values
    
    # Calculate percentiles
    percentiles = np.array([(all_scores < s).mean() * 100 for s in scores])
    
    print(f"\n{'='*70}")
    print(f"{src_name} → {tgt_name} (N = {n:,})")
    print(f"{'='*70}")
    
    # Score distribution
    print(f"\nRAW SCORES (z-normalized):")
    print(f"  Min:    {scores.min():.3f}")
    print(f"  25th:   {np.percentile(scores, 25):.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    print(f"  75th:   {np.percentile(scores, 75):.3f}")
    print(f"  Max:    {scores.max():.3f}")
    print(f"  Mean:   {scores.mean():.3f}")
    print(f"  Std:    {scores.std():.3f}")
    
    # Percentile distribution
    print(f"\nPERCENTILES (position in corpus):")
    print(f"  Min:    {percentiles.min():.2f}%")
    print(f"  25th:   {np.percentile(percentiles, 25):.2f}%")
    print(f"  Median: {np.median(percentiles):.2f}%")
    print(f"  75th:   {np.percentile(percentiles, 75):.2f}%")
    print(f"  Max:    {percentiles.max():.2f}%")
    print(f"  Mean:   {percentiles.mean():.2f}%")
    
    # How many in each range?
    print(f"\nDISTRIBUTION BY PERCENTILE RANGE:")
    ranges = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 95), (95, 99), (99, 100)]
    for low, high in ranges:
        count = ((percentiles >= low) & (percentiles < high)).sum()
        pct = count / n * 100
        bar = '█' * int(pct / 2)
        print(f"  {low:>3}-{high:<3}%: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # Top 10 scores
    print(f"\nTOP 10 CHAPTER PAIRS:")
    top_idx = np.argsort(scores)[-10:][::-1]
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i:>2}. Score: {scores[idx]:.3f}  Percentile: {percentiles[idx]:.2f}%")
    
    # Expected vs observed
    expected_max = (n / (n + 1)) * 100
    print(f"\nEXPECTED vs OBSERVED:")
    print(f"  Expected max (random): {expected_max:.2f}%")
    print(f"  Observed max:          {percentiles.max():.2f}%")
    print(f"  Difference:            {percentiles.max() - expected_max:+.2f}%")
    
    # What percentile of the CORPUS distribution is the pair's median?
    corpus_median = np.median(all_scores)
    pair_median = np.median(scores)
    pair_median_pctl = (all_scores < pair_median).mean() * 100
    print(f"\nCORPUS COMPARISON:")
    print(f"  Corpus median score:     {corpus_median:.3f}")
    print(f"  This pair median score:  {pair_median:.3f}")
    print(f"  This pair median → corpus percentile: {pair_median_pctl:.2f}%")


def main():
    df = load_data()
    
    cross_author = df[df['same_author'] == 0]
    all_scores = cross_author['influence_score'].values
    
    print(f"\nCorpus: {len(all_scores):,} cross-author pairs")
    print(f"Corpus mean score: {all_scores.mean():.3f}")
    print(f"Corpus std score: {all_scores.std():.3f}")
    
    # Documented influence pairs
    pairs_to_analyze = [
        ("Eliot", "Lawrence"),
        ("Thackeray", "Disraeli"),
        ("Dickens", "Collins"),
        ("Brontë", "Gaskell"),
    ]
    
    # True negatives
    pairs_to_analyze += [
        ("Cross", "Conrad"),
        ("Barry", "Gissing"),
    ]
    
    # High scorers from the analysis
    pairs_to_analyze += [
        ("Morris", "Machen"),
        ("Morris", "Wells"),
    ]
    
    for src, tgt in pairs_to_analyze:
        analyze_pair(df, cross_author, src, tgt, all_scores)
    
    # Also check: what does the TOP of the corpus look like?
    print(f"\n{'='*70}")
    print("CORPUS TOP 0.01% SCORES")
    print(f"{'='*70}")
    
    top_threshold = np.percentile(all_scores, 99.99)
    print(f"\n99.99th percentile score: {top_threshold:.3f}")
    print(f"Number of pairs above: {(all_scores > top_threshold).sum():,}")
    
    top_01_threshold = np.percentile(all_scores, 99.9)
    print(f"\n99.9th percentile score: {top_01_threshold:.3f}")
    print(f"Number of pairs above: {(all_scores > top_01_threshold).sum():,}")
    
    top_1_threshold = np.percentile(all_scores, 99)
    print(f"\n99th percentile score: {top_1_threshold:.3f}")
    print(f"Number of pairs above: {(all_scores > top_1_threshold).sum():,}")


if __name__ == "__main__":
    main()