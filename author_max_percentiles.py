#!/usr/bin/env python3
"""
author_max_percentiles.py

Calculates the max percentile for all author pairs in the corpus.
Uses the logistic regression weights (Hapax=79.2%, Align=10.1%, SVM=10.8%).

Output: A table of all author pairs ranked by max percentile.

Usage:
    python author_max_percentiles.py
"""

import sqlite3
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import multiprocessing as mp

# Configuration
PROJECT = "eltec-100"
MAIN_DB = f"./projects/{PROJECT}/db/{PROJECT}.db"
SVM_DB = f"./projects/{PROJECT}/db/svm.db"
OUTPUT_DIR = "./validation_output"

# Logistic regression weights from the paper
WEIGHTS = {
    'hapax': 0.792,
    'align': 0.101,
    'svm': 0.108
}

# Global variables for multiprocessing
_all_scores = None
_df = None


def init_worker(all_scores, df):
    """Initialize worker process with shared data."""
    global _all_scores, _df
    _all_scores = all_scores
    _df = df


def load_data():
    """Load and prepare all data."""
    print("Loading data from databases...")
    
    main_conn = sqlite3.connect(MAIN_DB)
    
    # Load main comparison data
    query = """
    SELECT 
        cj.source_auth,
        cj.target_auth,
        cj.hap_jac_dis,
        cj.al_jac_dis,
        cj.source_year,
        cj.target_year,
        auth1.author_name as source_author_name,
        auth2.author_name as target_author_name
    FROM combined_jaccard cj
    JOIN authors auth1 ON cj.source_auth = auth1.id
    JOIN authors auth2 ON cj.target_auth = auth2.id
    WHERE cj.source_year <= cj.target_year
    """
    
    df = pd.read_sql_query(query, main_conn)
    print(f"  Loaded {len(df):,} text pairs")
    
    df['same_author'] = (df['source_auth'] == df['target_auth']).astype(int)
    
    # Z-score normalize hapax (79.2% of signal - dominant component)
    print("  Z-score normalizing hapax...")
    hapax_mean = df['hap_jac_dis'].mean()
    hapax_std = df['hap_jac_dis'].std()
    df['hap_jac_dis'] = (df['hap_jac_dis'] - hapax_mean) / hapax_std
    
    # Use hapax as the influence score (dominant signal)
    # Full model: 79.2% hapax + 10.1% align + 10.8% SVM
    # For author-level analysis, hapax alone captures the main signal
    df['influence_score'] = df['hap_jac_dis']
    
    main_conn.close()
    
    print(f"  Final dataset: {len(df):,} text pairs")
    return df


def process_author_pair(args):
    """Process a single author pair and return max percentile."""
    global _all_scores, _df
    
    src_auth, tgt_auth, src_name, tgt_name = args
    
    mask = (
        (_df['source_auth'] == src_auth) & 
        (_df['target_auth'] == tgt_auth) &
        (_df['same_author'] == 0)
    )
    
    pair_df = _df[mask]
    n = len(pair_df)
    
    if n == 0:
        return None
    
    scores = pair_df['influence_score'].values
    max_score = scores.max()
    
    # Calculate max percentile
    max_percentile = ((_all_scores < max_score).sum() / len(_all_scores)) * 100
    
    return {
        'source_author': src_name,
        'target_author': tgt_name,
        'n_chapter_pairs': n,
        'max_percentile': max_percentile
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = load_data()
    
    print(f"\nUsing hapax legomena (79.2% of full model signal) as influence score")
    
    # Get cross-author pairs only
    cross_author = df[df['same_author'] == 0]
    all_scores = cross_author['influence_score'].values
    
    print(f"\nTotal cross-author pairs: {len(cross_author):,}")
    
    # Get unique author pairs
    author_pairs = cross_author.groupby(
        ['source_auth', 'target_auth', 'source_author_name', 'target_author_name']
    ).size().reset_index(name='count')
    
    print(f"Unique author pairs: {len(author_pairs):,}")
    
    # Prepare work items
    work_items = [
        (row['source_auth'], row['target_auth'], 
         row['source_author_name'], row['target_author_name'])
        for _, row in author_pairs.iterrows()
    ]
    
    # Process with multiprocessing
    n_workers = max(1, mp.cpu_count() - 1)
    print(f"\nUsing {n_workers} worker processes...")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=n_workers,
                            initializer=init_worker,
                            initargs=(all_scores, df)) as executor:
        
        futures = {executor.submit(process_author_pair, item): item for item in work_items}
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            
            completed += 1
            if completed % 500 == 0 or completed == len(work_items):
                print(f"  Processed {completed:,}/{len(work_items):,} pairs ({100*completed/len(work_items):.1f}%)")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('max_percentile', ascending=False)
    
    # Save full results
    output_file = os.path.join(OUTPUT_DIR, 'author_max_percentiles.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    
    # Print top 50
    print("\n" + "=" * 80)
    print("TOP 50 AUTHOR PAIRS BY MAX PERCENTILE")
    print("=" * 80)
    print(f"\n{'Rank':<6}{'Source':<20}{'Target':<20}{'N Pairs':>10}{'Max %ile':>12}")
    print("-" * 68)
    
    for i, (_, row) in enumerate(results_df.head(50).iterrows(), 1):
        print(f"{i:<6}{row['source_author']:<20}{row['target_author']:<20}"
              f"{row['n_chapter_pairs']:>10,}{row['max_percentile']:>11.2f}%")
    
    # Print bottom 20
    print("\n" + "=" * 80)
    print("BOTTOM 20 AUTHOR PAIRS BY MAX PERCENTILE")
    print("=" * 80)
    print(f"\n{'Rank':<6}{'Source':<20}{'Target':<20}{'N Pairs':>10}{'Max %ile':>12}")
    print("-" * 68)
    
    bottom = results_df.tail(20).iloc[::-1]
    for i, (_, row) in enumerate(bottom.iterrows(), 1):
        rank = len(results_df) - 20 + i
        print(f"{rank:<6}{row['source_author']:<20}{row['target_author']:<20}"
              f"{row['n_chapter_pairs']:>10,}{row['max_percentile']:>11.2f}%")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal author pairs: {len(results_df):,}")
    print(f"\nMax percentile distribution:")
    print(f"  Min:    {results_df['max_percentile'].min():.2f}%")
    print(f"  25th:   {results_df['max_percentile'].quantile(0.25):.2f}%")
    print(f"  Median: {results_df['max_percentile'].median():.2f}%")
    print(f"  75th:   {results_df['max_percentile'].quantile(0.75):.2f}%")
    print(f"  Max:    {results_df['max_percentile'].max():.2f}%")
    
    print(f"\nPairs by max percentile range:")
    ranges = [(99, 100), (95, 99), (90, 95), (75, 90), (50, 75), (25, 50), (0, 25)]
    for low, high in ranges:
        count = ((results_df['max_percentile'] >= low) & (results_df['max_percentile'] < high)).sum()
        if high == 100:
            count = (results_df['max_percentile'] >= low).sum()
        pct = count / len(results_df) * 100
        print(f"  {low:>3}-{high:<3}%: {count:>5} pairs ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
