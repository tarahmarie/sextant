#!/usr/bin/env python3
"""
lowest_influence_pairs_mt.py

MULTITHREADED VERSION: Uses parallel processing to rank all author pairs.

Usage:
    python lowest_influence_pairs_mt.py

Author: Generated for Tarah Wheeler's Sextant project
"""

import sqlite3
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

# Configuration
PROJECT = "eltec-100"
MAIN_DB = f"./projects/{PROJECT}/db/{PROJECT}.db"
SVM_DB = f"./projects/{PROJECT}/db/svm.db"
OUTPUT_DIR = "./validation_output"

# Global variable for worker processes
GLOBAL_ALL_SCORES = None


def init_worker(all_scores):
    """Initialize worker process with shared data."""
    global GLOBAL_ALL_SCORES
    GLOBAL_ALL_SCORES = all_scores


def process_author_pair(args):
    """Process a single author pair - called by worker processes."""
    global GLOBAL_ALL_SCORES
    
    src_name, tgt_name, group_scores, year_gap, n_pairs = args
    
    # Calculate percentiles for this group
    percentiles = np.array([(GLOBAL_ALL_SCORES < score).mean() * 100 for score in group_scores])
    
    return {
        'source_author': src_name,
        'target_author': tgt_name,
        'n_chapter_pairs': n_pairs,
        'max_percentile': percentiles.max(),
        'min_percentile': percentiles.min(),
        'mean_percentile': percentiles.mean(),
        'median_percentile': np.median(percentiles),
        'std_percentile': percentiles.std(),
        'pct_above_99': (percentiles >= 99).mean() * 100,
        'pct_above_95': (percentiles >= 95).mean() * 100,
        'pct_above_90': (percentiles >= 90).mean() * 100,
        'pct_below_10': (percentiles <= 10).mean() * 100,
        'pct_below_5': (percentiles <= 5).mean() * 100,
        'avg_year_gap': year_gap,
    }


def load_data():
    """Load and prepare data from the project databases."""
    print("Loading data from databases...")
    
    main_conn = sqlite3.connect(MAIN_DB)
    
    query = """
    SELECT 
        cj.source_auth,
        cj.target_auth,
        cj.source_text,
        cj.target_text,
        cj.source_year,
        cj.target_year,
        cj.hap_jac_dis,
        cj.al_jac_dis,
        cj.pair_id,
        a1.source_filename as source_name,
        a2.source_filename as target_name,
        a1.chapter_num as source_chapter,
        a2.chapter_num as target_chapter,
        auth1.author_name as source_author_name,
        auth2.author_name as target_author_name
    FROM combined_jaccard cj
    JOIN all_texts a1 ON cj.source_text = a1.text_id
    JOIN all_texts a2 ON cj.target_text = a2.text_id
    JOIN authors auth1 ON cj.source_auth = auth1.id
    JOIN authors auth2 ON cj.target_auth = auth2.id
    """
    
    df = pd.read_sql_query(query, main_conn)
    print(f"  Loaded {len(df):,} text pairs")
    
    # Temporal filter
    df = df[df['source_year'] <= df['target_year']].copy()
    print(f"  After temporal filter: {len(df):,} pairs")
    
    df['same_author'] = (df['source_auth'] == df['target_auth']).astype(int)
    
    # Load SVM scores
    print(f"  Loading SVM scores...")
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
    
    # Match SVM scores
    print("  Matching SVM scores...")
    text_lookup_df = text_df.set_index('text_id')[['novel', 'number']]
    
    df = df.merge(
        text_lookup_df.rename(columns={'novel': 'target_novel', 'number': 'target_chapter_num'}),
        left_on='target_text', right_index=True, how='left'
    )
    
    source_novel_df = pd.DataFrame.from_dict(novels_dict, orient='index', columns=['source_novel_name'])
    df = df.merge(source_novel_df, left_on='source_auth', right_index=True, how='left')
    
    id_vars = ['novel', 'number']
    value_vars = [col for col in chapter_df.columns if col not in id_vars]
    chapter_melted = chapter_df.melt(
        id_vars=id_vars, value_vars=value_vars,
        var_name='source_novel_name', value_name='svm_score'
    )
    chapter_melted['number'] = chapter_melted['number'].astype(str)
    
    df = df.merge(
        chapter_melted,
        left_on=['target_novel', 'target_chapter_num', 'source_novel_name'],
        right_on=['novel', 'number', 'source_novel_name'],
        how='left'
    )
    
    cols_to_drop = ['target_novel', 'target_chapter_num', 'source_novel_name', 'novel', 'number']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df = df.dropna(subset=['svm_score'])
    
    # Z-score normalize
    print("  Z-score normalizing variables...")
    for col in ['hap_jac_dis', 'al_jac_dis', 'svm_score']:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_raw'] = df[col]
        df[col] = (df[col] - mean) / std
    
    print(f"  Final dataset: {len(df):,} text pairs")
    
    return df


def compute_influence_scores(df):
    """Compute influence scores with LOGISTIC REGRESSION coefficients."""
    # From the paper: Hapax 79.2%, SVM 10.8%, Alignment 10.1%
    weights = (0.792, 0.101, 0.108)
    hap_w, al_w, svm_w = weights
    
    df['influence_score'] = (
        df['hap_jac_dis'] * hap_w +
        df['al_jac_dis'] * al_w +
        df['svm_score'] * svm_w
    )
    
    return df, weights


def rank_all_author_pairs_mt(df, n_workers=None):
    """
    Compute statistics for ALL cross-author pairs using multiprocessing.
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"\nUsing {n_workers} worker processes...")
    
    cross_author = df[df['same_author'] == 0].copy()
    total_pairs = len(cross_author)
    all_scores = cross_author['influence_score'].values
    
    print(f"Total cross-author pairs: {total_pairs:,}")
    
    # Group by author pairs and prepare work items
    print("Preparing work items...")
    author_pairs = cross_author.groupby(['source_author_name', 'target_author_name'])
    
    work_items = []
    for (src_name, tgt_name), group in author_pairs:
        group_scores = group['influence_score'].values
        year_gap = group['target_year'].mean() - group['source_year'].mean()
        n_pairs = len(group)
        work_items.append((src_name, tgt_name, group_scores, year_gap, n_pairs))
    
    print(f"Processing {len(work_items):,} author pairs...")
    
    # Process in parallel
    results = []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=n_workers, 
                            initializer=init_worker, 
                            initargs=(all_scores,)) as executor:
        futures = {executor.submit(process_author_pair, item): item for item in work_items}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 500 == 0 or completed == len(work_items):
                print(f"  Processed {completed:,}/{len(work_items):,} pairs ({completed/len(work_items)*100:.1f}%)")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('max_percentile', ascending=True)
    
    return results_df


def analyze_results(results_df):
    """Analyze and print results."""
    
    # Bottom 50
    print("\n" + "=" * 80)
    print("BOTTOM 50 AUTHOR PAIRS (lowest max percentile)")
    print("=" * 80)
    
    bottom_50 = results_df.head(50)
    print(f"\n{'Rank':<5} {'Source → Target':<35} {'Max%':<10} {'Mean%':<10} {'N':<8} {'Gap':<6}")
    print("-" * 75)
    
    for i, (_, row) in enumerate(bottom_50.iterrows(), 1):
        pair_name = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair_name:<35} {row['max_percentile']:<10.2f} {row['mean_percentile']:<10.2f} "
              f"{row['n_chapter_pairs']:<8} {row['avg_year_gap']:<6.0f}")
    
    # Top 50
    print("\n" + "=" * 80)
    print("TOP 50 AUTHOR PAIRS (highest max percentile)")
    print("=" * 80)
    
    top_50 = results_df.tail(50).iloc[::-1]
    print(f"\n{'Rank':<5} {'Source → Target':<35} {'Max%':<10} {'Mean%':<10} {'N':<8} {'Gap':<6}")
    print("-" * 75)
    
    for i, (_, row) in enumerate(top_50.iterrows(), 1):
        pair_name = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair_name:<35} {row['max_percentile']:<10.2f} {row['mean_percentile']:<10.2f} "
              f"{row['n_chapter_pairs']:<8} {row['avg_year_gap']:<6.0f}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal author pairs analyzed: {len(results_df):,}")
    print(f"\nMax percentile distribution:")
    print(f"  Min:    {results_df['max_percentile'].min():.2f}%")
    print(f"  5th:    {results_df['max_percentile'].quantile(0.05):.2f}%")
    print(f"  25th:   {results_df['max_percentile'].quantile(0.25):.2f}%")
    print(f"  Median: {results_df['max_percentile'].median():.2f}%")
    print(f"  75th:   {results_df['max_percentile'].quantile(0.75):.2f}%")
    print(f"  95th:   {results_df['max_percentile'].quantile(0.95):.2f}%")
    print(f"  Max:    {results_df['max_percentile'].max():.2f}%")
    
    # Thresholds
    below_90 = (results_df['max_percentile'] < 90).sum()
    below_80 = (results_df['max_percentile'] < 80).sum()
    below_70 = (results_df['max_percentile'] < 70).sum()
    below_50 = (results_df['max_percentile'] < 50).sum()
    
    print(f"\nAuthor pairs below thresholds:")
    print(f"  Max < 90%: {below_90} ({below_90/len(results_df)*100:.1f}%)")
    print(f"  Max < 80%: {below_80} ({below_80/len(results_df)*100:.1f}%)")
    print(f"  Max < 70%: {below_70} ({below_70/len(results_df)*100:.1f}%)")
    print(f"  Max < 50%: {below_50} ({below_50/len(results_df)*100:.1f}%)")
    
    # Key finding
    lowest = results_df['max_percentile'].min()
    print(f"\n" + "=" * 80)
    print(f"KEY FINDING: The lowest max percentile is {lowest:.2f}%")
    print("=" * 80)
    
    return bottom_50, top_50


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    df, weights = compute_influence_scores(df)
    
    print(f"\nUsing weights: Hapax={weights[0]:.1%}, Align={weights[1]:.1%}, SVM={weights[2]:.1%}")
    
    # Rank all pairs with multiprocessing
    results_df = rank_all_author_pairs_mt(df)
    
    # Analyze
    bottom_50, top_50 = analyze_results(results_df)
    
    # Save
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'all_author_pairs_ranked.csv'), index=False)
    bottom_50.to_csv(os.path.join(OUTPUT_DIR, 'bottom_50_pairs.csv'), index=False)
    top_50.to_csv(os.path.join(OUTPUT_DIR, 'top_50_pairs.csv'), index=False)
    
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - all_author_pairs_ranked.csv")
    print(f"  - bottom_50_pairs.csv")
    print(f"  - top_50_pairs.csv")


if __name__ == "__main__":
    main()