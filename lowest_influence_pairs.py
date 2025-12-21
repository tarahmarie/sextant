#!/usr/bin/env python3
"""
lowest_influence_pairs.py

Finds the book pairings with the LOWEST influence scores in Sextant.
These represent pairs where the model detects minimal textual relationship.

This helps us understand:
1. What does "no influence" look like to Sextant?
2. Are there patterns in low-scoring pairs (genre, time gap, etc.)?
3. Can we use low-scorers as true negative controls?

Usage:
    python lowest_influence_pairs.py

Output:
    - lowest_influence_pairs.csv: All author pairs ranked by max percentile
    - bottom_50_pairs.txt: Detailed analysis of 50 lowest-scoring pairs

Author: Generated for Tarah Wheeler's Sextant project
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from collections import defaultdict

# Configuration
PROJECT = "eltec-100"
MAIN_DB = f"./projects/{PROJECT}/db/{PROJECT}.db"
SVM_DB = f"./projects/{PROJECT}/db/svm.db"
OUTPUT_DIR = "./validation_output"


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
    
    # Also get dir info for source/target novels
    text_to_dir = pd.read_sql_query("SELECT text_id, dir FROM all_texts", main_conn)
    dir_to_name = pd.read_sql_query("SELECT id, dir as dir_name FROM dirs", main_conn)
    
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


def rank_all_author_pairs(df):
    """
    Compute statistics for ALL cross-author pairs and rank them.
    """
    print("\nComputing statistics for all author pairs...")
    
    cross_author = df[df['same_author'] == 0].copy()
    total_pairs = len(cross_author)
    
    # Calculate percentile for every pair (this is slow but comprehensive)
    print("  Calculating percentiles for all pairs...")
    all_scores = cross_author['influence_score'].values
    
    # Group by author pairs
    author_pairs = cross_author.groupby(['source_auth', 'target_auth', 
                                          'source_author_name', 'target_author_name'])
    
    results = []
    
    for (src_id, tgt_id, src_name, tgt_name), group in author_pairs:
        n_pairs = len(group)
        
        # Calculate percentiles for this group
        group_scores = group['influence_score'].values
        percentiles = np.array([(all_scores < score).mean() * 100 for score in group_scores])
        
        # Get year info
        src_years = group['source_year'].unique()
        tgt_years = group['target_year'].unique()
        year_gap = group['target_year'].mean() - group['source_year'].mean()
        
        results.append({
            'source_author': src_name,
            'target_author': tgt_name,
            'source_id': src_id,
            'target_id': tgt_id,
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
            'pct_below_1': (percentiles <= 1).mean() * 100,
            'avg_year_gap': year_gap,
            'source_year_range': f"{src_years.min()}-{src_years.max()}" if len(src_years) > 1 else str(src_years[0]),
            'target_year_range': f"{tgt_years.min()}-{tgt_years.max()}" if len(tgt_years) > 1 else str(tgt_years[0]),
        })
    
    results_df = pd.DataFrame(results)
    
    # Sort by max_percentile ascending (lowest first)
    results_df = results_df.sort_values('max_percentile', ascending=True)
    
    return results_df


def analyze_lowest_pairs(results_df, n=50):
    """Analyze the lowest-scoring pairs in detail."""
    
    print(f"\n" + "=" * 80)
    print(f"BOTTOM {n} AUTHOR PAIRS BY MAX PERCENTILE")
    print("(These pairs show the LEAST textual similarity)")
    print("=" * 80)
    
    bottom_n = results_df.head(n)
    
    print(f"\n{'Rank':<5} {'Source → Target':<35} {'Max%':<8} {'Mean%':<8} {'N':<8} {'Gap':<6}")
    print("-" * 75)
    
    for i, (_, row) in enumerate(bottom_n.iterrows(), 1):
        pair_name = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair_name:<35} {row['max_percentile']:<8.2f} {row['mean_percentile']:<8.2f} "
              f"{row['n_chapter_pairs']:<8} {row['avg_year_gap']:<6.0f}")
    
    return bottom_n


def analyze_highest_pairs(results_df, n=50):
    """Analyze the highest-scoring pairs for comparison."""
    
    print(f"\n" + "=" * 80)
    print(f"TOP {n} AUTHOR PAIRS BY MAX PERCENTILE")
    print("(For comparison with lowest pairs)")
    print("=" * 80)
    
    top_n = results_df.tail(n).iloc[::-1]  # Reverse to show highest first
    
    print(f"\n{'Rank':<5} {'Source → Target':<35} {'Max%':<8} {'Mean%':<8} {'N':<8} {'Gap':<6}")
    print("-" * 75)
    
    for i, (_, row) in enumerate(top_n.iterrows(), 1):
        pair_name = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair_name:<35} {row['max_percentile']:<8.2f} {row['mean_percentile']:<8.2f} "
              f"{row['n_chapter_pairs']:<8} {row['avg_year_gap']:<6.0f}")
    
    return top_n


def summary_statistics(results_df):
    """Print summary statistics about the distribution of author pair scores."""
    
    print(f"\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal author pairs analyzed: {len(results_df):,}")
    print(f"\nMax percentile distribution across all author pairs:")
    print(f"  Min:    {results_df['max_percentile'].min():.2f}%")
    print(f"  5th:    {results_df['max_percentile'].quantile(0.05):.2f}%")
    print(f"  25th:   {results_df['max_percentile'].quantile(0.25):.2f}%")
    print(f"  Median: {results_df['max_percentile'].median():.2f}%")
    print(f"  75th:   {results_df['max_percentile'].quantile(0.75):.2f}%")
    print(f"  95th:   {results_df['max_percentile'].quantile(0.95):.2f}%")
    print(f"  Max:    {results_df['max_percentile'].max():.2f}%")
    
    print(f"\nMean percentile distribution across all author pairs:")
    print(f"  Min:    {results_df['mean_percentile'].min():.2f}%")
    print(f"  Median: {results_df['mean_percentile'].median():.2f}%")
    print(f"  Max:    {results_df['mean_percentile'].max():.2f}%")
    
    # How many pairs have max < 90%?
    below_90 = (results_df['max_percentile'] < 90).sum()
    below_95 = (results_df['max_percentile'] < 95).sum()
    below_99 = (results_df['max_percentile'] < 99).sum()
    
    print(f"\nAuthor pairs with max percentile below thresholds:")
    print(f"  Max < 90%: {below_90} ({below_90/len(results_df)*100:.1f}%)")
    print(f"  Max < 95%: {below_95} ({below_95/len(results_df)*100:.1f}%)")
    print(f"  Max < 99%: {below_99} ({below_99/len(results_df)*100:.1f}%)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    df, weights = compute_influence_scores(df)
    
    print(f"\nUsing weights: Hapax={weights[0]:.0%}, Align={weights[1]:.0%}, SVM={weights[2]:.0%}")
    
    # Rank all author pairs
    results_df = rank_all_author_pairs(df)
    
    # Save full results
    output_file = os.path.join(OUTPUT_DIR, 'all_author_pairs_ranked.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nFull rankings saved to: {output_file}")
    
    # Analyze lowest pairs
    bottom_50 = analyze_lowest_pairs(results_df, n=50)
    
    # Analyze highest pairs for comparison
    top_50 = analyze_highest_pairs(results_df, n=50)
    
    # Summary statistics
    summary_statistics(results_df)
    
    # Save bottom 50 details
    bottom_file = os.path.join(OUTPUT_DIR, 'bottom_50_pairs.csv')
    bottom_50.to_csv(bottom_file, index=False)
    
    # Save top 50 details
    top_file = os.path.join(OUTPUT_DIR, 'top_50_pairs.csv')
    top_50.to_csv(top_file, index=False)
    
    print(f"\nBottom 50 pairs saved to: {bottom_file}")
    print(f"Top 50 pairs saved to: {top_file}")
    
    # Print key insight
    print(f"\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    lowest_max = results_df['max_percentile'].min()
    print(f"""
The LOWEST max percentile for any author pair is {lowest_max:.2f}%.

This means that even the most dissimilar authors in your corpus
have at least one chapter pair scoring at the {lowest_max:.0f}th percentile.

If this number is high (e.g., >80%), it suggests Sextant may be detecting
general Victorian prose similarity rather than specific influence.

If this number is low (e.g., <50%), there ARE author pairs that Sextant
considers genuinely dissimilar, which could serve as negative controls.
""")


if __name__ == "__main__":
    main()