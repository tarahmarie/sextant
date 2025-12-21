#!/usr/bin/env python3
"""
lowest_influence_pairs_sample.py

FAST VERSION: Samples 1% of pairs to quickly find lowest-scoring author pairings.

Usage:
    python lowest_influence_pairs_sample.py

Author: Generated for Tarah Wheeler's Sextant project
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
SAMPLE_FRACTION = 0.01  # 1% sample


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


def rank_author_pairs_fast(df, sample_frac=0.01):
    """
    FAST: Sample pairs and compute approximate statistics.
    """
    print(f"\nUsing {sample_frac*100:.0f}% sample for fast analysis...")
    
    cross_author = df[df['same_author'] == 0].copy()
    total_pairs = len(cross_author)
    print(f"Total cross-author pairs: {total_pairs:,}")
    
    # Sample for percentile calculation
    sample_size = int(total_pairs * sample_frac)
    sample_df = cross_author.sample(n=sample_size, random_state=42)
    sample_scores = sample_df['influence_score'].values
    print(f"Sample size: {sample_size:,}")
    
    # Group by author pairs
    author_pairs = cross_author.groupby(['source_auth', 'target_auth', 
                                          'source_author_name', 'target_author_name'])
    
    print(f"Computing statistics for {len(author_pairs):,} author pairs...")
    
    results = []
    
    for (src_id, tgt_id, src_name, tgt_name), group in author_pairs:
        n_pairs = len(group)
        group_scores = group['influence_score'].values
        
        # Use sample to estimate percentiles (much faster)
        max_score = group_scores.max()
        min_score = group_scores.min()
        mean_score = group_scores.mean()
        
        # Percentile based on sample
        max_pctl = (sample_scores < max_score).mean() * 100
        min_pctl = (sample_scores < min_score).mean() * 100
        mean_pctl = (sample_scores < mean_score).mean() * 100
        
        # Year gap
        year_gap = group['target_year'].mean() - group['source_year'].mean()
        
        results.append({
            'source_author': src_name,
            'target_author': tgt_name,
            'n_chapter_pairs': n_pairs,
            'max_percentile': max_pctl,
            'min_percentile': min_pctl,
            'mean_percentile': mean_pctl,
            'max_score': max_score,
            'min_score': min_score,
            'avg_year_gap': year_gap,
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('max_percentile', ascending=True)
    
    return results_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    df, weights = compute_influence_scores(df)
    
    print(f"\nUsing weights: Hapax={weights[0]:.1%}, Align={weights[1]:.1%}, SVM={weights[2]:.1%}")
    
    # Fast ranking
    results_df = rank_author_pairs_fast(df, sample_frac=SAMPLE_FRACTION)
    
    # Show bottom 30
    print("\n" + "=" * 80)
    print("BOTTOM 30 AUTHOR PAIRS (lowest max percentile)")
    print("=" * 80)
    
    bottom_30 = results_df.head(30)
    print(f"\n{'Rank':<5} {'Source → Target':<35} {'Max%':<10} {'Mean%':<10} {'N':<8}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(bottom_30.iterrows(), 1):
        pair_name = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair_name:<35} {row['max_percentile']:<10.2f} {row['mean_percentile']:<10.2f} {row['n_chapter_pairs']:<8}")
    
    # Show top 30
    print("\n" + "=" * 80)
    print("TOP 30 AUTHOR PAIRS (highest max percentile)")
    print("=" * 80)
    
    top_30 = results_df.tail(30).iloc[::-1]
    print(f"\n{'Rank':<5} {'Source → Target':<35} {'Max%':<10} {'Mean%':<10} {'N':<8}")
    print("-" * 70)
    
    for i, (_, row) in enumerate(top_30.iterrows(), 1):
        pair_name = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair_name:<35} {row['max_percentile']:<10.2f} {row['mean_percentile']:<10.2f} {row['n_chapter_pairs']:<8}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal author pairs: {len(results_df):,}")
    print(f"\nMax percentile distribution:")
    print(f"  Minimum: {results_df['max_percentile'].min():.2f}%")
    print(f"  5th percentile: {results_df['max_percentile'].quantile(0.05):.2f}%")
    print(f"  25th percentile: {results_df['max_percentile'].quantile(0.25):.2f}%")
    print(f"  Median: {results_df['max_percentile'].median():.2f}%")
    print(f"  75th percentile: {results_df['max_percentile'].quantile(0.75):.2f}%")
    print(f"  Maximum: {results_df['max_percentile'].max():.2f}%")
    
    # How many below thresholds?
    below_90 = (results_df['max_percentile'] < 90).sum()
    below_80 = (results_df['max_percentile'] < 80).sum()
    below_70 = (results_df['max_percentile'] < 70).sum()
    below_50 = (results_df['max_percentile'] < 50).sum()
    
    print(f"\nAuthor pairs below thresholds:")
    print(f"  Max < 90%: {below_90} ({below_90/len(results_df)*100:.1f}%)")
    print(f"  Max < 80%: {below_80} ({below_80/len(results_df)*100:.1f}%)")
    print(f"  Max < 70%: {below_70} ({below_70/len(results_df)*100:.1f}%)")
    print(f"  Max < 50%: {below_50} ({below_50/len(results_df)*100:.1f}%)")
    
    # Key insight
    lowest = results_df['max_percentile'].min()
    print(f"\n" + "=" * 80)
    print(f"KEY FINDING: The lowest max percentile is {lowest:.2f}%")
    print("=" * 80)
    
    if lowest > 80:
        print("\nThis suggests Sextant sees ALL Victorian authors as similar.")
        print("The model may be detecting period style rather than influence.")
    elif lowest > 50:
        print("\nThere is SOME discrimination, but the floor is still high.")
    else:
        print("\nThere ARE genuinely dissimilar pairs that could serve as negative controls.")
    
    # Save
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'author_pairs_ranked_sample.csv'), index=False)
    print(f"\nResults saved to: {OUTPUT_DIR}/author_pairs_ranked_sample.csv")


if __name__ == "__main__":
    main()