#!/usr/bin/env python3
"""
discriminant_analysis.py

Deeper analysis to find metrics that discriminate documented influence
from implausible pairings.

Examines:
1. Distribution shapes (not just max/mean)
2. Individual variable contributions (hapax vs alignment vs SVM)
3. Rank positions of top pairs
4. Proportion of pairs above thresholds
5. Statistical tests for distribution differences

Usage:
    python discriminant_analysis.py

Author: Generated for Tarah Wheeler's Sextant project
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
import os

# Configuration
PROJECT = "eltec-100"
MAIN_DB = f"./projects/{PROJECT}/db/{PROJECT}.db"
SVM_DB = f"./projects/{PROJECT}/db/svm.db"
OUTPUT_DIR = "./validation_output"

# ============================================================================
# ALL TEST PAIRS - Positive and Negative
# ============================================================================

POSITIVE_PAIRS = [
    {"name": "Eliot → Lawrence", "source_id": 34, "target_id": 19, "type": "positive"},
    {"name": "Thackeray → Disraeli", "source_id": 9, "target_id": 31, "type": "positive"},
    {"name": "Dickens → Collins", "source_id": 25, "target_id": 41, "type": "positive"},
    {"name": "Brontë → Collins", "source_id": 37, "target_id": 41, "type": "positive"},
    {"name": "Eliot → Hardy", "source_id": 34, "target_id": 29, "type": "positive"},
    {"name": "Dickens → Hardy", "source_id": 25, "target_id": 29, "type": "positive"},
    {"name": "Thackeray → Trollope", "source_id": 9, "target_id": 1, "type": "positive"},
    {"name": "Brontë → Gaskell", "source_id": 37, "target_id": 27, "type": "positive"},
    {"name": "Gaskell → Dickens", "source_id": 27, "target_id": 25, "type": "positive"},
]

NEGATIVE_PAIRS = [
    {"name": "Ainsworth → Woolf", "source_id": 4, "target_id": 47, "type": "negative"},
    {"name": "Reynolds → Forster", "source_id": 20, "target_id": 75, "type": "negative"},
    {"name": "Yonge → Lawrence", "source_id": 21, "target_id": 19, "type": "negative"},
    {"name": "Oliphant → Machen", "source_id": 2, "target_id": 7, "type": "negative"},
    {"name": "Stretton → Wells", "source_id": 63, "target_id": 14, "type": "negative"},
    {"name": "Kingsley → Glyn", "source_id": 36, "target_id": 12, "type": "negative"},
]

ALL_PAIRS = POSITIVE_PAIRS + NEGATIVE_PAIRS


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
    
    df = df[df['source_year'] <= df['target_year']].copy()
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
    
    # Z-score normalize AND keep raw values
    print("  Z-score normalizing variables...")
    for col in ['hap_jac_dis', 'al_jac_dis', 'svm_score']:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_raw'] = df[col]
        df[col] = (df[col] - mean) / std
    
    print(f"  Final dataset: {len(df):,} text pairs")
    
    return df


def compute_all_scores(df):
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


def get_pair_data(df, source_id, target_id):
    """Extract data for a specific author pair."""
    mask = (
        (df['source_auth'] == source_id) & 
        (df['target_auth'] == target_id) &
        (df['same_author'] == 0)
    )
    return df[mask].copy()


def calculate_detailed_stats(pairs_df, cross_author_df, pair_name):
    """Calculate detailed statistics for a pair."""
    if len(pairs_df) == 0:
        return None
    
    # Calculate percentiles
    pairs_df = pairs_df.copy()
    pairs_df['percentile'] = pairs_df['influence_score'].apply(
        lambda x: (cross_author_df['influence_score'] < x).mean() * 100
    )
    
    # Also calculate percentiles for individual variables
    pairs_df['hapax_pctl'] = pairs_df['hap_jac_dis'].apply(
        lambda x: (cross_author_df['hap_jac_dis'] < x).mean() * 100
    )
    pairs_df['align_pctl'] = pairs_df['al_jac_dis'].apply(
        lambda x: (cross_author_df['al_jac_dis'] < x).mean() * 100
    )
    pairs_df['svm_pctl'] = pairs_df['svm_score'].apply(
        lambda x: (cross_author_df['svm_score'] < x).mean() * 100
    )
    
    pairs_df = pairs_df.sort_values('influence_score', ascending=False)
    
    # Get ranks in full corpus
    all_scores_sorted = cross_author_df['influence_score'].sort_values(ascending=False).values
    top_score = pairs_df['influence_score'].iloc[0]
    rank = np.searchsorted(-all_scores_sorted, -top_score) + 1
    
    n = len(pairs_df)
    
    return {
        'name': pair_name,
        'n_pairs': n,
        # Combined score stats
        'max_pctl': pairs_df['percentile'].max(),
        'top10_mean': pairs_df.head(10)['percentile'].mean() if n >= 10 else pairs_df['percentile'].mean(),
        'median_pctl': pairs_df['percentile'].median(),
        'mean_pctl': pairs_df['percentile'].mean(),
        'std_pctl': pairs_df['percentile'].std(),
        'above_99': (pairs_df['percentile'] >= 99).sum(),
        'above_95': (pairs_df['percentile'] >= 95).sum(),
        'above_90': (pairs_df['percentile'] >= 90).sum(),
        'pct_above_99': (pairs_df['percentile'] >= 99).mean() * 100,
        'pct_above_95': (pairs_df['percentile'] >= 95).mean() * 100,
        'pct_above_90': (pairs_df['percentile'] >= 90).mean() * 100,
        'top_rank': rank,
        # Individual variable stats (max percentile for each)
        'hapax_max_pctl': pairs_df['hapax_pctl'].max(),
        'hapax_top10_mean': pairs_df.nlargest(10, 'hapax_pctl')['hapax_pctl'].mean() if n >= 10 else pairs_df['hapax_pctl'].mean(),
        'align_max_pctl': pairs_df['align_pctl'].max(),
        'align_top10_mean': pairs_df.nlargest(10, 'align_pctl')['align_pctl'].mean() if n >= 10 else pairs_df['align_pctl'].mean(),
        'svm_max_pctl': pairs_df['svm_pctl'].max(),
        'svm_top10_mean': pairs_df.nlargest(10, 'svm_pctl')['svm_pctl'].mean() if n >= 10 else pairs_df['svm_pctl'].mean(),
        # Distribution data for later analysis
        'all_percentiles': pairs_df['percentile'].values,
        'all_hapax_pctl': pairs_df['hapax_pctl'].values,
        'all_align_pctl': pairs_df['align_pctl'].values,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    df, weights = compute_all_scores(df)
    
    cross_author = df[df['same_author'] == 0]
    print(f"\nTotal cross-author pairs: {len(cross_author):,}")
    print(f"Weights used: Hapax={weights[0]:.0%}, Align={weights[1]:.0%}, SVM={weights[2]:.0%}")
    
    # Analyze all pairs
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF ALL PAIRS")
    print("=" * 80)
    
    results = []
    
    for pair_info in ALL_PAIRS:
        pairs_df = get_pair_data(df, pair_info['source_id'], pair_info['target_id'])
        stats = calculate_detailed_stats(pairs_df, cross_author, pair_info['name'])
        
        if stats:
            stats['type'] = pair_info['type']
            results.append(stats)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['all_percentiles', 'all_hapax_pctl', 'all_align_pctl']} 
                               for r in results])
    
    # ========================================================================
    # ANALYSIS 1: Compare distributions
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 1: DISTRIBUTION COMPARISON")
    print("=" * 80)
    
    pos_results = [r for r in results if r['type'] == 'positive']
    neg_results = [r for r in results if r['type'] == 'negative']
    
    print("\n--- Combined Score Percentiles ---")
    print(f"{'Metric':<25} {'Positive':<15} {'Negative':<15} {'Difference':<15}")
    print("-" * 70)
    
    for metric in ['max_pctl', 'top10_mean', 'median_pctl', 'mean_pctl', 'pct_above_99', 'pct_above_95']:
        pos_mean = np.mean([r[metric] for r in pos_results if r[metric] is not None])
        neg_mean = np.mean([r[metric] for r in neg_results if r[metric] is not None])
        diff = pos_mean - neg_mean
        print(f"{metric:<25} {pos_mean:<15.2f} {neg_mean:<15.2f} {diff:+.2f}")
    
    # ========================================================================
    # ANALYSIS 2: Individual variable contributions
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: INDIVIDUAL VARIABLE ANALYSIS")
    print("=" * 80)
    
    print("\n--- Hapax Legomena Only ---")
    print(f"{'Pair':<25} {'Type':<10} {'Max %ile':<12} {'Top-10 Mean':<12}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x['hapax_max_pctl'], reverse=True):
        print(f"{r['name']:<25} {r['type']:<10} {r['hapax_max_pctl']:<12.2f} {r['hapax_top10_mean']:<12.2f}")
    
    print("\n--- Sequence Alignment Only ---")
    print(f"{'Pair':<25} {'Type':<10} {'Max %ile':<12} {'Top-10 Mean':<12}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x['align_max_pctl'], reverse=True):
        print(f"{r['name']:<25} {r['type']:<10} {r['align_max_pctl']:<12.2f} {r['align_top10_mean']:<12.2f}")
    
    print("\n--- SVM (Style) Only ---")
    print(f"{'Pair':<25} {'Type':<10} {'Max %ile':<12} {'Top-10 Mean':<12}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x['svm_max_pctl'], reverse=True):
        print(f"{r['name']:<25} {r['type']:<10} {r['svm_max_pctl']:<12.2f} {r['svm_top10_mean']:<12.2f}")
    
    # ========================================================================
    # ANALYSIS 3: Rank-based analysis
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: RANK OF TOP PAIR (out of 5.79M)")
    print("=" * 80)
    
    print(f"\n{'Pair':<25} {'Type':<10} {'Top Rank':<12} {'Max %ile':<12}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x['top_rank']):
        print(f"{r['name']:<25} {r['type']:<10} {r['top_rank']:<12,} {r['max_pctl']:<12.2f}")
    
    # ========================================================================
    # ANALYSIS 4: Statistical tests
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 4: STATISTICAL TESTS")
    print("=" * 80)
    
    # Collect all percentiles for positive vs negative
    pos_all_pctl = np.concatenate([r['all_percentiles'] for r in pos_results])
    neg_all_pctl = np.concatenate([r['all_percentiles'] for r in neg_results])
    
    # Mann-Whitney U test
    u_stat, p_value = mannwhitneyu(pos_all_pctl, neg_all_pctl, alternative='greater')
    print(f"\nMann-Whitney U test (positive > negative):")
    print(f"  U statistic: {u_stat:,.0f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Positive mean percentile: {pos_all_pctl.mean():.2f}")
    print(f"  Negative mean percentile: {neg_all_pctl.mean():.2f}")
    
    # Same for alignment specifically
    pos_all_align = np.concatenate([r['all_align_pctl'] for r in pos_results])
    neg_all_align = np.concatenate([r['all_align_pctl'] for r in neg_results])
    
    u_stat_align, p_value_align = mannwhitneyu(pos_all_align, neg_all_align, alternative='greater')
    print(f"\nMann-Whitney U test for ALIGNMENT only:")
    print(f"  U statistic: {u_stat_align:,.0f}")
    print(f"  p-value: {p_value_align:.2e}")
    print(f"  Positive mean: {pos_all_align.mean():.2f}")
    print(f"  Negative mean: {neg_all_align.mean():.2f}")
    
    # Same for hapax specifically
    pos_all_hapax = np.concatenate([r['all_hapax_pctl'] for r in pos_results])
    neg_all_hapax = np.concatenate([r['all_hapax_pctl'] for r in neg_results])
    
    u_stat_hapax, p_value_hapax = mannwhitneyu(pos_all_hapax, neg_all_hapax, alternative='greater')
    print(f"\nMann-Whitney U test for HAPAX only:")
    print(f"  U statistic: {u_stat_hapax:,.0f}")
    print(f"  p-value: {p_value_hapax:.2e}")
    print(f"  Positive mean: {pos_all_hapax.mean():.2f}")
    print(f"  Negative mean: {neg_all_hapax.mean():.2f}")
    
    # ========================================================================
    # ANALYSIS 5: Percentage of pairs above thresholds
    # ========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 5: CONCENTRATION OF HIGH-SCORING PAIRS")
    print("=" * 80)
    
    print(f"\n{'Pair':<25} {'Type':<10} {'N':<8} {'%≥99th':<10} {'%≥95th':<10} {'%≥90th':<10}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x['pct_above_99'], reverse=True):
        print(f"{r['name']:<25} {r['type']:<10} {r['n_pairs']:<8} {r['pct_above_99']:<10.2f} {r['pct_above_95']:<10.2f} {r['pct_above_90']:<10.2f}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: KEY DISCRIMINATING METRICS")
    print("=" * 80)
    
    # Find metrics where positive > negative
    pos_df = results_df[results_df['type'] == 'positive']
    neg_df = results_df[results_df['type'] == 'negative']
    
    print("\nMetrics where POSITIVE pairs score higher (mean):")
    for col in ['max_pctl', 'top10_mean', 'median_pctl', 'pct_above_99', 'pct_above_95', 
                'hapax_max_pctl', 'align_max_pctl', 'svm_max_pctl']:
        pos_mean = pos_df[col].mean()
        neg_mean = neg_df[col].mean()
        if pos_mean > neg_mean:
            print(f"  {col}: positive={pos_mean:.2f}, negative={neg_mean:.2f}, diff={pos_mean-neg_mean:+.2f}")
    
    print("\nMetrics where NEGATIVE pairs score higher (mean):")
    for col in ['max_pctl', 'top10_mean', 'median_pctl', 'pct_above_99', 'pct_above_95',
                'hapax_max_pctl', 'align_max_pctl', 'svm_max_pctl']:
        pos_mean = pos_df[col].mean()
        neg_mean = neg_df[col].mean()
        if neg_mean > pos_mean:
            print(f"  {col}: positive={pos_mean:.2f}, negative={neg_mean:.2f}, diff={neg_mean-pos_mean:+.2f}")
    
    # Save detailed results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'discriminant_analysis_results.csv'), index=False)
    print(f"\nDetailed results saved to: {OUTPUT_DIR}/discriminant_analysis_results.csv")


if __name__ == "__main__":
    main()