#!/usr/bin/env python3
"""
comprehensive_analysis.py

Computes multiple metrics for author pair analysis:
1. Mean/median percentiles (not just max)
2. Sample-size corrected scores
3. Effect sizes (Cohen's d)
4. Proportion above thresholds
5. Bootstrap confidence intervals

Usage:
    python comprehensive_analysis.py

Author: Generated for Tarah Wheeler's Sextant project
"""

import sqlite3
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT = "eltec-100"
MAIN_DB = f"./projects/{PROJECT}/db/{PROJECT}.db"
SVM_DB = f"./projects/{PROJECT}/db/svm.db"
OUTPUT_DIR = "./validation_output"

# Global variable for worker processes
GLOBAL_DATA = None


def init_worker(all_scores, corpus_mean, corpus_std):
    """Initialize worker process with shared data."""
    global GLOBAL_DATA
    GLOBAL_DATA = {
        'all_scores': all_scores,
        'corpus_mean': corpus_mean,
        'corpus_std': corpus_std
    }


def expected_max_percentile(n):
    """
    Expected maximum percentile for n random draws from uniform(0,100).
    E[max] = n/(n+1) * 100
    """
    return (n / (n + 1)) * 100


def expected_max_std(n):
    """
    Standard deviation of max for n draws from uniform(0,100).
    Var[max] = n / ((n+1)^2 * (n+2))
    """
    var = (n * 10000) / ((n + 1) ** 2 * (n + 2))
    return np.sqrt(var)


def bootstrap_mean_ci(scores, n_bootstrap=1000, ci=95):
    """Compute bootstrap confidence interval for mean."""
    if len(scores) < 2:
        return scores.mean(), scores.mean(), scores.mean()
    
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(sample.mean())
    
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return scores.mean(), lower, upper


def process_author_pair(args):
    """Process a single author pair with comprehensive metrics."""
    global GLOBAL_DATA
    
    src_name, tgt_name, group_scores, year_gap, n_pairs = args
    all_scores = GLOBAL_DATA['all_scores']
    corpus_mean = GLOBAL_DATA['corpus_mean']
    corpus_std = GLOBAL_DATA['corpus_std']
    
    # Basic stats
    pair_mean = group_scores.mean()
    pair_median = np.median(group_scores)
    pair_std = group_scores.std() if len(group_scores) > 1 else 0
    pair_max = group_scores.max()
    pair_min = group_scores.min()
    
    # Calculate percentiles for all scores in this group
    percentiles = np.array([(all_scores < score).mean() * 100 for score in group_scores])
    
    # Percentile-based metrics
    max_pctl = percentiles.max()
    mean_pctl = percentiles.mean()
    median_pctl = np.median(percentiles)
    
    # Proportion above thresholds
    pct_above_99 = (percentiles >= 99).mean() * 100
    pct_above_95 = (percentiles >= 95).mean() * 100
    pct_above_90 = (percentiles >= 90).mean() * 100
    pct_above_75 = (percentiles >= 75).mean() * 100
    pct_above_50 = (percentiles >= 50).mean() * 100
    
    # Sample-size correction for max percentile
    expected_max = expected_max_percentile(n_pairs)
    expected_max_sd = expected_max_std(n_pairs)
    
    # How many SDs above/below expected is the observed max?
    if expected_max_sd > 0:
        max_pctl_corrected = (max_pctl - expected_max) / expected_max_sd
    else:
        max_pctl_corrected = 0
    
    # Also compute "excess" - simple difference from expected
    max_pctl_excess = max_pctl - expected_max
    
    # Effect size: Cohen's d for this author pair vs corpus
    # d = (pair_mean - corpus_mean) / corpus_std
    cohens_d = (pair_mean - corpus_mean) / corpus_std if corpus_std > 0 else 0
    
    # Also compute effect size for percentiles
    # Mean percentile under null (random) would be 50
    pctl_effect = (mean_pctl - 50) / 28.87  # SD of uniform(0,100) ≈ 28.87
    
    # Bootstrap CI for mean percentile (reduced bootstrap for speed)
    mean_pctl_val, mean_pctl_lower, mean_pctl_upper = bootstrap_mean_ci(percentiles, n_bootstrap=500)
    
    # Confidence that pair is above 50th percentile (one-sample t-test)
    if len(percentiles) > 1 and percentiles.std() > 0:
        t_stat, p_value = stats.ttest_1samp(percentiles, 50)
        above_chance_p = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    else:
        above_chance_p = 0.5
    
    return {
        'source_author': src_name,
        'target_author': tgt_name,
        'n_chapter_pairs': n_pairs,
        'avg_year_gap': year_gap,
        
        # Raw score stats
        'score_mean': pair_mean,
        'score_median': pair_median,
        'score_std': pair_std,
        'score_max': pair_max,
        
        # Percentile stats
        'max_percentile': max_pctl,
        'mean_percentile': mean_pctl,
        'median_percentile': median_pctl,
        
        # Sample-size corrected
        'expected_max_pctl': expected_max,
        'max_pctl_excess': max_pctl_excess,
        'max_pctl_z_score': max_pctl_corrected,
        
        # Effect sizes
        'cohens_d': cohens_d,
        'percentile_effect': pctl_effect,
        
        # Proportion above thresholds
        'pct_above_99': pct_above_99,
        'pct_above_95': pct_above_95,
        'pct_above_90': pct_above_90,
        'pct_above_75': pct_above_75,
        'pct_above_50': pct_above_50,
        
        # Bootstrap CI
        'mean_pctl_ci_lower': mean_pctl_lower,
        'mean_pctl_ci_upper': mean_pctl_upper,
        
        # Statistical significance
        'above_chance_p': above_chance_p,
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
        df[col] = (df[col] - mean) / std
    
    print(f"  Final dataset: {len(df):,} text pairs")
    
    return df


def compute_influence_scores(df):
    """Compute influence scores with LOGISTIC REGRESSION coefficients."""
    weights = (0.792, 0.101, 0.108)
    hap_w, al_w, svm_w = weights
    
    df['influence_score'] = (
        df['hap_jac_dis'] * hap_w +
        df['al_jac_dis'] * al_w +
        df['svm_score'] * svm_w
    )
    
    return df, weights


def analyze_all_pairs(df, n_workers=None):
    """Analyze all author pairs with comprehensive metrics."""
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"\nUsing {n_workers} worker processes...")
    
    cross_author = df[df['same_author'] == 0].copy()
    all_scores = cross_author['influence_score'].values
    corpus_mean = all_scores.mean()
    corpus_std = all_scores.std()
    
    print(f"Total cross-author pairs: {len(cross_author):,}")
    print(f"Corpus mean score: {corpus_mean:.4f}")
    print(f"Corpus std score: {corpus_std:.4f}")
    
    # Prepare work items
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
                            initargs=(all_scores, corpus_mean, corpus_std)) as executor:
        futures = {executor.submit(process_author_pair, item): item for item in work_items}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 500 == 0 or completed == len(work_items):
                print(f"  Processed {completed:,}/{len(work_items):,} pairs ({completed/len(work_items)*100:.1f}%)")
    
    return pd.DataFrame(results)


def print_analysis(results_df):
    """Print comprehensive analysis results."""
    
    # =========================================================================
    # SECTION 1: Comparison of Metrics
    # =========================================================================
    print("\n" + "=" * 100)
    print("SECTION 1: COMPARING DIFFERENT METRICS")
    print("=" * 100)
    
    print("\n--- Distribution of Key Metrics Across All Author Pairs ---\n")
    
    metrics = ['max_percentile', 'mean_percentile', 'median_percentile', 
               'cohens_d', 'max_pctl_excess', 'pct_above_95']
    
    print(f"{'Metric':<20} {'Min':>10} {'5th':>10} {'25th':>10} {'Median':>10} {'75th':>10} {'95th':>10} {'Max':>10}")
    print("-" * 100)
    
    for metric in metrics:
        values = results_df[metric]
        print(f"{metric:<20} {values.min():>10.2f} {values.quantile(0.05):>10.2f} "
              f"{values.quantile(0.25):>10.2f} {values.median():>10.2f} "
              f"{values.quantile(0.75):>10.2f} {values.quantile(0.95):>10.2f} {values.max():>10.2f}")
    
    # =========================================================================
    # SECTION 2: Rankings by Different Metrics
    # =========================================================================
    print("\n" + "=" * 100)
    print("SECTION 2: TOP 30 PAIRS BY DIFFERENT METRICS")
    print("=" * 100)
    
    # By mean percentile
    print("\n--- TOP 30 by MEAN PERCENTILE (recommended primary metric) ---\n")
    top_by_mean = results_df.nlargest(30, 'mean_percentile')
    print(f"{'Rank':<5} {'Pair':<30} {'Mean%':>8} {'Med%':>8} {'Max%':>8} {'N':>7} {'d':>7}")
    print("-" * 80)
    for i, (_, row) in enumerate(top_by_mean.iterrows(), 1):
        pair = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair:<30} {row['mean_percentile']:>8.2f} {row['median_percentile']:>8.2f} "
              f"{row['max_percentile']:>8.2f} {row['n_chapter_pairs']:>7} {row['cohens_d']:>7.2f}")
    
    # By Cohen's d
    print("\n--- TOP 30 by COHEN'S D (effect size) ---\n")
    top_by_d = results_df.nlargest(30, 'cohens_d')
    print(f"{'Rank':<5} {'Pair':<30} {'d':>8} {'Mean%':>8} {'Max%':>8} {'N':>7}")
    print("-" * 75)
    for i, (_, row) in enumerate(top_by_d.iterrows(), 1):
        pair = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair:<30} {row['cohens_d']:>8.2f} {row['mean_percentile']:>8.2f} "
              f"{row['max_percentile']:>8.2f} {row['n_chapter_pairs']:>7}")
    
    # By sample-size corrected max
    print("\n--- TOP 30 by SAMPLE-SIZE CORRECTED MAX (z-score) ---\n")
    top_by_corrected = results_df.nlargest(30, 'max_pctl_z_score')
    print(f"{'Rank':<5} {'Pair':<30} {'Z':>8} {'Max%':>8} {'Exp%':>8} {'N':>7}")
    print("-" * 75)
    for i, (_, row) in enumerate(top_by_corrected.iterrows(), 1):
        pair = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair:<30} {row['max_pctl_z_score']:>8.2f} {row['max_percentile']:>8.2f} "
              f"{row['expected_max_pctl']:>8.2f} {row['n_chapter_pairs']:>7}")
    
    # By % above 95th
    print("\n--- TOP 30 by PROPORTION ABOVE 95th PERCENTILE ---\n")
    top_by_pct95 = results_df.nlargest(30, 'pct_above_95')
    print(f"{'Rank':<5} {'Pair':<30} {'%>95':>8} {'%>99':>8} {'Mean%':>8} {'N':>7}")
    print("-" * 75)
    for i, (_, row) in enumerate(top_by_pct95.iterrows(), 1):
        pair = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair:<30} {row['pct_above_95']:>8.2f} {row['pct_above_99']:>8.2f} "
              f"{row['mean_percentile']:>8.2f} {row['n_chapter_pairs']:>7}")
    
    # =========================================================================
    # SECTION 3: Bottom pairs by different metrics
    # =========================================================================
    print("\n" + "=" * 100)
    print("SECTION 3: BOTTOM 30 PAIRS BY DIFFERENT METRICS")
    print("=" * 100)
    
    # By mean percentile
    print("\n--- BOTTOM 30 by MEAN PERCENTILE ---\n")
    bottom_by_mean = results_df.nsmallest(30, 'mean_percentile')
    print(f"{'Rank':<5} {'Pair':<30} {'Mean%':>8} {'Med%':>8} {'Max%':>8} {'N':>7} {'d':>7}")
    print("-" * 80)
    for i, (_, row) in enumerate(bottom_by_mean.iterrows(), 1):
        pair = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair:<30} {row['mean_percentile']:>8.2f} {row['median_percentile']:>8.2f} "
              f"{row['max_percentile']:>8.2f} {row['n_chapter_pairs']:>7} {row['cohens_d']:>7.2f}")
    
    # By Cohen's d
    print("\n--- BOTTOM 30 by COHEN'S D ---\n")
    bottom_by_d = results_df.nsmallest(30, 'cohens_d')
    print(f"{'Rank':<5} {'Pair':<30} {'d':>8} {'Mean%':>8} {'Max%':>8} {'N':>7}")
    print("-" * 75)
    for i, (_, row) in enumerate(bottom_by_d.iterrows(), 1):
        pair = f"{row['source_author']} → {row['target_author']}"
        print(f"{i:<5} {pair:<30} {row['cohens_d']:>8.2f} {row['mean_percentile']:>8.2f} "
              f"{row['max_percentile']:>8.2f} {row['n_chapter_pairs']:>7}")
    
    # =========================================================================
    # SECTION 4: Documented Influence Pairs
    # =========================================================================
    print("\n" + "=" * 100)
    print("SECTION 4: DOCUMENTED INFLUENCE PAIRS - ALL METRICS")
    print("=" * 100)
    
    documented_pairs = [
        ("Eliot", "Lawrence"),
        ("Thackeray", "Disraeli"),
        ("Dickens", "Collins"),
        ("Brontë", "Collins"),
        ("Eliot", "Hardy"),
        ("Dickens", "Hardy"),
        ("Thackeray", "Trollope"),
        ("Brontë", "Gaskell"),
        ("Gaskell", "Dickens"),
    ]
    
    print(f"\n{'Pair':<25} {'N':>6} {'Mean%':>7} {'Med%':>7} {'Max%':>7} {'d':>6} "
          f"{'%>95':>6} {'%>99':>6} {'Z':>7} {'p':>8}")
    print("-" * 100)
    
    for src, tgt in documented_pairs:
        mask = (results_df['source_author'] == src) & (results_df['target_author'] == tgt)
        if mask.sum() > 0:
            row = results_df[mask].iloc[0]
            pair = f"{src} → {tgt}"
            print(f"{pair:<25} {row['n_chapter_pairs']:>6} {row['mean_percentile']:>7.2f} "
                  f"{row['median_percentile']:>7.2f} {row['max_percentile']:>7.2f} "
                  f"{row['cohens_d']:>6.2f} {row['pct_above_95']:>6.2f} {row['pct_above_99']:>6.2f} "
                  f"{row['max_pctl_z_score']:>7.2f} {row['above_chance_p']:>8.4f}")
        else:
            print(f"{src} → {tgt}: NOT FOUND")
    
    # =========================================================================
    # SECTION 5: True Negative Controls (Bottom 10 by mean)
    # =========================================================================
    print("\n" + "=" * 100)
    print("SECTION 5: TRUE NEGATIVE CONTROLS (Bottom 10 by Mean Percentile)")
    print("=" * 100)
    
    true_negatives = results_df.nsmallest(10, 'mean_percentile')
    
    print(f"\n{'Pair':<25} {'N':>6} {'Mean%':>7} {'Med%':>7} {'Max%':>7} {'d':>6} "
          f"{'%>95':>6} {'%>99':>6} {'Z':>7}")
    print("-" * 95)
    
    for _, row in true_negatives.iterrows():
        pair = f"{row['source_author']} → {row['target_author']}"
        print(f"{pair:<25} {row['n_chapter_pairs']:>6} {row['mean_percentile']:>7.2f} "
              f"{row['median_percentile']:>7.2f} {row['max_percentile']:>7.2f} "
              f"{row['cohens_d']:>6.2f} {row['pct_above_95']:>6.2f} {row['pct_above_99']:>6.2f} "
              f"{row['max_pctl_z_score']:>7.2f}")
    
    # =========================================================================
    # SECTION 6: Statistical Comparison
    # =========================================================================
    print("\n" + "=" * 100)
    print("SECTION 6: STATISTICAL COMPARISON - DOCUMENTED vs TRUE NEGATIVES")
    print("=" * 100)
    
    # Get documented pairs data
    doc_data = []
    for src, tgt in documented_pairs:
        mask = (results_df['source_author'] == src) & (results_df['target_author'] == tgt)
        if mask.sum() > 0:
            doc_data.append(results_df[mask].iloc[0])
    
    doc_df = pd.DataFrame(doc_data)
    neg_df = true_negatives
    
    print("\n--- Mean Values by Group ---\n")
    metrics_to_compare = ['mean_percentile', 'median_percentile', 'max_percentile', 
                          'cohens_d', 'pct_above_95', 'pct_above_99']
    
    print(f"{'Metric':<20} {'Documented':>12} {'Negatives':>12} {'Difference':>12} {'Ratio':>10}")
    print("-" * 70)
    
    for metric in metrics_to_compare:
        doc_mean = doc_df[metric].mean()
        neg_mean = neg_df[metric].mean()
        diff = doc_mean - neg_mean
        ratio = doc_mean / neg_mean if neg_mean != 0 else float('inf')
        print(f"{metric:<20} {doc_mean:>12.2f} {neg_mean:>12.2f} {diff:>+12.2f} {ratio:>10.2f}x")
    
    # Mann-Whitney U test
    print("\n--- Mann-Whitney U Tests (Documented > Negatives) ---\n")
    
    for metric in metrics_to_compare:
        doc_vals = doc_df[metric].values
        neg_vals = neg_df[metric].values
        
        if len(doc_vals) > 0 and len(neg_vals) > 0:
            u_stat, p_val = stats.mannwhitneyu(doc_vals, neg_vals, alternative='greater')
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{metric:<20} U={u_stat:>8.0f}  p={p_val:<10.4f} {significance}")
    
    # =========================================================================
    # SECTION 7: Correlation between metrics
    # =========================================================================
    print("\n" + "=" * 100)
    print("SECTION 7: CORRELATION BETWEEN METRICS")
    print("=" * 100)
    
    corr_metrics = ['max_percentile', 'mean_percentile', 'cohens_d', 
                    'pct_above_95', 'n_chapter_pairs', 'max_pctl_z_score']
    
    print("\n")
    print(f"{'':>20}", end='')
    for m in corr_metrics:
        print(f"{m[:8]:>10}", end='')
    print()
    
    for m1 in corr_metrics:
        print(f"{m1[:20]:<20}", end='')
        for m2 in corr_metrics:
            corr = results_df[m1].corr(results_df[m2])
            print(f"{corr:>10.3f}", end='')
        print()
    
    # =========================================================================
    # SECTION 8: Sample Size Effects
    # =========================================================================
    print("\n" + "=" * 100)
    print("SECTION 8: SAMPLE SIZE EFFECTS")
    print("=" * 100)
    
    # Correlation of N with metrics
    print("\n--- Correlation of N (chapter pairs) with metrics ---\n")
    
    for metric in ['max_percentile', 'mean_percentile', 'cohens_d', 'pct_above_95', 'max_pctl_z_score']:
        corr = results_df['n_chapter_pairs'].corr(results_df[metric])
        print(f"N vs {metric:<20}: r = {corr:+.3f}")
    
    print("\n--- Max Percentile by Sample Size Bins ---\n")
    
    results_df['n_bin'] = pd.cut(results_df['n_chapter_pairs'], 
                                  bins=[0, 50, 200, 1000, 5000, 50000],
                                  labels=['1-50', '51-200', '201-1000', '1001-5000', '5000+'])
    
    print(f"{'N Range':<15} {'Count':>8} {'Mean Max%':>12} {'Mean Mean%':>12} {'Mean d':>10}")
    print("-" * 60)
    
    for bin_label in ['1-50', '51-200', '201-1000', '1001-5000', '5000+']:
        bin_data = results_df[results_df['n_bin'] == bin_label]
        if len(bin_data) > 0:
            print(f"{bin_label:<15} {len(bin_data):>8} {bin_data['max_percentile'].mean():>12.2f} "
                  f"{bin_data['mean_percentile'].mean():>12.2f} {bin_data['cohens_d'].mean():>10.2f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    df, weights = compute_influence_scores(df)
    
    print(f"\nUsing weights: Hapax={weights[0]:.1%}, Align={weights[1]:.1%}, SVM={weights[2]:.1%}")
    
    # Analyze all pairs
    results_df = analyze_all_pairs(df)
    
    # Print comprehensive analysis
    print_analysis(results_df)
    
    # Save results
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'comprehensive_analysis.csv'), index=False)
    
    print(f"\n" + "=" * 100)
    print(f"Results saved to: {OUTPUT_DIR}/comprehensive_analysis.csv")
    print("=" * 100)


if __name__ == "__main__":
    main()