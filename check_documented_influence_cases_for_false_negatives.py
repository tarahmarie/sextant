#!/usr/bin/env python3
"""
check_untested_influence_pairs.py

Queries the Sextant database for documented literary influence relationships
that are NOT in the 8 anchor cases from the SIGHUM paper. These are potential
false negatives that confirmation assessors may ask about.

Run from your sextant root directory:
    python check_untested_influence_pairs.py

Requires:
- Main database: ./projects/eltec-100/db/eltec-100.db
- SVM database:  ./projects/eltec-100/db/svm.db

Output:
- Console summary table
- CSV file: untested_influence_results.csv
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT = "eltec-100"
MAIN_DB = f"./projects/{PROJECT}/db/{PROJECT}.db"
SVM_DB = f"./projects/{PROJECT}/db/svm.db"
OUTPUT_CSV = "./untested_influence_results.csv"

# These are the 8 anchor cases already in your paper - we EXCLUDE these
EXISTING_ANCHOR_CASES = [
    ("Eliot", "Lawrence"),
    ("Thackeray", "Disraeli"),
    ("Dickens", "Collins"),
    ("Thackeray", "Trollope"),
    ("Dickens", "Hardy"),
    ("Eliot", "Hardy"),
    ("Gaskell", "Dickens"),
    ("Bront", "Gaskell"),  # Brontë matching
]

# These are the NEW untested pairs we want to check
UNTESTED_PAIRS = [
    {
        "label": "Dickens → Reade",
        "source_pattern": "Dickens",
        "target_pattern": "Reade",
        "scholarship": "Reade called Dickens 'my master' (C.L. Reade & Compton Reade, 1887). "
                        "Personal friends; Reade turned to social problem novels under Dickens's influence.",
        "citations": "Reade & Reade (1887); Burns (1961); Smith (1976)",
    },
    {
        "label": "Dickens → Gissing",
        "source_pattern": "Dickens",
        "target_pattern": "Gissing",
        "scholarship": "Gissing wrote prefaces for 5 Dickens novels (1900-01) and "
                        "Charles Dickens: A Critical Study (1898). 'Saw London through Dickens's eyes.'",
        "citations": "Gissing (1898); Coustillas (2007); Poole (1975)",
    },
    {
        "label": "Collins → Braddon",
        "source_pattern": "Collins",
        "target_pattern": "Braddon",
        "scholarship": "Henry James (The Nation, 1865): Collins is 'to Miss Braddon what Richardson "
                        "is to Miss Austen.' Braddon inverted Collins's Woman in White victim into "
                        "Lady Audley's Secret villain.",
        "citations": "James (1865); Pykett (1992); Wolff (1979)",
    },
    {
        "label": "Disraeli → Kingsley",
        "source_pattern": "Disraeli",
        "target_pattern": "Kingsley",
        "scholarship": "Condition-of-England novel tradition. Disraeli's Coningsby (1844) and "
                        "Sybil (1845) established the template; Kingsley's Yeast (1848) follows.",
        "citations": "Cazamian (1903/1973); Gallagher (1985); Brantlinger (1977)",
    },
    {
        "label": "Gaskell → Kingsley",
        "source_pattern": "Gaskell",
        "target_pattern": "Kingsley",
        "scholarship": "Industrial novel tradition. Mary Barton (1848) and Yeast (1848) grouped "
                        "together in scholarship on social fiction.",
        "citations": "Cazamian (1903/1973); Gallagher (1985); Flint (1987)",
    },
    {
        "label": "Emily Brontë → Hardy",
        "source_pattern": "Bront",
        "target_pattern": "Hardy",
        "scholarship": "Wuthering Heights (1847) and Hardy's landscape-driven fiction share deep "
                        "structural connections. Return of the Native frequently compared to WH.",
        "citations": "Williams (1970); Cecil (1934); Guerard (1964)",
    },
    {
        "label": "Dickens → Trollope (Anthony)",
        "source_pattern": "Dickens",
        "target_pattern": "Trollope",
        "scholarship": "Trollope acknowledged Dickens's influence while also criticizing his methods. "
                        "Complex rivalry-influence relationship.",
        "citations": "Trollope, Autobiography (1883); Skilton (1972); Super (1988)",
    },
    {
        "label": "Hardy → Lawrence",
        "source_pattern": "Hardy",
        "target_pattern": "Lawrence",
        "scholarship": "Williams (1970) argues Hardy is the 'clear link' between Eliot and Lawrence. "
                        "Tests the intermediate step in the Eliot→Hardy→Lawrence chain.",
        "citations": "Williams (1970); Leavis (1948); Worthen (1991)",
    },
    {
        "label": "Eliot → Gissing",
        "source_pattern": "Eliot",
        "target_pattern": "Gissing",
        "scholarship": "Gissing admired Eliot alongside Dickens. His realist approach owes much "
                        "to her psychological depth.",
        "citations": "Coustillas (2007); Grylls (1986); Collie (1979)",
    },
    {
        "label": "Dickens → Braddon",
        "source_pattern": "Dickens",
        "target_pattern": "Braddon",
        "scholarship": "Braddon serialised in Dickens-adjacent publishing world. Sensation fiction "
                        "lineage runs through Dickens to Collins to Braddon.",
        "citations": "Carnell (2000); Pykett (1992); Brantlinger (1982)",
    },
]


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def load_combined_data():
    """
    Load all pairs with hapax, alignment, and SVM scores.
    
    Replicates the data loading from logistic_regression.py exactly:
    - Hapax + alignment from combined_jaccard in main DB
    - SVM scores from chapter_assessments in svm.db (wide format)
    - SVM score = P(target chapter was written by source author)
    
    FAILS if SVM scores cannot be loaded.
    """
    print("Loading data from databases...")
    
    # ---- Validate both databases exist ----
    if not os.path.exists(MAIN_DB):
        print(f"FATAL: Main database not found at {MAIN_DB}")
        print(f"  Run this script from your sextant root directory.")
        sys.exit(1)
    
    if not os.path.exists(SVM_DB):
        print(f"FATAL: SVM database not found at {SVM_DB}")
        print(f"  SVM scores are required. Cannot proceed without them.")
        sys.exit(1)
    
    # ---- Load main database ----
    conn_main = sqlite3.connect(MAIN_DB)
    
    # Get author names
    authors = pd.read_sql_query("SELECT id, author_name FROM authors", conn_main)
    author_map = dict(zip(authors['id'], authors['author_name']))
    
    # Get text info (needed for SVM join)
    texts = pd.read_sql_query("""
        SELECT text_id, author_id, source_filename, chapter_num, year,
               short_name_for_svm
        FROM all_texts
    """, conn_main)
    
    # Build text_id -> (novel, chapter_num) lookup for SVM matching
    # short_name_for_svm looks like "ENG18721—Eliot"
    text_lookup = texts.set_index('text_id')[['short_name_for_svm', 'chapter_num']].to_dict('index')
    
    # Build author_id -> novel name lookup (for SVM column matching)
    # Each author's novel column in chapter_assessments matches short_name_for_svm
    # NOTE: Some authors have multiple novels. The SVM column name corresponds to
    # the novel, not the author. We need to try all novels for a given author.
    # For now, get all unique (author_id, short_name_for_svm) pairs
    author_novel_groups = texts.groupby('author_id')['short_name_for_svm'].apply(
        lambda x: list(x.unique())
    ).to_dict()
    # For the merge, we need a single novel per author - use the first one
    # (this matches how logistic_regression.py builds novels_dict)
    author_novels = {aid: novels[0] for aid, novels in author_novel_groups.items()}
    
    # Load combined_jaccard
    print("  Loading combined_jaccard table...")
    df = pd.read_sql_query("""
        SELECT source_auth, source_year, source_text,
               target_auth, target_year, target_text,
               hap_jac_dis, al_jac_dis, pair_id,
               source_length, target_length
        FROM combined_jaccard
    """, conn_main)
    print(f"  Loaded {len(df):,} pairs from combined_jaccard")
    
    conn_main.close()
    
    # ---- Load SVM scores (wide format) ----
    print("  Loading SVM chapter_assessments (REQUIRED)...")
    conn_svm = sqlite3.connect(SVM_DB)
    
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn_svm
    )
    
    if 'chapter_assessments' not in tables['name'].values:
        print(f"FATAL: chapter_assessments table not found in {SVM_DB}")
        print(f"  Available tables: {tables['name'].tolist()}")
        sys.exit(1)
    
    chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", conn_svm)
    conn_svm.close()
    
    author_cols = [c for c in chapter_df.columns if c not in ['novel', 'number']]
    print(f"  Loaded {len(chapter_df):,} chapter assessments x {len(author_cols)} authors")
    
    # ---- Melt SVM table for efficient merge ----
    # From wide: novel, number, Author1_col, Author2_col, ...
    # To long:   novel, number, source_novel_name, svm_score
    print("  Reshaping SVM data (wide → long)...")
    chapter_melted = chapter_df.melt(
        id_vars=['novel', 'number'],
        value_vars=author_cols,
        var_name='source_novel_name',
        value_name='svm_score'
    )
    chapter_melted['number'] = chapter_melted['number'].astype(str)
    print(f"  Melted to {len(chapter_melted):,} rows")
    
    # ---- Map target text_id → (novel, chapter) for SVM join ----
    print("  Mapping target texts to SVM identifiers...")
    
    # target_novel = short_name_for_svm of the target text
    # target_chapter = chapter_num of the target text (as string)
    target_novels = []
    target_chapters = []
    for tid in df['target_text'].values:
        info = text_lookup.get(tid)
        if info:
            target_novels.append(info['short_name_for_svm'])
            target_chapters.append(str(info['chapter_num']))
        else:
            target_novels.append(None)
            target_chapters.append(None)
    
    df['target_novel'] = target_novels
    df['target_chapter'] = target_chapters
    
    # source_novel_name = short_name_for_svm of the source author's novel
    df['source_novel_name'] = df['source_auth'].map(author_novels)
    
    # ---- Merge SVM scores ----
    # SVM score = P(target chapter was written by source author)
    print("  Merging SVM scores...")
    pre_merge_len = len(df)
    df = df.merge(
        chapter_melted,
        left_on=['target_novel', 'target_chapter', 'source_novel_name'],
        right_on=['novel', 'number', 'source_novel_name'],
        how='left'
    )
    
    # Clean up merge columns
    df = df.drop(columns=['novel', 'number', 'target_novel', 'target_chapter',
                          'source_novel_name'], errors='ignore')
    
    # Check SVM match rate
    svm_matched = df['svm_score'].notna().sum()
    svm_rate = svm_matched / len(df) * 100
    print(f"  SVM match rate: {svm_matched:,} / {len(df):,} ({svm_rate:.1f}%)")
    
    if svm_rate < 50:
        print(f"FATAL: SVM match rate too low ({svm_rate:.1f}%).")
        print(f"  This indicates a join key mismatch.")
        print(f"  Sample target_novel values: {df['target_novel'].head(3).tolist() if 'target_novel' in df.columns else 'N/A'}")
        print(f"  Sample SVM novel values: {chapter_melted['novel'].unique()[:3].tolist()}")
        sys.exit(1)
    
    # Drop rows without SVM scores
    df = df.dropna(subset=['svm_score'])
    print(f"  After dropping NaN SVM: {len(df):,} pairs")
    
    # ---- Map author names and same_author flag ----
    df['source_author_name'] = df['source_auth'].map(author_map)
    df['target_author_name'] = df['target_auth'].map(author_map)
    df['same_author'] = (df['source_auth'] == df['target_auth']).astype(int)
    
    # ---- Diagnostics ----
    same = df[df['same_author'] == 1]['svm_score']
    diff = df[df['same_author'] == 0]['svm_score']
    print(f"\n  SVM score diagnostics:")
    print(f"    Same-author:  mean={same.mean():.4f}, std={same.std():.4f}")
    print(f"    Cross-author: mean={diff.mean():.4f}, std={diff.std():.4f}")
    print(f"    Difference:   {same.mean() - diff.mean():.4f}")
    
    if same.std() < 0.001 and diff.std() < 0.001:
        print(f"FATAL: SVM scores have no variance. The join is producing wrong values.")
        print(f"  This likely means self-attribution scores are being used instead of")
        print(f"  cross-author scores. Check the merge logic.")
        sys.exit(1)
    
    return df, author_map, texts


def compute_influence_scores(df):
    """
    Compute influence scores using the logistic regression approach.
    
    Z-score normalizes all three variables, then applies the trained
    logistic regression coefficients.
    
    Coefficients from the SIGHUM paper:
        hapax:     β = -1.232  (79.2% contribution via SHAP)
        alignment: β = -0.157  (10.1%)
        svm:       β = +0.168  (10.8%)
        intercept: β ≈ -4.5
    """
    print("\nComputing influence scores...")
    
    feature_cols = ['hap_jac_dis', 'al_jac_dis', 'svm_score']
    
    # Z-score normalize
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            print(f"FATAL: {col} has zero variance. Cannot normalize.")
            sys.exit(1)
        col_z = col + '_z'
        df[col_z] = (df[col] - mean) / std
        print(f"  {col}: mean={mean:.4f}, std={std:.4f}")
    
    # Apply logistic regression coefficients
    BETA_HAPAX = -1.232
    BETA_ALIGN = -0.157
    BETA_SVM = 0.168
    INTERCEPT = -4.5
    
    df['logit'] = (
        INTERCEPT +
        BETA_HAPAX * df['hap_jac_dis_z'] +
        BETA_ALIGN * df['al_jac_dis_z'] +
        BETA_SVM * df['svm_score_z']
    )
    
    df['influence_prob'] = 1 / (1 + np.exp(-df['logit']))
    
    print(f"  Score range: {df['influence_prob'].min():.4f} - {df['influence_prob'].max():.4f}")
    print(f"  Mean: {df['influence_prob'].mean():.4f}, Median: {df['influence_prob'].median():.4f}")
    
    return df


def get_pair_stats(df_cross, source_pattern, target_pattern):
    """
    Get statistics for a specific author pairing.
    
    Returns max percentile, median percentile, number of chapter pairs,
    and the top-scoring individual pair.
    """
    # Find pairs matching this author combination
    mask = (
        df_cross['source_author_name'].str.contains(source_pattern, case=False, na=False) &
        df_cross['target_author_name'].str.contains(target_pattern, case=False, na=False)
    )
    pairs = df_cross[mask]
    
    if len(pairs) == 0:
        # Try reverse direction
        mask_rev = (
            df_cross['source_author_name'].str.contains(target_pattern, case=False, na=False) &
            df_cross['target_author_name'].str.contains(source_pattern, case=False, na=False)
        )
        pairs = df_cross[mask_rev]
        if len(pairs) > 0:
            return get_pair_stats_from_pairs(df_cross, pairs, reversed=True)
        return None
    
    return get_pair_stats_from_pairs(df_cross, pairs, reversed=False)


def get_pair_stats_from_pairs(df_cross, pairs, reversed=False):
    """Compute statistics for a set of chapter pairs."""
    # Drop any pairs with NaN influence_prob (shouldn't happen with proper SVM loading)
    valid_pairs = pairs.dropna(subset=['influence_prob'])
    if len(valid_pairs) == 0:
        print(f"    WARNING: {len(pairs)} pairs found but ALL have NaN influence_prob")
        print(f"    This means SVM scores are missing for these pairs.")
        return None
    
    if len(valid_pairs) < len(pairs):
        print(f"    WARNING: {len(pairs) - len(valid_pairs)} of {len(pairs)} pairs have NaN scores (dropped)")
    
    all_scores = df_cross['influence_prob'].dropna().values
    
    # Percentile of each pair's score within all cross-author pairs
    pair_percentiles = []
    for score in valid_pairs['influence_prob'].values:
        pctl = (all_scores < score).mean() * 100
        pair_percentiles.append(pctl)
    
    pair_percentiles = np.array(pair_percentiles)
    
    # Key statistics
    max_pctl = pair_percentiles.max()
    median_pctl = np.median(pair_percentiles)
    mean_pctl = pair_percentiles.mean()
    
    # How many pairs above various thresholds
    above_99 = (pair_percentiles >= 99).sum()
    above_95 = (pair_percentiles >= 95).sum()
    above_90 = (pair_percentiles >= 90).sum()
    
    # Top-scoring pair details
    top_idx = valid_pairs['influence_prob'].idxmax()
    top_pair = valid_pairs.loc[top_idx]
    
    # Cohen's d: effect size comparing this author-pair's scores to corpus
    if len(valid_pairs) > 1:
        cohens_d = (valid_pairs['influence_prob'].mean() - df_cross['influence_prob'].dropna().mean()) / df_cross['influence_prob'].dropna().std()
    else:
        cohens_d = np.nan
    
    return {
        'n_pairs': len(valid_pairs),
        'max_percentile': max_pctl,
        'median_percentile': median_pctl,
        'mean_percentile': mean_pctl,
        'above_99': above_99,
        'above_95': above_95,
        'above_90': above_90,
        'cohens_d': cohens_d,
        'top_score': top_pair['influence_prob'],
        'top_source_text': top_pair['source_text'],
        'top_target_text': top_pair['target_text'],
        'reversed': reversed,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("UNTESTED INFLUENCE PAIRS: CHECKING FOR POTENTIAL FALSE NEGATIVES")
    print("=" * 70)
    print(f"\nThese are documented influence relationships NOT in your 8 anchor cases.")
    print(f"If Sextant doesn't detect them, you need an explanation for assessors.\n")
    
    # Load data
    df, author_map, texts = load_combined_data()
    
    # Compute influence scores
    df = compute_influence_scores(df)
    
    # Filter to cross-author pairs only
    df_cross = df[df['same_author'] == 0].copy()
    print(f"\nTotal cross-author pairs: {len(df_cross):,}")
    
    # Print available authors for reference
    print(f"\nAuthors in corpus ({len(author_map)}):")
    for aid, name in sorted(author_map.items(), key=lambda x: x[1]):
        print(f"  {aid:3d}: {name}")
    
    # =================================================================
    # First, verify the existing anchor cases still work
    # =================================================================
    print("\n" + "=" * 70)
    print("SANITY CHECK: Existing anchor cases")
    print("=" * 70)
    
    for source_pat, target_pat in EXISTING_ANCHOR_CASES:
        result = get_pair_stats(df_cross, source_pat, target_pat)
        if result:
            label = f"{source_pat} → {target_pat}"
            print(f"  {label:<25s}  max={result['max_percentile']:6.2f}%  "
                  f"median={result['median_percentile']:6.2f}%  "
                  f"N={result['n_pairs']:,}")
        else:
            print(f"  {source_pat} → {target_pat}: NOT FOUND IN CORPUS")
    
    # =================================================================
    # Now check the untested pairs
    # =================================================================
    print("\n" + "=" * 70)
    print("UNTESTED INFLUENCE PAIRS")
    print("=" * 70)
    
    results = []
    
    for pair_info in UNTESTED_PAIRS:
        label = pair_info['label']
        source_pat = pair_info['source_pattern']
        target_pat = pair_info['target_pattern']
        
        # Handle the Dickens→Gissing case where the patterns are swapped
        # because of how the note describes direction
        if pair_info.get('note') and 'source=Dickens, target=Gissing' in pair_info.get('note', ''):
            source_pat = 'Dickens'
            target_pat = 'Gissing'
        
        result = get_pair_stats(df_cross, source_pat, target_pat)
        
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"  Scholarship: {pair_info['scholarship'][:80]}...")
        print(f"  Citations: {pair_info['citations']}")
        
        if result:
            direction = " (REVERSED)" if result['reversed'] else ""
            print(f"  N chapter pairs: {result['n_pairs']:,}{direction}")
            print(f"  Max percentile:  {result['max_percentile']:.2f}%")
            print(f"  Median percentile: {result['median_percentile']:.2f}%")
            print(f"  Mean percentile: {result['mean_percentile']:.2f}%")
            print(f"  Cohen's d:       {result['cohens_d']:.3f}" if not np.isnan(result['cohens_d']) else "  Cohen's d: N/A (single pair)")
            print(f"  Pairs ≥99th:     {result['above_99']}")
            print(f"  Pairs ≥95th:     {result['above_95']}")
            print(f"  Pairs ≥90th:     {result['above_90']}")
            
            # Assessment
            if result['max_percentile'] >= 99:
                assessment = "✅ STRONG DETECTION"
            elif result['max_percentile'] >= 95:
                assessment = "⚠️  MODERATE DETECTION"
            elif result['max_percentile'] >= 90:
                assessment = "⚠️  WEAK DETECTION"
            else:
                assessment = "❌ NOT DETECTED"
            print(f"  Assessment:      {assessment}")
            
            results.append({
                'pair': label,
                'n_pairs': result['n_pairs'],
                'max_percentile': result['max_percentile'],
                'median_percentile': result['median_percentile'],
                'mean_percentile': result['mean_percentile'],
                'cohens_d': result['cohens_d'],
                'above_99': result['above_99'],
                'above_95': result['above_95'],
                'above_90': result['above_90'],
                'reversed': result['reversed'],
                'citations': pair_info['citations'],
                'assessment': assessment,
            })
        else:
            print(f"  ❌ NO PAIRS FOUND - authors may not be in corpus or temporal filter excludes them")
            results.append({
                'pair': label,
                'n_pairs': 0,
                'max_percentile': np.nan,
                'median_percentile': np.nan,
                'mean_percentile': np.nan,
                'cohens_d': np.nan,
                'above_99': 0,
                'above_95': 0,
                'above_90': 0,
                'reversed': False,
                'citations': pair_info['citations'],
                'assessment': '❌ NOT IN CORPUS',
            })
    
    # =================================================================
    # Summary table
    # =================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'Pair':<30s} {'N':>6s} {'Max%':>8s} {'Med%':>8s} {'d':>7s} {'≥99':>5s} {'Assessment'}")
    print("─" * 85)
    
    for _, row in results_df.iterrows():
        d_str = f"{row['cohens_d']:.2f}" if not np.isnan(row['cohens_d']) else "N/A"
        max_str = f"{row['max_percentile']:.2f}" if not np.isnan(row['max_percentile']) else "N/A"
        med_str = f"{row['median_percentile']:.2f}" if not np.isnan(row['median_percentile']) else "N/A"
        print(f"{row['pair']:<30s} {row['n_pairs']:>6d} {max_str:>8s} {med_str:>8s} "
              f"{d_str:>7s} {row['above_99']:>5d} {row['assessment']}")
    
    # =================================================================
    # Comparison with anchor cases
    # =================================================================
    print("\n\n" + "=" * 70)
    print("COMPARISON: ANCHOR CASES vs UNTESTED PAIRS")
    print("=" * 70)
    
    detected = results_df[results_df['max_percentile'] >= 99]
    moderate = results_df[(results_df['max_percentile'] >= 90) & (results_df['max_percentile'] < 99)]
    undetected = results_df[results_df['max_percentile'] < 90]
    not_in_corpus = results_df[results_df['n_pairs'] == 0]
    
    print(f"\n  Strong detection (≥99th): {len(detected)} pairs")
    for _, row in detected.iterrows():
        print(f"    {row['pair']}: {row['max_percentile']:.2f}%")
    
    print(f"\n  Moderate detection (90-99th): {len(moderate)} pairs")
    for _, row in moderate.iterrows():
        print(f"    {row['pair']}: {row['max_percentile']:.2f}%")
    
    print(f"\n  Not detected (<90th): {len(undetected)} pairs")
    for _, row in undetected.iterrows():
        if row['n_pairs'] > 0:
            print(f"    {row['pair']}: {row['max_percentile']:.2f}%")
    
    print(f"\n  Not in corpus: {len(not_in_corpus)} pairs")
    for _, row in not_in_corpus.iterrows():
        print(f"    {row['pair']}")
    
    # =================================================================
    # Save results
    # =================================================================
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")
    
    # =================================================================
    # Talking points for confirmation
    # =================================================================
    print("\n\n" + "=" * 70)
    print("TALKING POINTS FOR CONFIRMATION")
    print("=" * 70)
    
    testable = results_df[results_df['n_pairs'] > 0]
    if len(testable) > 0:
        avg_max = testable['max_percentile'].mean()
        print(f"\n  Average max percentile across untested pairs: {avg_max:.1f}%")
        
        strong_rate = len(testable[testable['max_percentile'] >= 99]) / len(testable) * 100
        print(f"  Strong detection rate (≥99th): {strong_rate:.0f}%")
        
        any_detection_rate = len(testable[testable['max_percentile'] >= 90]) / len(testable) * 100
        print(f"  Any detection rate (≥90th): {any_detection_rate:.0f}%")
        
        print(f"\n  If most untested pairs are detected → your 8 anchor cases are representative")
        print(f"  If some are missed → explain why (different influence mechanism, genre vs style, etc.)")
        print(f"  If none are detected → your model may only work for specific influence types")


if __name__ == "__main__":
    main()