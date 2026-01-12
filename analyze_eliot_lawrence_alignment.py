"""
Why Does Sequence Alignment Help Eliot → Lawrence So Much?

The ablation study showed:
- With alignment: Eliot→Lawrence ranks #225 out of 5.8M pairs
- Without alignment: Eliot→Lawrence ranks #83,262

This script investigates WHY by examining:
1. The actual alignment scores for Eliot→Lawrence pairs
2. How those scores compare to other anchor cases
3. What the top-ranking Eliot→Lawrence pairs look like with vs without alignment

Author: Tarah Wheeler
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

from util import get_project_name

warnings.filterwarnings('ignore')

RANDOM_STATE = 42

VALIDATION_CASES = [
    ('Eliot', 'Lawrence', 'Eliot → Lawrence'),
    ('Thackeray', 'Disraeli', 'Thackeray → Disraeli'),
    ('Dickens', 'Collins', 'Dickens → Collins'),
    ('Thackeray', 'Trollope', 'Thackeray → Trollope'),
    ('Dickens', 'Hardy', 'Dickens → Hardy'),
    ('Eliot', 'Hardy', 'Eliot → Hardy'),
    ('Gaskell', 'Dickens', 'Gaskell → Dickens'),
    ('Bront', 'Gaskell', 'Brontë → Gaskell'),
]


def load_data():
    """Load data with all three variables."""
    project_name = get_project_name()
    main_db_path = f"./projects/{project_name}/db/{project_name}.db"
    main_conn = sqlite3.connect(main_db_path)
    
    query = """
    SELECT 
        cj.source_auth, cj.target_auth, cj.source_text, cj.target_text,
        cj.source_year, cj.target_year, cj.hap_jac_dis, cj.al_jac_dis, cj.pair_id,
        a1.source_filename as source_name, a2.source_filename as target_name,
        a1.chapter_num as source_chapter, a2.chapter_num as target_chapter,
        auth1.author_name as source_author_name, auth2.author_name as target_author_name
    FROM combined_jaccard cj
    JOIN all_texts a1 ON cj.source_text = a1.text_id
    JOIN all_texts a2 ON cj.target_text = a2.text_id
    JOIN authors auth1 ON cj.source_auth = auth1.id
    JOIN authors auth2 ON cj.target_auth = auth2.id
    """
    
    df = pd.read_sql_query(query, main_conn)
    df = df[df['source_year'] <= df['target_year']].copy()
    df['same_author'] = (df['source_auth'] == df['target_auth']).astype(int)
    
    # Load SVM scores
    svm_db_path = f"./projects/{project_name}/db/svm.db"
    svm_conn = sqlite3.connect(svm_db_path)
    chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
    svm_conn.close()
    
    text_df = pd.read_sql_query("SELECT text_id, source_filename, chapter_num FROM all_texts", main_conn)
    
    def extract_novel_name(filename):
        parts = filename.split('-chapter')[0]
        parts = parts.split('-')[1]
        return parts
    
    text_df['novel'] = text_df['source_filename'].apply(extract_novel_name)
    text_df['number'] = text_df['chapter_num'].astype(str)
    
    dirs_df = pd.read_sql_query("SELECT id, dir FROM dirs", main_conn)
    novels_dict = {}
    for _, row in dirs_df.iterrows():
        dir_name = row['dir']
        novel_name = dir_name.split('-')[1] if '-' in dir_name else dir_name
        novels_dict[row['id']] = novel_name
    
    main_conn.close()
    
    text_lookup_df = text_df.set_index('text_id')[['novel', 'number']]
    df = df.merge(
        text_lookup_df.rename(columns={'novel': 'target_novel', 'number': 'target_chapter_str'}),
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
        left_on=['target_novel', 'target_chapter_str', 'source_novel_name'],
        right_on=['novel', 'number', 'source_novel_name'],
        how='left'
    )
    
    cols_to_drop = ['target_novel', 'target_chapter_str', 'source_novel_name', 'novel', 'number']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.dropna(subset=['svm_score'])
    
    return df


def analyze_anchor_case_features(df, source_substr, target_substr, label):
    """Analyze the feature distributions for a specific anchor case."""
    cross_author = df[df['same_author'] == 0]
    
    case_pairs = cross_author[
        (cross_author['source_author_name'].str.contains(source_substr, case=False, na=False)) &
        (cross_author['target_author_name'].str.contains(target_substr, case=False, na=False))
    ]
    
    if len(case_pairs) == 0:
        return None
    
    return {
        'label': label,
        'n_pairs': len(case_pairs),
        'hap_jac_dis_mean': case_pairs['hap_jac_dis'].mean(),
        'hap_jac_dis_min': case_pairs['hap_jac_dis'].min(),
        'hap_jac_dis_std': case_pairs['hap_jac_dis'].std(),
        'al_jac_dis_mean': case_pairs['al_jac_dis'].mean(),
        'al_jac_dis_min': case_pairs['al_jac_dis'].min(),
        'al_jac_dis_std': case_pairs['al_jac_dis'].std(),
        'svm_score_mean': case_pairs['svm_score'].mean(),
        'svm_score_max': case_pairs['svm_score'].max(),
        'svm_score_std': case_pairs['svm_score'].std(),
    }


def get_corpus_percentiles(df, values, column_name, lower_is_better=True):
    """Calculate what percentile a set of values falls at in the corpus."""
    cross_author = df[df['same_author'] == 0]
    all_values = cross_author[column_name].values
    
    percentiles = []
    for v in values:
        if lower_is_better:
            # For distance metrics, lower is better
            pct = (all_values > v).sum() / len(all_values) * 100
        else:
            # For scores, higher is better
            pct = (all_values < v).sum() / len(all_values) * 100
        percentiles.append(pct)
    
    return percentiles


def main():
    print("=" * 80)
    print("ANALYSIS: Why Does Alignment Help Eliot → Lawrence?")
    print("=" * 80)
    
    print("\nLoading data...")
    df = load_data()
    cross_author = df[df['same_author'] == 0]
    print(f"Loaded {len(cross_author):,} cross-author pairs")
    
    # Get corpus-wide statistics
    print("\n" + "=" * 80)
    print("CORPUS-WIDE FEATURE DISTRIBUTIONS (Cross-Author Pairs)")
    print("=" * 80)
    
    for col, name, lower_better in [
        ('hap_jac_dis', 'Hapax Jaccard Distance', True),
        ('al_jac_dis', 'Alignment Jaccard Distance', True),
        ('svm_score', 'SVM Score', False)
    ]:
        vals = cross_author[col]
        print(f"\n{name}:")
        print(f"  Mean: {vals.mean():.6f}")
        print(f"  Std:  {vals.std():.6f}")
        print(f"  Min:  {vals.min():.6f}")
        print(f"  Max:  {vals.max():.6f}")
        print(f"  1st percentile:  {vals.quantile(0.01):.6f}")
        print(f"  99th percentile: {vals.quantile(0.99):.6f}")
    
    # Analyze each anchor case
    print("\n" + "=" * 80)
    print("ANCHOR CASE FEATURE ANALYSIS")
    print("=" * 80)
    
    results = []
    for source_substr, target_substr, label in VALIDATION_CASES:
        stats = analyze_anchor_case_features(df, source_substr, target_substr, label)
        if stats:
            results.append(stats)
    
    # Create comparison table
    print("\n--- Mean Feature Values by Anchor Case ---")
    print(f"\n{'Case':<22} {'N pairs':>8} {'Hapax':>10} {'Align':>10} {'SVM':>10}")
    print("-" * 65)
    
    for r in results:
        print(f"{r['label']:<22} {r['n_pairs']:>8,} {r['hap_jac_dis_mean']:>10.4f} "
              f"{r['al_jac_dis_mean']:>10.4f} {r['svm_score_mean']:>10.4f}")
    
    # Compare Eliot→Lawrence specifically
    print("\n" + "=" * 80)
    print("DEEP DIVE: Eliot → Lawrence vs Other Anchor Cases")
    print("=" * 80)
    
    eliot_lawrence = [r for r in results if 'Eliot → Lawrence' in r['label']][0]
    
    # Get the best Eliot→Lawrence pair
    el_pairs = cross_author[
        (cross_author['source_author_name'].str.contains('Eliot', case=False, na=False)) &
        (cross_author['target_author_name'].str.contains('Lawrence', case=False, na=False))
    ].copy()
    
    print(f"\nEliot → Lawrence has {len(el_pairs):,} chapter pairs")
    
    # What percentile is their BEST alignment score at?
    best_align = el_pairs['al_jac_dis'].min()
    align_percentile = (cross_author['al_jac_dis'] > best_align).sum() / len(cross_author) * 100
    
    best_hapax = el_pairs['hap_jac_dis'].min()
    hapax_percentile = (cross_author['hap_jac_dis'] > best_hapax).sum() / len(cross_author) * 100
    
    best_svm = el_pairs['svm_score'].max()
    svm_percentile = (cross_author['svm_score'] < best_svm).sum() / len(cross_author) * 100
    
    print(f"\nBest Eliot→Lawrence pair feature percentiles:")
    print(f"  Hapax (lower=better):  {hapax_percentile:.2f}th percentile (score: {best_hapax:.6f})")
    print(f"  Alignment (lower=better): {align_percentile:.2f}th percentile (score: {best_align:.6f})")
    print(f"  SVM (higher=better):   {svm_percentile:.2f}th percentile (score: {best_svm:.6f})")
    
    # Compare to other anchor cases
    print("\n--- Best Alignment Scores by Anchor Case ---")
    print(f"\n{'Case':<22} {'Best al_jac_dis':>15} {'Corpus %ile':>12}")
    print("-" * 55)
    
    for source_substr, target_substr, label in VALIDATION_CASES:
        case_pairs = cross_author[
            (cross_author['source_author_name'].str.contains(source_substr, case=False, na=False)) &
            (cross_author['target_author_name'].str.contains(target_substr, case=False, na=False))
        ]
        if len(case_pairs) > 0:
            best = case_pairs['al_jac_dis'].min()
            pct = (cross_author['al_jac_dis'] > best).sum() / len(cross_author) * 100
            print(f"{label:<22} {best:>15.6f} {pct:>11.2f}%")
    
    # Show the actual best Eliot→Lawrence pairs
    print("\n" + "=" * 80)
    print("TOP 10 ELIOT → LAWRENCE PAIRS (by 3-variable model probability)")
    print("=" * 80)
    
    # Train quick model to get probabilities
    features = ['hap_jac_dis', 'svm_score', 'al_jac_dis']
    X = df[features].values
    y = df['same_author'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    # Score Eliot→Lawrence pairs
    X_el = el_pairs[features].values
    X_el_scaled = scaler.transform(X_el)
    el_pairs['prob_3var'] = model.predict_proba(X_el_scaled)[:, 1]
    
    # Also get 2-var probabilities
    features_2var = ['hap_jac_dis', 'svm_score']
    X_2var = df[features_2var].values
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        X_2var, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    scaler_2 = StandardScaler()
    X_train_2_scaled = scaler_2.fit_transform(X_train_2)
    model_2 = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model_2.fit(X_train_2_scaled, y_train_2)
    
    X_el_2 = el_pairs[features_2var].values
    X_el_2_scaled = scaler_2.transform(X_el_2)
    el_pairs['prob_2var'] = model_2.predict_proba(X_el_2_scaled)[:, 1]
    
    # Show top pairs
    top_el = el_pairs.nlargest(10, 'prob_3var')
    
    print(f"\n{'Source':<30} {'Target':<30} {'Hapax':>8} {'Align':>8} {'SVM':>6} {'P(3v)':>7} {'P(2v)':>7}")
    print("-" * 105)
    
    for _, row in top_el.iterrows():
        src = row['source_name'][:28] if len(row['source_name']) > 28 else row['source_name']
        tgt = row['target_name'][:28] if len(row['target_name']) > 28 else row['target_name']
        print(f"{src:<30} {tgt:<30} {row['hap_jac_dis']:>8.4f} {row['al_jac_dis']:>8.4f} "
              f"{row['svm_score']:>6.3f} {row['prob_3var']:>7.4f} {row['prob_2var']:>7.4f}")
    
    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    
    # Calculate correlation between alignment and probability boost
    el_pairs['prob_boost'] = el_pairs['prob_3var'] - el_pairs['prob_2var']
    corr = el_pairs['al_jac_dis'].corr(el_pairs['prob_boost'])
    
    print(f"""
Correlation between alignment score and probability boost: {corr:.4f}
(Negative correlation expected: lower al_jac_dis → higher boost from alignment)

The 3-variable model ranks Eliot→Lawrence at #225 because:
""")
    
    # What's special about the top pair?
    best_pair = el_pairs.loc[el_pairs['prob_3var'].idxmax()]
    
    print(f"The top-ranked Eliot→Lawrence pair:")
    print(f"  Source: {best_pair['source_name']}")
    print(f"  Target: {best_pair['target_name']}")
    print(f"  Hapax distance:     {best_pair['hap_jac_dis']:.6f} ({hapax_percentile:.1f}th percentile)")
    print(f"  Alignment distance: {best_pair['al_jac_dis']:.6f} ({align_percentile:.1f}th percentile)")
    print(f"  SVM score:          {best_pair['svm_score']:.4f} ({svm_percentile:.1f}th percentile)")
    print(f"  3-var probability:  {best_pair['prob_3var']:.4f}")
    print(f"  2-var probability:  {best_pair['prob_2var']:.4f}")
    print(f"  Boost from alignment: {best_pair['prob_3var'] - best_pair['prob_2var']:+.4f}")


if __name__ == "__main__":
    main()
