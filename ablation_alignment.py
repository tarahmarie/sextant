"""
Ablation Study: Does Removing Sequence Alignment Hurt Anchor Case Rankings?

This script compares the 8 anchor case percentile rankings between:
1. Full 3-variable model (hapax + SVM + alignment)
2. Reduced 2-variable model (hapax + SVM only)

This directly addresses whether the ~2% SHAP contribution from sequence alignment
is doing meaningful work for detecting documented influence relationships.

Author: Tarah Wheeler
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
    """Load and prepare data (copied from logistic_regression.py)."""
    project_name = get_project_name()
    main_db_path = f"./projects/{project_name}/db/{project_name}.db"
    main_conn = sqlite3.connect(main_db_path)
    
    query = """
    SELECT 
        cj.source_auth, cj.target_auth, cj.source_text, cj.target_text,
        cj.source_year, cj.target_year, cj.hap_jac_dis, cj.al_jac_dis, cj.pair_id,
        a1.source_filename as source_name, a2.source_filename as target_name,
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
        text_lookup_df.rename(columns={'novel': 'target_novel', 'number': 'target_chapter'}),
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
        left_on=['target_novel', 'target_chapter', 'source_novel_name'],
        right_on=['novel', 'number', 'source_novel_name'],
        how='left'
    )
    
    cols_to_drop = ['target_novel', 'target_chapter', 'source_novel_name', 'novel', 'number']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df = df.dropna(subset=['svm_score'])
    
    return df


def train_and_evaluate(df, feature_names, model_name):
    """
    Train a model and return anchor case percentiles.
    
    Returns dict with:
    - validation_results: list of dicts with label, percentile, rank
    - test_auc: held-out test AUC
    - cv_auc: cross-validation AUC mean
    """
    X = df[feature_names].values
    y = df['same_author'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values,
        test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    # Test AUC
    y_test_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_prob)
    
    # Score cross-author pairs
    cross_author = df[df['same_author'] == 0].copy()
    X_cross = cross_author[feature_names].values
    X_cross_scaled = scaler.transform(X_cross)
    cross_author['influence_prob'] = model.predict_proba(X_cross_scaled)[:, 1]
    
    # Rank
    ranked = cross_author.sort_values('influence_prob', ascending=False).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)
    total_pairs = len(ranked)
    
    # Validate anchor cases
    validation_results = []
    for source_substr, target_substr, label in VALIDATION_CASES:
        case_pairs = cross_author[
            (cross_author['source_author_name'].str.contains(source_substr, case=False, na=False)) &
            (cross_author['target_author_name'].str.contains(target_substr, case=False, na=False))
        ]
        
        if len(case_pairs) == 0:
            validation_results.append({'label': label, 'percentile': None, 'rank': None, 'n_pairs': 0})
            continue
        
        best_pair = case_pairs.loc[case_pairs['influence_prob'].idxmax()]
        best_rank = ranked[ranked['pair_id'] == best_pair['pair_id']]['rank'].values[0]
        percentile = (1 - best_rank / total_pairs) * 100
        
        validation_results.append({
            'label': label,
            'percentile': percentile,
            'rank': best_rank,
            'n_pairs': len(case_pairs),
            'max_prob': best_pair['influence_prob']
        })
    
    return {
        'validation_results': validation_results,
        'test_auc': test_auc,
        'total_cross_author_pairs': total_pairs
    }


def main():
    print("=" * 80)
    print("ABLATION STUDY: Does Removing Sequence Alignment Hurt Anchor Case Rankings?")
    print("=" * 80)
    
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df):,} pairs ({df['same_author'].sum():,} same-author)")
    
    # Full 3-variable model
    print("\n" + "-" * 80)
    print("MODEL 1: Full Sextant (Hapax + SVM + Alignment)")
    print("-" * 80)
    
    features_3var = ['hap_jac_dis', 'svm_score', 'al_jac_dis']
    results_3var = train_and_evaluate(df, features_3var, "3-variable")
    print(f"Test AUC: {results_3var['test_auc']:.4f}")
    
    # Reduced 2-variable model
    print("\n" + "-" * 80)
    print("MODEL 2: Reduced (Hapax + SVM only, NO Alignment)")
    print("-" * 80)
    
    features_2var = ['hap_jac_dis', 'svm_score']
    results_2var = train_and_evaluate(df, features_2var, "2-variable")
    print(f"Test AUC: {results_2var['test_auc']:.4f}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Anchor Case Percentiles")
    print("=" * 80)
    
    total_pairs = results_3var['total_cross_author_pairs']
    print(f"\nTotal cross-author pairs being ranked: {total_pairs:,}")
    
    print(f"\n{'Influence Case':<22} {'3-var rank':>10} {'3-var %':>9} {'2-var rank':>10} {'2-var %':>9} {'Δ rank':>8}")
    print("-" * 80)
    
    total_delta = 0
    improvements = 0
    degradations = 0
    
    for r3, r2 in zip(results_3var['validation_results'], results_2var['validation_results']):
        label = r3['label']
        p3 = r3['percentile']
        p2 = r2['percentile']
        rank3 = r3['rank']
        rank2 = r2['rank']
        
        if p3 is None or p2 is None:
            print(f"{label:<22} {'N/A':>10} {'N/A':>9} {'N/A':>10} {'N/A':>9} {'N/A':>8}")
            continue
        
        delta_pct = p2 - p3  # positive = 2-var is better percentile
        delta_rank = rank3 - rank2  # positive = 2-var has better (lower) rank
        total_delta += delta_pct
        
        if delta_rank > 0:
            improvements += 1  # 2-var ranks higher (better)
        elif delta_rank < 0:
            degradations += 1  # 3-var ranks higher (better)
        
        print(f"{label:<22} {rank3:>10,} {p3:>8.2f}% {rank2:>10,} {p2:>8.2f}% {delta_rank:>+8,}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    avg_p3 = np.mean([r['percentile'] for r in results_3var['validation_results'] if r['percentile'] is not None])
    avg_p2 = np.mean([r['percentile'] for r in results_2var['validation_results'] if r['percentile'] is not None])
    
    print(f"\nTest AUC:")
    print(f"  3-variable model: {results_3var['test_auc']:.4f}")
    print(f"  2-variable model: {results_2var['test_auc']:.4f}")
    print(f"  Δ AUC: {results_2var['test_auc'] - results_3var['test_auc']:+.4f}")
    
    print(f"\nAnchor Case Percentiles:")
    print(f"  3-variable average: {avg_p3:.2f}%")
    print(f"  2-variable average: {avg_p2:.2f}%")
    print(f"  Δ average: {avg_p2 - avg_p3:+.2f}%")
    
    print(f"\n  Cases where 3-var ranks better: {degradations}")
    print(f"  Cases where 2-var ranks better: {improvements}")
    print(f"  Cases with same rank: {8 - degradations - improvements}")
    
    # Above 99th percentile check
    above_99_3var = sum(1 for r in results_3var['validation_results'] if r['percentile'] and r['percentile'] >= 99)
    above_99_2var = sum(1 for r in results_2var['validation_results'] if r['percentile'] and r['percentile'] >= 99)
    
    print(f"\n  3-var: {above_99_3var}/8 cases above 99th percentile")
    print(f"  2-var: {above_99_2var}/8 cases above 99th percentile")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if avg_p3 > avg_p2 + 0.5:
        print("""
Alignment contributes meaningfully to anchor case detection.
Removing it degrades average percentile rankings.
→ Keep alignment in the model.
""")
    elif avg_p2 > avg_p3 + 0.5:
        print("""
Surprisingly, the 2-variable model performs BETTER on anchor cases.
This suggests alignment may be adding noise rather than signal for influence detection.
→ Consider whether alignment is justified on theoretical rather than empirical grounds.
""")
    else:
        print("""
The models perform similarly on anchor cases (difference < 0.5%).
Alignment's ~2% SHAP contribution is real but small.
→ Alignment can be justified on theoretical grounds (capturing direct textual borrowing)
  even if its empirical contribution to ranking is marginal.
""")


if __name__ == "__main__":
    main()
