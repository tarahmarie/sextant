"""
anchor_case_analysis.py

Extracts data for the Eliot-Lawrence anchor case and generates graphics
for the conference paper.

Uses the SAME methodology as logistic_regression.py:
- Loads all three variables (hapax, alignment, SVM)
- Z-score normalizes them
- Applies grid search optimal weights OR logistic regression coefficients
- Computes percentiles among cross-author pairs

Run from your dh-trace root directory:
    python anchor_case_analysis.py

Outputs:
- anchor_case_results.txt: Summary statistics
- variable_contribution_chart.png: Bar chart of variable contributions
- percentile_distribution.png: Histogram showing where anchor case falls
- all_eliot_lawrence_pairs.csv: All pairs ranked by influence score
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import os
import sys

# Configuration
PROJECT = "eltec-100"
MAIN_DB = f"./projects/{PROJECT}/db/{PROJECT}.db"
SVM_DB = f"./projects/{PROJECT}/db/svm.db"
OUTPUT_DIR = "./anchor_case_output"

# ELTEC IDs for anchor case texts
MIDDLEMARCH_ID = "1872-ENG18721"  # Middlemarch (1872)
WOMEN_IN_LOVE_ID = "1920-ENG19200"  # Women in Love (1920)


def load_data():
    """
    Load and prepare data from the project databases.
    Mirrors the methodology in logistic_regression.py
    """
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    # Connect to main database
    print(f"\nConnecting to: {MAIN_DB}")
    main_conn = sqlite3.connect(MAIN_DB)
    
    # Load combined_jaccard with text names
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
    print(f"Loaded {len(df):,} text pairs")
    
    # Temporal filter
    df = df[df['source_year'] <= df['target_year']].copy()
    print(f"After temporal filter: {len(df):,} pairs")
    
    # Create same_author target
    df['same_author'] = (df['source_auth'] == df['target_auth']).astype(int)
    
    # Load SVM scores
    print(f"\nLoading SVM scores from: {SVM_DB}")
    svm_conn = sqlite3.connect(SVM_DB)
    chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
    svm_conn.close()
    
    # Get text metadata for SVM matching
    text_query = "SELECT text_id, source_filename, chapter_num FROM all_texts"
    text_df = pd.read_sql_query(text_query, main_conn)
    
    def extract_novel_name(filename):
        parts = filename.split('-chapter')[0]
        parts = parts.split('-')[1]
        return parts
    
    text_df['novel'] = text_df['source_filename'].apply(extract_novel_name)
    text_df['number'] = text_df['chapter_num'].astype(str)
    
    # Get novel names for SVM columns
    dirs_query = "SELECT id, dir FROM dirs"
    dirs_df = pd.read_sql_query(dirs_query, main_conn)
    novels_dict = {}
    for _, row in dirs_df.iterrows():
        dir_name = row['dir']
        novel_name = dir_name.split('-')[1] if '-' in dir_name else dir_name
        novels_dict[row['id']] = novel_name
    
    main_conn.close()
    
    # Match SVM scores
    print("Matching SVM scores (this may take a minute)...")
    
    text_lookup_df = text_df.set_index('text_id')[['novel', 'number']]
    
    df = df.merge(
        text_lookup_df.rename(columns={'novel': 'target_novel', 'number': 'target_chapter_num'}),
        left_on='target_text',
        right_index=True,
        how='left'
    )
    
    source_novel_df = pd.DataFrame.from_dict(novels_dict, orient='index', columns=['source_novel_name'])
    df = df.merge(
        source_novel_df,
        left_on='source_auth',
        right_index=True,
        how='left'
    )
    
    # Reshape chapter_df for merge
    id_vars = ['novel', 'number']
    value_vars = [col for col in chapter_df.columns if col not in id_vars]
    chapter_melted = chapter_df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='source_novel_name',
        value_name='svm_score'
    )
    chapter_melted['number'] = chapter_melted['number'].astype(str)
    
    df = df.merge(
        chapter_melted,
        left_on=['target_novel', 'target_chapter_num', 'source_novel_name'],
        right_on=['novel', 'number', 'source_novel_name'],
        how='left'
    )
    
    # Clean up
    cols_to_drop = ['target_novel', 'target_chapter_num', 'source_novel_name', 'novel', 'number']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    print(f"Pairs with valid SVM scores: {(~df['svm_score'].isna()).sum():,}")
    df = df.dropna(subset=['svm_score'])
    
    # Z-score normalize all three variables
    print("\nZ-score normalizing variables...")
    for col in ['hap_jac_dis', 'al_jac_dis', 'svm_score']:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_raw'] = df[col]
        df[col] = (df[col] - mean) / std
    
    print(f"\nFinal dataset: {len(df):,} text pairs")
    
    return df


def optimize_weights(df):
    """
    Grid search for optimal weights (same as logistic_regression.py)
    """
    print("\n" + "=" * 70)
    print("STEP 2: OPTIMIZING WEIGHTS (Grid Search)")
    print("=" * 70)
    
    best_auc = 0
    best_weights = None
    
    step = 0.05
    y_true = df['same_author'].values
    
    for hap_w in np.arange(0.0, 1.01, step):
        for al_w in np.arange(0.0, 1.01 - hap_w, step):
            svm_w = round(1.0 - hap_w - al_w, 2)
            if svm_w < 0:
                continue
            
            scores = (
                df['hap_jac_dis'].values * hap_w +
                df['al_jac_dis'].values * al_w +
                df['svm_score'].values * svm_w
            )
            
            try:
                auc = roc_auc_score(y_true, scores)
                if auc < 0.5:
                    auc = 1 - auc
            except:
                auc = 0.5
            
            if auc > best_auc:
                best_auc = auc
                best_weights = (round(hap_w, 2), round(al_w, 2), round(svm_w, 2))
    
    print(f"\nOptimal weights:")
    print(f"  Hapax:     {best_weights[0]:.2f} ({best_weights[0]*100:.0f}%)")
    print(f"  Alignment: {best_weights[1]:.2f} ({best_weights[1]*100:.0f}%)")
    print(f"  SVM:       {best_weights[2]:.2f} ({best_weights[2]*100:.0f}%)")
    print(f"  ROC AUC:   {best_auc:.4f}")
    
    return best_weights, best_auc


def run_logistic_regression(df):
    """
    Run logistic regression for coefficient comparison
    """
    print("\n" + "=" * 70)
    print("STEP 3: LOGISTIC REGRESSION")
    print("=" * 70)
    
    predictors = ['hap_jac_dis', 'al_jac_dis', 'svm_score']
    X = df[predictors].copy()
    y = df['same_author'].copy()
    X_with_const = sm.add_constant(X)
    
    model = sm.Logit(y, X_with_const)
    result = model.fit(disp=0)
    
    print("\nCoefficients:")
    for var in predictors:
        coef = result.params[var]
        p = result.pvalues[var]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {var:<15} {coef:>10.4f} (p={p:.2e}) {sig}")
    
    # Normalized contributions
    coefs = result.params[predictors]
    abs_coefs = np.abs(coefs)
    total = abs_coefs.sum()
    
    print("\nRelative contributions (normalized |coefficients|):")
    contributions = {}
    for var in predictors:
        pct = abs(result.params[var]) / total * 100
        contributions[var] = pct
        if var == 'hap_jac_dis':
            label = "Hapax (vocabulary)"
        elif var == 'al_jac_dis':
            label = "Alignment (phrasing)"
        else:
            label = "SVM (style)"
        print(f"  {label:<25} {pct:.1f}%")
    
    print(f"\nROC AUC: {roc_auc_score(y, result.predict(X_with_const)):.4f}")
    
    return result, contributions


def compute_influence_scores(df, weights):
    """
    Compute influence scores using the optimal weights
    """
    hap_w, al_w, svm_w = weights
    
    df['influence_score'] = (
        df['hap_jac_dis'] * hap_w +
        df['al_jac_dis'] * al_w +
        df['svm_score'] * svm_w
    )
    
    return df


def find_eliot_lawrence_pairs(df):
    """
    Find all Eliot -> Lawrence pairs (Middlemarch -> Women in Love)
    """
    print("\n" + "=" * 70)
    print("STEP 4: FINDING ELIOT → LAWRENCE PAIRS")
    print("=" * 70)
    
    # Find pairs by ELTEC ID
    eliot_mask = df['source_name'].str.contains(MIDDLEMARCH_ID, na=False)
    lawrence_mask = df['target_name'].str.contains(WOMEN_IN_LOVE_ID, na=False)
    
    el_pairs = df[eliot_mask & lawrence_mask].copy()
    
    print(f"\nFound {len(el_pairs)} Middlemarch → Women in Love pairs")
    
    if len(el_pairs) == 0:
        print("WARNING: No pairs found!")
        return None, None
    
    # Get cross-author pairs for percentile calculation
    cross_author = df[df['same_author'] == 0]
    
    # Calculate percentile for each pair
    el_pairs['percentile'] = el_pairs['influence_score'].apply(
        lambda x: (cross_author['influence_score'] < x).mean() * 100
    )
    
    # Sort by influence score
    el_pairs = el_pairs.sort_values('influence_score', ascending=False)
    
    return el_pairs, cross_author


def print_top_pairs(el_pairs, n=20):
    """
    Print top Eliot-Lawrence pairs with percentiles
    """
    print(f"\n--- TOP {n} ELIOT → LAWRENCE CHAPTER PAIRS ---\n")
    
    for i, (_, row) in enumerate(el_pairs.head(n).iterrows(), 1):
        src_ch = row['source_chapter']
        tgt_ch = row['target_chapter']
        print(f"{i:3}. Middlemarch ch{src_ch:>2} → WiL ch{tgt_ch:>2}  "
              f"Score: {row['influence_score']:.4f}  "
              f"Pctl: {row['percentile']:.2f}%  "
              f"Hap: {1-row['hap_jac_dis_raw']:.4f}  "
              f"Align: {1-row['al_jac_dis_raw']:.4f}  "
              f"SVM: {row['svm_score_raw']:.4f}")


def generate_variable_contribution_chart(contributions, output_dir):
    """
    Generate bar chart showing variable contributions from logistic regression
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['Hapax Legomena\n(Vocabulary)', 'SVM\n(Style)', 'Alignment\n(Phrasing)']
    values = [
        contributions['hap_jac_dis'],
        contributions['svm_score'],
        contributions['al_jac_dis']
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Sort by value
    sorted_data = sorted(zip(values, labels, colors), reverse=True)
    values, labels, colors = zip(*sorted_data)
    
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Contribution to Model (%)', fontsize=12)
    ax.set_title('Variable Contributions to Influence Detection\n(Logistic Regression Coefficients)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variable_contribution_chart.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'variable_contribution_chart.pdf'))
    print(f"\nSaved variable contribution chart")
    plt.close()


def generate_percentile_distribution(el_pairs, cross_author, output_dir):
    """
    Generate histogram showing where top Eliot-Lawrence pairs fall
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram of all cross-author scores
    ax.hist(cross_author['influence_score'], bins=50, 
            color='lightsteelblue', edgecolor='steelblue',
            alpha=0.7, label='All cross-author pairs')
    
    # Mark top Eliot-Lawrence pair
    top_pair = el_pairs.iloc[0]
    ax.axvline(x=top_pair['influence_score'], color='crimson', linestyle='--', linewidth=2.5,
               label=f'Top Eliot→Lawrence pair ({top_pair["percentile"]:.1f}th pctl)')
    
    ax.set_xlabel('Influence Score (z-normalized, weighted)', fontsize=12)
    ax.set_ylabel('Number of Text Pairs', fontsize=12)
    ax.set_title('Distribution of Influence Scores: Cross-Author Pairs\nMiddlemarch → Women in Love', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'percentile_distribution.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'percentile_distribution.pdf'))
    print(f"Saved percentile distribution chart")
    plt.close()


def save_results(el_pairs, contributions, weights, output_dir):
    """
    Save results to files
    """
    # Save all pairs
    el_pairs.to_csv(os.path.join(output_dir, 'all_eliot_lawrence_pairs.csv'), index=False)
    print(f"Saved {len(el_pairs)} pairs to CSV")
    
    # Save summary
    with open(os.path.join(output_dir, 'anchor_case_results.txt'), 'w') as f:
        f.write("ANCHOR CASE ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write("Pairing: George Eliot (Middlemarch) → D.H. Lawrence (Women in Love)\n\n")
        
        f.write("OPTIMAL WEIGHTS (Grid Search, max ROC AUC):\n")
        f.write(f"  Hapax:     {weights[0]:.2f} ({weights[0]*100:.0f}%)\n")
        f.write(f"  Alignment: {weights[1]:.2f} ({weights[1]*100:.0f}%)\n")
        f.write(f"  SVM:       {weights[2]:.2f} ({weights[2]*100:.0f}%)\n\n")
        
        f.write("LOGISTIC REGRESSION CONTRIBUTIONS:\n")
        f.write(f"  Hapax (vocabulary):   {contributions['hap_jac_dis']:.1f}%\n")
        f.write(f"  Alignment (phrasing): {contributions['al_jac_dis']:.1f}%\n")
        f.write(f"  SVM (style):          {contributions['svm_score']:.1f}%\n\n")
        
        f.write("TOP 10 ELIOT → LAWRENCE PAIRS:\n")
        f.write("-" * 50 + "\n")
        for i, (_, row) in enumerate(el_pairs.head(10).iterrows(), 1):
            f.write(f"{i:2}. Middlemarch ch{row['source_chapter']:>2} → WiL ch{row['target_chapter']:>2}  "
                   f"Pctl: {row['percentile']:.2f}%\n")
    
    print(f"Saved summary to anchor_case_results.txt")


def main():
    print("=" * 70)
    print("ANCHOR CASE ANALYSIS: Eliot → Lawrence")
    print("Using Full Model (Hapax + Alignment + SVM)")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_data()
    
    # Optimize weights
    weights, auc = optimize_weights(df)
    
    # Run logistic regression
    result, contributions = run_logistic_regression(df)
    
    # Compute influence scores
    df = compute_influence_scores(df, weights)
    
    # Find Eliot-Lawrence pairs
    el_pairs, cross_author = find_eliot_lawrence_pairs(df)
    
    if el_pairs is not None:
        # Print top pairs
        print_top_pairs(el_pairs, n=20)
        
        # Generate charts
        generate_variable_contribution_chart(contributions, OUTPUT_DIR)
        generate_percentile_distribution(el_pairs, cross_author, OUTPUT_DIR)
        
        # Save results
        save_results(el_pairs, contributions, weights, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()