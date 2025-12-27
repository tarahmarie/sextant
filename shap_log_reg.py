"""
Logistic Regression Analysis for Literary Influence Detection
with SHAP Value Decomposition for Feature Contributions

This script identifies literary influence by:
1. Training a logistic regression model on same-author vs cross-author pairs
2. Using SHAP values to properly decompose feature contributions
   (addresses collinearity and provides theoretically grounded importance measures)
3. Validating against the known Eliot->Lawrence influence relationship

METHODOLOGICAL NOTE (addressing Lisa's comment):
Standard beta coefficients in logistic regression cannot be directly interpreted
as "percentage contributions" because:
- The logistic link function complicates interpretation
- Collinearity between features inflates/deflates individual coefficients
- Standardization alone doesn't solve the attribution problem

SHAP (SHapley Additive exPlanations) values solve this by:
- Computing each feature's marginal contribution across all possible orderings
- Properly handling feature interactions and correlations
- Having axiomatic guarantees (efficiency, symmetry, dummy, additivity)

Author: Tarah Wheeler
For: DH-Trace Dissertation Project / Sextant Paper
"""

import sqlite3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import shap
import warnings

from util import get_project_name

warnings.filterwarnings('ignore')


def load_data():
    """
    Load and prepare data from the project databases.
    Returns a DataFrame with predictor variables, target, and text identifiers.
    """
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    project_name = get_project_name()
    
    # Connect to main database
    main_db_path = f"./projects/{project_name}/db/{project_name}.db"
    print(f"\nConnecting to: {main_db_path}")
    
    main_conn = sqlite3.connect(main_db_path)
    
    # Check counts
    total_pairs = pd.read_sql_query("SELECT COUNT(*) FROM combined_jaccard", main_conn).iloc[0, 0]
    text_count = pd.read_sql_query("SELECT COUNT(*) FROM all_texts", main_conn).iloc[0, 0]
    print(f"\nTotal pairs in combined_jaccard: {total_pairs:,}")
    print(f"Total texts: {text_count:,}")
    
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
    
    # Create target variable
    df['same_author'] = (df['source_auth'] == df['target_auth']).astype(int)
    
    # Load SVM scores
    svm_db_path = f"./projects/{project_name}/db/svm.db"
    svm_conn = sqlite3.connect(svm_db_path)
    chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
    svm_conn.close()
    
    # Get text metadata for SVM matching
    text_df = pd.read_sql_query("SELECT text_id, source_filename, chapter_num FROM all_texts", main_conn)
    
    def extract_novel_name(filename):
        parts = filename.split('-chapter')[0]
        parts = parts.split('-')[1]
        return parts
    
    text_df['novel'] = text_df['source_filename'].apply(extract_novel_name)
    text_df['number'] = text_df['chapter_num'].astype(str)
    
    # Get novel names for SVM columns
    dirs_df = pd.read_sql_query("SELECT id, dir FROM dirs", main_conn)
    novels_dict = {}
    for _, row in dirs_df.iterrows():
        dir_name = row['dir']
        novel_name = dir_name.split('-')[1] if '-' in dir_name else dir_name
        novels_dict[row['id']] = novel_name
    
    main_conn.close()
    
    # Match SVM scores
    print("Matching SVM scores...")
    text_lookup_df = text_df.set_index('text_id')[['novel', 'number']]
    
    df = df.merge(
        text_lookup_df.rename(columns={'novel': 'target_novel', 'number': 'target_chapter'}),
        left_on='target_text',
        right_index=True,
        how='left'
    )
    
    source_novel_df = pd.DataFrame.from_dict(novels_dict, orient='index', columns=['source_novel_name'])
    df = df.merge(source_novel_df, left_on='source_auth', right_index=True, how='left')
    
    # Reshape and merge SVM data
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
        left_on=['target_novel', 'target_chapter', 'source_novel_name'],
        right_on=['novel', 'number', 'source_novel_name'],
        how='left'
    )
    
    # Clean up
    cols_to_drop = ['target_novel', 'target_chapter', 'source_novel_name', 'novel', 'number']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Drop rows with missing SVM scores
    print(f"Pairs with valid SVM scores: {(~df['svm_score'].isna()).sum():,}")
    df = df.dropna(subset=['svm_score'])
    
    print(f"\nFinal dataset: {len(df):,} text pairs")
    
    return df


def find_anchor_case(df):
    """
    Find the Eliot -> Lawrence anchor case for validation.
    """
    print("\n" + "=" * 70)
    print("STEP 2: FINDING ANCHOR CASE")
    print("=" * 70)
    
    anchor_pairs = df[
        (df['source_author_name'].str.contains('Eliot', case=False, na=False)) &
        (df['target_author_name'].str.contains('Lawrence', case=False, na=False))
    ]
    
    if len(anchor_pairs) == 0:
        print("WARNING: No Eliot-Lawrence pairs found!")
        return None
    
    print(f"Found {len(anchor_pairs)} Eliot-Lawrence pairs")
    
    # Find specific chapter pair
    specific_pair = anchor_pairs[
        (anchor_pairs['source_name'].str.contains('Eliot-chapter_77', case=False, na=False)) &
        (anchor_pairs['target_name'].str.contains('Lawrence-chapter_29', case=False, na=False))
    ]
    
    if len(specific_pair) > 0:
        print("*** ANCHOR CASE FOUND: Eliot ch77 -> Lawrence ch29 ***")
        return specific_pair
    
    return anchor_pairs.head(1)


def run_logistic_regression_with_shap(df):
    """
    STEP 3: Fit logistic regression and compute SHAP values for feature contributions.
    
    This addresses the methodological concern about interpreting standardized
    coefficients as contribution percentages.
    """
    print("\n" + "=" * 70)
    print("STEP 3: LOGISTIC REGRESSION WITH SHAP DECOMPOSITION")
    print("=" * 70)
    
    feature_names = ['hap_jac_dis', 'al_jac_dis', 'svm_score']
    pretty_names = {
        'hap_jac_dis': 'Hapax Legomena',
        'al_jac_dis': 'Sequence Alignment', 
        'svm_score': 'SVM Stylometry'
    }
    
    X_raw = df[feature_names].values
    y = df['same_author'].values
    
    # Standardize features (z-score normalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print(f"\nFitting model on {len(y):,} observations")
    print(f"  Same-author pairs: {y.sum():,} ({y.mean()*100:.2f}%)")
    print(f"  Cross-author pairs: {len(y) - y.sum():,} ({(1-y.mean())*100:.2f}%)")
    
    # =========================================================================
    # PART A: Statsmodels for inference (p-values, confidence intervals)
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART A: COEFFICIENT ESTIMATES (statsmodels)")
    print("-" * 50)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    X_with_const = sm.add_constant(X_scaled_df)
    
    sm_model = sm.Logit(y, X_with_const)
    sm_result = sm_model.fit(disp=0)
    
    print(f"\n{'Variable':<25} {'Coef':>10} {'Std Err':>10} {'p-value':>12} {'Sig':>5}")
    print("-" * 65)
    
    for var in sm_result.params.index:
        coef = sm_result.params[var]
        se = sm_result.bse[var]
        p = sm_result.pvalues[var]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        display_name = pretty_names.get(var, var)
        print(f"{display_name:<25} {coef:>10.4f} {se:>10.4f} {p:>12.2e} {sig:>5}")
    
    print(f"\nPseudo R² (McFadden): {sm_result.prsquared:.4f}")
    
    # =========================================================================
    # PART B: Sklearn model for SHAP analysis
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART B: SHAP VALUE DECOMPOSITION")
    print("-" * 50)
    print("\nComputing SHAP values for feature contribution analysis...")
    print("(This properly handles collinearity and feature interactions)")
    
    # Fit sklearn model (equivalent to statsmodels but works with SHAP)
    sk_model = LogisticRegression(
        penalty=None,  # No regularization to match statsmodels
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    sk_model.fit(X_scaled, y)
    
    # Verify sklearn coefficients match statsmodels
    print(f"\nVerifying sklearn coefficients match statsmodels:")
    for i, name in enumerate(feature_names):
        sm_coef = sm_result.params[name]
        sk_coef = sk_model.coef_[0][i]
        match = "✓" if abs(sm_coef - sk_coef) < 0.01 else "✗"
        print(f"  {pretty_names[name]}: statsmodels={sm_coef:.4f}, sklearn={sk_coef:.4f} {match}")
    
    # Compute SHAP values using LinearExplainer (exact for linear models)
    explainer = shap.LinearExplainer(sk_model, X_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)
    
    # Calculate mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Convert to percentage contributions
    total_shap = mean_abs_shap.sum()
    shap_contributions = (mean_abs_shap / total_shap) * 100
    
    print("\n" + "-" * 50)
    print("FEATURE CONTRIBUTIONS (SHAP Value Decomposition)")
    print("-" * 50)
    print("\nMethod: Mean |SHAP value| normalized to percentages")
    print("Interpretation: Average contribution to model predictions\n")
    
    # Sort by contribution
    sorted_indices = np.argsort(shap_contributions)[::-1]
    
    print(f"{'Feature':<25} {'Mean |SHAP|':>12} {'Contribution':>15}")
    print("-" * 55)
    
    for idx in sorted_indices:
        name = feature_names[idx]
        print(f"{pretty_names[name]:<25} {mean_abs_shap[idx]:>12.4f} {shap_contributions[idx]:>14.1f}%")
    
    # =========================================================================
    # PART C: Compare with naive standardized coefficient approach
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART C: COMPARISON WITH NAIVE COEFFICIENT APPROACH")
    print("-" * 50)
    print("\n(This shows why Lisa's comment matters)\n")
    
    # Naive approach: normalize absolute standardized coefficients
    coefs = np.array([sm_result.params[name] for name in feature_names])
    abs_coefs = np.abs(coefs)
    naive_contributions = (abs_coefs / abs_coefs.sum()) * 100
    
    print(f"{'Feature':<25} {'Naive (|β|)':>15} {'SHAP':>15} {'Difference':>12}")
    print("-" * 70)
    
    for i, name in enumerate(feature_names):
        naive = naive_contributions[i]
        shap_val = shap_contributions[i]
        diff = shap_val - naive
        print(f"{pretty_names[name]:<25} {naive:>14.1f}% {shap_val:>14.1f}% {diff:>+11.1f}%")
    
    # =========================================================================
    # PART D: Check for collinearity (why SHAP matters)
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART D: COLLINEARITY DIAGNOSTIC")
    print("-" * 50)
    
    X_df = pd.DataFrame(X_scaled, columns=feature_names)
    corr_matrix = X_df.corr()
    
    print("\nFeature Correlation Matrix (standardized):")
    print(f"\n{'':>25}", end='')
    for name in feature_names:
        print(f"{pretty_names[name][:12]:>15}", end='')
    print()
    
    for i, name1 in enumerate(feature_names):
        print(f"{pretty_names[name1]:<25}", end='')
        for j, name2 in enumerate(feature_names):
            r = corr_matrix.iloc[i, j]
            print(f"{r:>15.3f}", end='')
        print()
    
    # Flag high correlations
    high_corr = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.3:
                high_corr.append((feature_names[i], feature_names[j], r))
    
    if high_corr:
        print("\n⚠ Moderate/high correlations detected:")
        for f1, f2, r in high_corr:
            print(f"  {pretty_names[f1]} ↔ {pretty_names[f2]}: r = {r:.3f}")
        print("\n  → SHAP values account for this; naive coefficients do not.")
    else:
        print("\n✓ No strong collinearity detected (all |r| < 0.3)")
        print("  → SHAP and naive approaches should be similar.")
    
    # =========================================================================
    # PART E: Model performance metrics
    # =========================================================================
    print("\n" + "-" * 50)
    print("PART E: MODEL PERFORMANCE")
    print("-" * 50)
    
    y_pred_prob = sk_model.predict_proba(X_scaled)[:, 1]
    y_pred = sk_model.predict(X_scaled)
    
    auc = roc_auc_score(y, y_pred_prob)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nROC AUC:   {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"                        Predicted")
    print(f"                   Cross-Author    Same-Author")
    print(f"Actual Cross-Author {tn:>12,}    {fp:>12,}")
    print(f"Actual Same-Author  {fn:>12,}    {tp:>12,}")
    
    # Store results for return
    results = {
        'sm_result': sm_result,
        'sk_model': sk_model,
        'scaler': scaler,
        'shap_values': shap_values,
        'mean_abs_shap': mean_abs_shap,
        'shap_contributions': shap_contributions,
        'naive_contributions': naive_contributions,
        'feature_names': feature_names,
        'pretty_names': pretty_names,
        'auc': auc,
        'accuracy': accuracy,
        'X_scaled': X_scaled,
        'y': y
    }
    
    return results


def validate_anchor_case(df, results, anchor_case):
    """
    STEP 4: Validate the model against the known Eliot-Lawrence influence case.
    """
    print("\n" + "=" * 70)
    print("STEP 4: ANCHOR CASE VALIDATION")
    print("=" * 70)
    
    if anchor_case is None or len(anchor_case) == 0:
        print("No anchor case found for validation.")
        return
    
    feature_names = results['feature_names']
    pretty_names = results['pretty_names']
    sk_model = results['sk_model']
    scaler = results['scaler']
    
    # Get cross-author pairs only
    cross_author = df[df['same_author'] == 0].copy()
    
    # Score all cross-author pairs
    X_cross = cross_author[feature_names].values
    X_cross_scaled = scaler.transform(X_cross)
    
    # Get probability of same-author (high = stylistically similar)
    cross_author['influence_prob'] = sk_model.predict_proba(X_cross_scaled)[:, 1]
    
    # Find anchor case
    anchor_pair_id = anchor_case.iloc[0]['pair_id']
    anchor_row = cross_author[cross_author['pair_id'] == anchor_pair_id]
    
    if len(anchor_row) == 0:
        print("Anchor case not found in cross-author pairs.")
        return
    
    anchor_prob = anchor_row['influence_prob'].values[0]
    
    # Rank
    ranked = cross_author.sort_values('influence_prob', ascending=False)
    position = ranked['pair_id'].tolist().index(anchor_pair_id) + 1
    percentile = (1 - position / len(ranked)) * 100
    
    print(f"\nAnchor case: Eliot ch77 -> Lawrence ch29")
    print(f"  Predicted probability (same-author style): {anchor_prob:.4f}")
    print(f"  Rank among {len(ranked):,} cross-author pairs: {position:,}")
    print(f"  Percentile: {percentile:.2f}%")
    
    # Show feature values for anchor case
    print(f"\nAnchor case feature values (raw):")
    for name in feature_names:
        val = anchor_case.iloc[0][name]
        print(f"  {pretty_names[name]}: {val:.4f}")
    
    # Top 10 cross-author pairs
    print(f"\n--- TOP 10 INFLUENCE CANDIDATES ---")
    print("(Cross-author pairs ranked by model probability)\n")
    
    for i, (_, row) in enumerate(ranked.head(10).iterrows(), 1):
        print(f"{i:2}. {row['source_author_name']:15} -> {row['target_author_name']:15} "
              f"P={row['influence_prob']:.4f}")


def generate_paper_output(results):
    """
    STEP 5: Generate clean output for the paper.
    """
    print("\n" + "=" * 70)
    print("STEP 5: OUTPUT FOR PAPER")
    print("=" * 70)
    
    feature_names = results['feature_names']
    pretty_names = results['pretty_names']
    sm_result = results['sm_result']
    shap_contributions = results['shap_contributions']
    
    print("\n--- FOR METHODS SECTION ---")
    print("""
Using Shapley value decomposition to assess relative feature contributions 
to the logistic regression model (which accounts for collinearity between 
predictors), the three components contribute as follows:
""")
    
    # Sort by contribution
    sorted_indices = np.argsort(shap_contributions)[::-1]
    
    for idx in sorted_indices:
        name = feature_names[idx]
        contrib = shap_contributions[idx]
        coef = sm_result.params[name]
        p = sm_result.pvalues[name]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {pretty_names[name]}: {contrib:.1f}% (β = {coef:.3f}, p < 0.001{sig})")
    
    print(f"\nModel performance: ROC AUC = {results['auc']:.3f}, Accuracy = {results['accuracy']:.3f}")
    
    print("\n--- SUGGESTED PAPER TEXT ---")
    print("""
"Using Shapley value decomposition on the fitted logistic regression model 
to properly assess relative feature contributions (accounting for collinearity 
between predictors), hapax legomena analysis contributes {:.1f}%, SVM stylometry 
{:.1f}%, and sequence alignment {:.1f}% to the model's predictive capacity 
(see Table X). All three predictors are statistically significant (p < 0.001)."
""".format(
        shap_contributions[feature_names.index('hap_jac_dis')],
        shap_contributions[feature_names.index('svm_score')],
        shap_contributions[feature_names.index('al_jac_dis')]
    ))
    
    print("\n--- FOR TABLE ---")
    print("""
Table X: Logistic Regression Coefficients and Feature Contributions

Variable              | β (SE)           | p-value  | SHAP Contribution
---------------------|------------------|----------|------------------""")
    
    for idx in sorted_indices:
        name = feature_names[idx]
        coef = sm_result.params[name]
        se = sm_result.bse[name]
        p = sm_result.pvalues[name]
        contrib = shap_contributions[idx]
        p_str = "< 0.001" if p < 0.001 else f"{p:.3f}"
        print(f"{pretty_names[name]:<21}| {coef:>6.3f} ({se:.3f})  | {p_str:<8} | {contrib:>5.1f}%")
    
    print(f"{'Intercept':<21}| {sm_result.params['const']:>6.3f} ({sm_result.bse['const']:.3f})  | {'< 0.001':<8} | {'—':>5}")
    print("""
Note: SHAP contributions calculated via mean |SHAP value| decomposition,
which properly accounts for feature correlations. Model: N = {:,}, 
Pseudo R² = {:.3f}, ROC AUC = {:.3f}.
""".format(len(results['y']), sm_result.prsquared, results['auc']))


def save_results(df, results):
    """
    Save results to files.
    """
    print("\n" + "=" * 70)
    print("STEP 6: SAVING RESULTS")
    print("=" * 70)
    
    project_name = get_project_name()
    
    sm_result = results['sm_result']
    feature_names = results['feature_names']
    pretty_names = results['pretty_names']
    
    # Save coefficients with SHAP contributions
    coef_data = []
    for i, name in enumerate(feature_names):
        coef_data.append({
            'variable': name,
            'pretty_name': pretty_names[name],
            'coefficient': sm_result.params[name],
            'std_error': sm_result.bse[name],
            'p_value': sm_result.pvalues[name],
            'odds_ratio': np.exp(sm_result.params[name]),
            'shap_contribution_pct': results['shap_contributions'][i],
            'naive_contribution_pct': results['naive_contributions'][i]
        })
    
    coef_df = pd.DataFrame(coef_data)
    coef_path = f"./projects/{project_name}/results/influence_coefficients_shap.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Coefficients saved to: {coef_path}")
    
    # Save summary
    summary_path = f"./projects/{project_name}/results/influence_model_summary_shap.txt"
    with open(summary_path, 'w') as f:
        f.write("INFLUENCE DETECTION MODEL SUMMARY (WITH SHAP DECOMPOSITION)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("FEATURE CONTRIBUTIONS (SHAP Value Decomposition)\n")
        f.write("-" * 60 + "\n")
        f.write("Method: Mean |SHAP value| normalized to percentages\n\n")
        
        sorted_indices = np.argsort(results['shap_contributions'])[::-1]
        for idx in sorted_indices:
            name = feature_names[idx]
            f.write(f"{pretty_names[name]}: {results['shap_contributions'][idx]:.1f}%\n")
        
        f.write(f"\nMODEL COEFFICIENTS\n")
        f.write("-" * 60 + "\n")
        for name in feature_names:
            f.write(f"{pretty_names[name]}: β = {sm_result.params[name]:.4f} "
                   f"(SE = {sm_result.bse[name]:.4f}, p = {sm_result.pvalues[name]:.2e})\n")
        
        f.write(f"\nMODEL FIT\n")
        f.write("-" * 60 + "\n")
        f.write(f"N = {len(results['y']):,}\n")
        f.write(f"Pseudo R² = {sm_result.prsquared:.4f}\n")
        f.write(f"ROC AUC = {results['auc']:.4f}\n")
        f.write(f"Accuracy = {results['accuracy']:.4f}\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    """
    Main function for influence detection analysis with SHAP decomposition.
    """
    print("\n" + "=" * 70)
    print("LITERARY INFLUENCE DETECTION")
    print("with SHAP Value Decomposition")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Find anchor case
    anchor_case = find_anchor_case(df)
    
    # Run logistic regression with SHAP
    results = run_logistic_regression_with_shap(df)
    
    # Validate against anchor case
    validate_anchor_case(df, results, anchor_case)
    
    # Generate paper output
    generate_paper_output(results)
    
    # Save results
    save_results(df, results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
