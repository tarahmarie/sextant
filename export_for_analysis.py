"""
export_for_analysis.py

Export targeted subsets of the DH-Trace database for external analysis.
Run from the sextant root directory.

Usage:
    python export_for_analysis.py [project_name]
    
If no project name provided, will prompt for selection.

Note: This script handles the split database architecture where:
- Main db (eltec-100.db): combined_jaccard with hapax and alignment scores
- SVM db (svm.db): chapter_assessments with SVM confidence scores (wide format)

The SVM table has one row per chapter, with columns for each novel showing
how similar that chapter is to that novel's style.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime


def get_project_name():
    """Get project name from command line or prompt."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    
    projects_dir = './projects'
    if not os.path.exists(projects_dir):
        print("Error: No projects directory found. Run from sextant root.")
        sys.exit(1)
    
    projects = [d for d in os.listdir(projects_dir) 
                if os.path.isdir(os.path.join(projects_dir, d))]
    
    if not projects:
        print("Error: No projects found.")
        sys.exit(1)
    
    print("\nAvailable projects:")
    for i, p in enumerate(projects, 1):
        print(f"  {i}) {p}")
    
    choice = input("\nSelect project number: ")
    try:
        return projects[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)


def connect_dbs(project_name):
    """Connect to both the main database and SVM database."""
    main_db_path = f'./projects/{project_name}/db/{project_name}.db'
    svm_db_path = f'./projects/{project_name}/db/svm.db'
    
    if not os.path.exists(main_db_path):
        print(f"Error: Main database not found at {main_db_path}")
        sys.exit(1)
    
    main_conn = sqlite3.connect(main_db_path)
    print(f"Connected to main db: {main_db_path}")
    
    svm_conn = None
    if os.path.exists(svm_db_path):
        svm_conn = sqlite3.connect(svm_db_path)
        print(f"Connected to SVM db: {svm_db_path}")
    else:
        print(f"Warning: SVM database not found at {svm_db_path}")
        print("  SVM scores will not be included in exports.")
    
    return main_conn, svm_conn


def load_svm_scores_long_format(svm_conn):
    """
    Load SVM scores and convert from wide to long format.
    
    The chapter_assessments table has:
    - novel: the chapter's novel (e.g., 'ENG18721—Eliot')
    - number: chapter number
    - columns for each novel: SVM similarity score
    
    We convert this to long format for easier merging with pairs.
    """
    if svm_conn is None:
        return None
    
    print("Loading SVM scores from wide format...")
    
    # Load the wide table
    query = "SELECT * FROM chapter_assessments"
    wide_df = pd.read_sql_query(query, svm_conn)
    
    print(f"  Loaded {len(wide_df)} chapters with {len(wide_df.columns)-2} novel scores each")
    
    # Create a chapter identifier
    wide_df['chapter_id'] = wide_df['novel'] + '-chapter_' + wide_df['number'].astype(str)
    
    # Get the novel columns (all except 'novel', 'number', 'chapter_id')
    novel_cols = [c for c in wide_df.columns if c not in ['novel', 'number', 'chapter_id']]
    
    # Melt to long format
    long_df = wide_df.melt(
        id_vars=['chapter_id', 'novel', 'number'],
        value_vars=novel_cols,
        var_name='target_novel',
        value_name='svm_score'
    )
    
    print(f"  Converted to long format: {len(long_df)} chapter-novel pairs")
    
    return long_df


def get_basic_stats(main_conn):
    """Get basic statistics about the database."""
    cursor = main_conn.cursor()
    
    stats = {}
    
    cursor.execute("SELECT COUNT(*) FROM combined_jaccard")
    stats['total_pairs'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM all_texts")
    stats['total_texts'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT author_name) FROM authors")
    stats['total_authors'] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM combined_jaccard WHERE source_auth = target_auth")
    stats['same_author_pairs'] = cursor.fetchone()[0]
    
    stats['cross_author_pairs'] = stats['total_pairs'] - stats['same_author_pairs']
    
    cursor.execute("SELECT COUNT(*) FROM combined_jaccard WHERE al_jac_dis IS NOT NULL AND al_jac_dis > 0")
    stats['pairs_with_alignments'] = cursor.fetchone()[0]
    
    return stats


def load_base_pairs(main_conn, limit=None):
    """Load base pair data from combined_jaccard with text names."""
    
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    query = f"""
    SELECT 
        cj.source_text,
        cj.target_text,
        t1.source_filename as source_filename,
        t2.source_filename as target_filename,
        a1.author_name as source_author,
        a2.author_name as target_author,
        cj.source_year,
        cj.target_year,
        cj.hap_jac_dis,
        cj.al_jac_dis,
        cj.source_auth,
        cj.target_auth
    FROM combined_jaccard cj
    JOIN all_texts t1 ON cj.source_text = t1.text_id
    JOIN all_texts t2 ON cj.target_text = t2.text_id
    JOIN authors a1 ON cj.source_auth = a1.id
    JOIN authors a2 ON cj.target_auth = a2.id
    WHERE cj.source_auth != cj.target_auth
    {limit_clause}
    """
    
    df = pd.read_sql_query(query, main_conn)
    return df


def merge_svm_scores(pairs_df, svm_long_df):
    """
    Merge SVM scores into pairs dataframe.
    
    For each pair (source_chapter, target_chapter), we want:
    - How much does source_chapter score on target_novel's style?
    
    This captures: "Does the source chapter write like the target's author?"
    """
    if svm_long_df is None:
        pairs_df['svm_score'] = np.nan
        return pairs_df
    
    print("Merging SVM scores with pairs...")
    
    # Extract chapter identifiers from filenames
    # Format: "1872-ENG18721—Eliot-chapter_77" -> need to match with SVM table
    
    # The SVM table has chapter_id like "ENG18721—Eliot-chapter_77"
    # We need to extract novel code from target to look up the score
    
    # Create lookup key for source chapter
    pairs_df['source_chapter_key'] = pairs_df['source_filename'].apply(
        lambda x: '-'.join(x.split('-')[1:]) if '-' in x else x
    )
    
    # Target novel (the style we're checking against)
    pairs_df['target_novel_key'] = pairs_df['target_filename'].apply(
        lambda x: x.split('-chapter_')[0].split('-', 1)[1] if '-chapter_' in x else x
    )
    
    # Create lookup from SVM long format
    svm_long_df['source_key'] = svm_long_df['novel'] + '-chapter_' + svm_long_df['number'].astype(str)
    
    # Merge: for each source chapter, get its score on the target novel's style
    svm_lookup = svm_long_df[['source_key', 'target_novel', 'svm_score']].copy()
    
    merged = pairs_df.merge(
        svm_lookup,
        left_on=['source_chapter_key', 'target_novel_key'],
        right_on=['source_key', 'target_novel'],
        how='left'
    )
    
    # Clean up
    merged = merged.drop(columns=['source_key', 'target_novel', 'source_chapter_key', 'target_novel_key'], errors='ignore')
    
    matched = merged['svm_score'].notna().sum()
    print(f"  Matched SVM scores for {matched:,} of {len(merged):,} pairs ({100*matched/len(merged):.1f}%)")
    
    return merged


def normalize_scores(df):
    """Z-score normalize the three variables."""
    for col in ['hap_jac_dis', 'al_jac_dis', 'svm_score']:
        if col in df.columns and df[col].notna().any():
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[f'{col}_z'] = (df[col] - mean) / std
            else:
                df[f'{col}_z'] = 0
    return df


def compute_influence_score(df):
    """
    Compute influence score using logistic regression normalized coefficients.
    From TF-IDF run: Hapax=79.2%, SVM=10.8%, Alignment=10.1%
    """
    # Use z-scored versions if available, otherwise raw
    hap = df.get('hap_jac_dis_z', df.get('hap_jac_dis', 0))
    al = df.get('al_jac_dis_z', df.get('al_jac_dis', 0))
    svm = df.get('svm_score_z', df.get('svm_score', 0))
    
    # Fill NaN with 0 for calculation
    hap = pd.to_numeric(hap, errors='coerce').fillna(0)
    al = pd.to_numeric(al, errors='coerce').fillna(0)
    svm = pd.to_numeric(svm, errors='coerce').fillna(0)
    
    df['influence_score'] = 0.792 * hap + 0.101 * al + 0.108 * svm
    return df


def export_high_alignment_pairs(main_conn, svm_conn, output_dir, limit=500):
    """Export cross-author pairs with highest alignment scores."""
    
    query = """
    SELECT 
        t1.source_filename as source_filename,
        t2.source_filename as target_filename,
        a1.author_name as source_author,
        a2.author_name as target_author,
        cj.source_year,
        cj.target_year,
        cj.hap_jac_dis,
        cj.al_jac_dis
    FROM combined_jaccard cj
    JOIN all_texts t1 ON cj.source_text = t1.text_id
    JOIN all_texts t2 ON cj.target_text = t2.text_id
    JOIN authors a1 ON cj.source_auth = a1.id
    JOIN authors a2 ON cj.target_auth = a2.id
    WHERE cj.source_auth != cj.target_auth
    AND cj.al_jac_dis IS NOT NULL
    ORDER BY cj.al_jac_dis DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, main_conn, params=(limit,))
    output_path = os.path.join(output_dir, 'high_alignment_pairs.csv')
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} high-alignment pairs to {output_path}")
    return df


def export_high_hapax_pairs(main_conn, svm_conn, output_dir, limit=500):
    """Export cross-author pairs with highest hapax overlap."""
    
    query = """
    SELECT 
        t1.source_filename as source_filename,
        t2.source_filename as target_filename,
        a1.author_name as source_author,
        a2.author_name as target_author,
        cj.source_year,
        cj.target_year,
        cj.hap_jac_dis,
        cj.al_jac_dis
    FROM combined_jaccard cj
    JOIN all_texts t1 ON cj.source_text = t1.text_id
    JOIN all_texts t2 ON cj.target_text = t2.text_id
    JOIN authors a1 ON cj.source_auth = a1.id
    JOIN authors a2 ON cj.target_auth = a2.id
    WHERE cj.source_auth != cj.target_auth
    ORDER BY cj.hap_jac_dis DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, main_conn, params=(limit,))
    output_path = os.path.join(output_dir, 'high_hapax_pairs.csv')
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} high-hapax pairs to {output_path}")
    return df


def export_double_signal_pairs(main_conn, svm_conn, output_dir, limit=500):
    """Export cross-author pairs strong on BOTH hapax AND alignment."""
    
    query = """
    SELECT 
        t1.source_filename as source_filename,
        t2.source_filename as target_filename,
        a1.author_name as source_author,
        a2.author_name as target_author,
        cj.source_year,
        cj.target_year,
        cj.hap_jac_dis,
        cj.al_jac_dis,
        (cj.hap_jac_dis + cj.al_jac_dis) as combined_score
    FROM combined_jaccard cj
    JOIN all_texts t1 ON cj.source_text = t1.text_id
    JOIN all_texts t2 ON cj.target_text = t2.text_id
    JOIN authors a1 ON cj.source_auth = a1.id
    JOIN authors a2 ON cj.target_auth = a2.id
    WHERE cj.source_auth != cj.target_auth
    AND cj.hap_jac_dis > 0
    AND cj.al_jac_dis > 0
    ORDER BY (cj.hap_jac_dis + cj.al_jac_dis) DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, main_conn, params=(limit,))
    output_path = os.path.join(output_dir, 'double_signal_pairs.csv')
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} double-signal pairs to {output_path}")
    return df


def export_triple_signal_pairs(main_conn, svm_conn, output_dir, limit=500):
    """
    Export cross-author pairs strong on ALL THREE: hapax, alignment, AND SVM.
    The trifecta - strongest influence candidates.
    """
    if svm_conn is None:
        print("Skipping triple-signal export (no SVM database)")
        return None
    
    print("Computing triple-signal pairs (this may take a moment)...")
    
    # Load SVM scores
    svm_long = load_svm_scores_long_format(svm_conn)
    
    # Load pairs with positive hapax and alignment
    query = """
    SELECT 
        cj.source_text,
        cj.target_text,
        t1.source_filename as source_filename,
        t2.source_filename as target_filename,
        a1.author_name as source_author,
        a2.author_name as target_author,
        cj.source_year,
        cj.target_year,
        cj.hap_jac_dis,
        cj.al_jac_dis
    FROM combined_jaccard cj
    JOIN all_texts t1 ON cj.source_text = t1.text_id
    JOIN all_texts t2 ON cj.target_text = t2.text_id
    JOIN authors a1 ON cj.source_auth = a1.id
    JOIN authors a2 ON cj.target_auth = a2.id
    WHERE cj.source_auth != cj.target_auth
    AND cj.hap_jac_dis > 0
    AND cj.al_jac_dis > 0
    """
    
    df = pd.read_sql_query(query, main_conn)
    print(f"  Found {len(df)} pairs with positive hapax AND alignment")
    
    if len(df) == 0:
        print("  No pairs found with both positive hapax and alignment")
        return None
    
    # Merge SVM scores
    df = merge_svm_scores(df, svm_long)
    
    # Filter to positive SVM scores
    df = df[df['svm_score'] > 0.5]  # Above average SVM similarity
    print(f"  After SVM filter (>0.5): {len(df)} triple-signal pairs")
    
    if len(df) == 0:
        print("  No triple-signal pairs found")
        # Save empty file for documentation
        output_path = os.path.join(output_dir, 'triple_signal_pairs.csv')
        pd.DataFrame().to_csv(output_path, index=False)
        return None
    
    # Normalize and compute combined score
    df = normalize_scores(df)
    df['triple_score'] = df.get('hap_jac_dis_z', 0) + df.get('al_jac_dis_z', 0) + df.get('svm_score_z', 0)
    
    # Sort and limit
    df = df.sort_values('triple_score', ascending=False).head(limit)
    
    # Select output columns
    output_cols = ['source_filename', 'target_filename', 'source_author', 'target_author',
                   'source_year', 'target_year', 'hap_jac_dis', 'al_jac_dis', 'svm_score', 'triple_score']
    df = df[[c for c in output_cols if c in df.columns]]
    
    output_path = os.path.join(output_dir, 'triple_signal_pairs.csv')
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} triple-signal pairs to {output_path}")
    return df


def export_top_influence_candidates(main_conn, svm_conn, output_dir, limit=200):
    """
    Export top influence candidates using all three variables with
    logistic regression normalized coefficients.
    Hapax=79.2%, SVM=10.8%, Alignment=10.1%
    """
    print("Computing top influence candidates with full model...")
    
    # Load SVM scores
    svm_long = load_svm_scores_long_format(svm_conn) if svm_conn else None
    
    # Load all cross-author pairs (sample if too large)
    query = """
    SELECT 
        cj.source_text,
        cj.target_text,
        t1.source_filename as source_filename,
        t2.source_filename as target_filename,
        a1.author_name as source_author,
        a2.author_name as target_author,
        cj.source_year,
        cj.target_year,
        cj.hap_jac_dis,
        cj.al_jac_dis
    FROM combined_jaccard cj
    JOIN all_texts t1 ON cj.source_text = t1.text_id
    JOIN all_texts t2 ON cj.target_text = t2.text_id
    JOIN authors a1 ON cj.source_auth = a1.id
    JOIN authors a2 ON cj.target_auth = a2.id
    WHERE cj.source_auth != cj.target_auth
    ORDER BY cj.hap_jac_dis DESC
    LIMIT 50000
    """
    
    df = pd.read_sql_query(query, main_conn)
    print(f"  Loaded top {len(df)} pairs by hapax")
    
    # Merge SVM scores
    df = merge_svm_scores(df, svm_long)
    
    # Normalize
    df = normalize_scores(df)
    
    # Compute influence score
    df = compute_influence_score(df)
    
    # Sort and limit
    df = df.sort_values('influence_score', ascending=False).head(limit)
    
    # Select output columns
    output_cols = ['source_filename', 'target_filename', 'source_author', 'target_author',
                   'source_year', 'target_year', 'hap_jac_dis', 'al_jac_dis', 'svm_score', 'influence_score']
    df = df[[c for c in output_cols if c in df.columns]]
    
    output_path = os.path.join(output_dir, 'top_influence_candidates.csv')
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} top influence candidates to {output_path}")
    return df


def export_anchor_case_context(main_conn, svm_conn, output_dir):
    """Export all pairs involving Eliot and Lawrence."""
    
    print("Exporting Eliot-Lawrence pairs...")
    
    svm_long = load_svm_scores_long_format(svm_conn) if svm_conn else None
    
    query = """
    SELECT 
        cj.source_text,
        cj.target_text,
        t1.source_filename as source_filename,
        t2.source_filename as target_filename,
        a1.author_name as source_author,
        a2.author_name as target_author,
        cj.source_year,
        cj.target_year,
        cj.hap_jac_dis,
        cj.al_jac_dis
    FROM combined_jaccard cj
    JOIN all_texts t1 ON cj.source_text = t1.text_id
    JOIN all_texts t2 ON cj.target_text = t2.text_id
    JOIN authors a1 ON cj.source_auth = a1.id
    JOIN authors a2 ON cj.target_auth = a2.id
    WHERE (a1.author_name LIKE '%Eliot%' AND a2.author_name LIKE '%Lawrence%')
       OR (a1.author_name LIKE '%Lawrence%' AND a2.author_name LIKE '%Eliot%')
    """
    
    df = pd.read_sql_query(query, main_conn)
    
    # Merge SVM scores
    df = merge_svm_scores(df, svm_long)
    
    # Normalize and compute influence score
    df = normalize_scores(df)
    df = compute_influence_score(df)
    
    df = df.sort_values('influence_score', ascending=False)
    
    output_cols = ['source_filename', 'target_filename', 'source_author', 'target_author',
                   'source_year', 'target_year', 'hap_jac_dis', 'al_jac_dis', 'svm_score', 'influence_score']
    df = df[[c for c in output_cols if c in df.columns]]
    
    output_path = os.path.join(output_dir, 'eliot_lawrence_pairs.csv')
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} Eliot-Lawrence pairs to {output_path}")
    return df


def export_author_pair_summary(main_conn, output_dir):
    """Export summary statistics by author pair."""
    
    query = """
    SELECT 
        a1.author_name as source_author,
        a2.author_name as target_author,
        COUNT(*) as pair_count,
        AVG(cj.hap_jac_dis) as avg_hapax,
        AVG(cj.al_jac_dis) as avg_alignment,
        MAX(cj.hap_jac_dis) as max_hapax,
        MAX(cj.al_jac_dis) as max_alignment
    FROM combined_jaccard cj
    JOIN authors a1 ON cj.source_auth = a1.id
    JOIN authors a2 ON cj.target_auth = a2.id
    WHERE cj.source_auth != cj.target_auth
    GROUP BY a1.author_name, a2.author_name
    HAVING COUNT(*) >= 10
    ORDER BY AVG(cj.hap_jac_dis) DESC
    LIMIT 100
    """
    
    df = pd.read_sql_query(query, main_conn)
    output_path = os.path.join(output_dir, 'author_pair_summary.csv')
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} author pair summaries to {output_path}")
    return df


def write_summary(output_dir, stats, project_name):
    """Write a summary file."""
    summary_path = os.path.join(output_dir, 'EXPORT_SUMMARY.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"DH-Trace Export Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Project: {project_name}\n")
        f.write(f"Export date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Database Statistics:\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"Total texts: {stats['total_texts']:,}\n")
        f.write(f"Total authors: {stats['total_authors']:,}\n")
        f.write(f"Total pairs: {stats['total_pairs']:,}\n")
        f.write(f"Same-author pairs: {stats['same_author_pairs']:,}\n")
        f.write(f"Cross-author pairs: {stats['cross_author_pairs']:,}\n")
        f.write(f"Pairs with alignments (>0): {stats['pairs_with_alignments']:,}\n\n")
        
        f.write(f"Logistic Regression Weights Used:\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"Hapax:     79.2%\n")
        f.write(f"SVM:       10.8%\n")
        f.write(f"Alignment: 10.1%\n\n")
        
        f.write(f"Exported Files:\n")
        f.write(f"-" * 30 + "\n")
        f.write(f"high_alignment_pairs.csv - Cross-author pairs with highest alignment\n")
        f.write(f"high_hapax_pairs.csv - Cross-author pairs with highest hapax\n")
        f.write(f"double_signal_pairs.csv - Pairs strong on hapax AND alignment\n")
        f.write(f"triple_signal_pairs.csv - Pairs strong on ALL THREE variables\n")
        f.write(f"top_influence_candidates.csv - Top pairs by weighted influence score\n")
        f.write(f"eliot_lawrence_pairs.csv - Anchor case context\n")
        f.write(f"author_pair_summary.csv - Summary by author pair\n")
    
    print(f"\nSummary written to {summary_path}")


def main():
    project_name = get_project_name()
    
    output_dir = f'./projects/{project_name}/exports'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    main_conn, svm_conn = connect_dbs(project_name)
    
    try:
        print("\nGetting database statistics...")
        stats = get_basic_stats(main_conn)
        print(f"  Total pairs: {stats['total_pairs']:,}")
        print(f"  Cross-author pairs: {stats['cross_author_pairs']:,}")
        print(f"  Pairs with alignments: {stats['pairs_with_alignments']:,}")
        print()
        
        print("Exporting data subsets...\n")
        
        export_high_alignment_pairs(main_conn, svm_conn, output_dir)
        export_high_hapax_pairs(main_conn, svm_conn, output_dir)
        export_double_signal_pairs(main_conn, svm_conn, output_dir)
        export_triple_signal_pairs(main_conn, svm_conn, output_dir)
        export_top_influence_candidates(main_conn, svm_conn, output_dir)
        export_anchor_case_context(main_conn, svm_conn, output_dir)
        export_author_pair_summary(main_conn, output_dir)
        
        write_summary(output_dir, stats, project_name)
        
        print(f"\n{'=' * 50}")
        print(f"Export complete!")
        print(f"Files saved to: {output_dir}")
        print(f"{'=' * 50}")
        
    finally:
        main_conn.close()
        if svm_conn:
            svm_conn.close()


if __name__ == "__main__":
    main()