#!/usr/bin/env python3
"""
validate_influence_pairs.py

Validates multiple documented literary influence relationships using the Sextant model.
Extends the anchor_case.py methodology to test additional influence pairs beyond Eliot-Lawrence.

Usage:
    python validate_influence_pairs.py --list-novels    # List all novels in corpus
    python validate_influence_pairs.py --validate       # Run validation on known pairs
    python validate_influence_pairs.py --query "Dickens" "Collins"  # Query specific pair

Author: Generated for Tarah Wheeler's Sextant project
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import argparse

# Configuration
PROJECT = "eltec-100"
MAIN_DB = f"./projects/{PROJECT}/db/{PROJECT}.db"
SVM_DB = f"./projects/{PROJECT}/db/svm.db"
OUTPUT_DIR = "./validation_output"


# ============================================================================
# DOCUMENTED INFLUENCE PAIRS
# Based on ELTeC-100 corpus author IDs:
#   34=Eliot, 19=Lawrence, 9=Thackeray, 31=Disraeli, 25=Dickens, 41=Collins,
#   69=Braddon, 29=Hardy, 37=Brontë, 27=Gaskell, 1=Trollope
# ============================================================================

INFLUENCE_PAIRS = [
    {
        "name": "Eliot → Lawrence",
        "source_author_id": 34,
        "source_author_name": "Eliot",
        "target_author_id": 19,
        "target_author_name": "Lawrence",
        "citations": ["Leavis (1948); Chambers (1935) p.105"],
        "notes": "Anchor case"
    },
    {
        "name": "Thackeray → Disraeli",
        "source_author_id": 9,
        "source_author_name": "Thackeray",
        "target_author_id": 31,
        "target_author_name": "Disraeli",
        "citations": ["Cline (1943) Review of English Studies"],
        "notes": "Discovered case - rivalry/parody"
    },
    {
        "name": "Dickens → Collins",
        "source_author_id": 25,
        "source_author_name": "Dickens",
        "target_author_id": 41,
        "target_author_name": "Collins",
        "citations": ["Nayder (2002); Peters (1991)"],
        "notes": "Mentorship 1851-1870"
    },
    {
        "name": "Collins → Braddon",
        "source_author_id": 41,
        "source_author_name": "Collins",
        "target_author_id": 69,
        "target_author_name": "Braddon",
        "citations": ["Pykett (1994); Wolff (1979)"],
        "notes": "Sensation fiction genre"
    },
    {
        "name": "Eliot → Hardy",
        "source_author_id": 34,
        "source_author_name": "Eliot",
        "target_author_id": 29,
        "target_author_name": "Hardy",
        "citations": ["Berle (1917); King (1978)"],
        "notes": "Victorian realism tradition"
    },
    {
        "name": "Brontë → Collins",
        "source_author_id": 37,
        "source_author_name": "Brontë",
        "target_author_id": 41,
        "target_author_name": "Collins",
        "citations": ["Heller (1992)"],
        "notes": "Gothic, female agency"
    },
    {
        "name": "Brontë → Gaskell",
        "source_author_id": 37,
        "source_author_name": "Brontë",
        "target_author_id": 27,
        "target_author_name": "Gaskell",
        "citations": ["Uglow (1993)"],
        "notes": "Close friends, female protagonists"
    },
    {
        "name": "Gaskell → Dickens",
        "source_author_id": 27,
        "source_author_name": "Gaskell",
        "target_author_id": 25,
        "target_author_name": "Dickens",
        "citations": ["Gallagher (1985)"],
        "notes": "Industrial novel genre"
    },
    {
        "name": "Dickens → Hardy",
        "source_author_id": 25,
        "source_author_name": "Dickens",
        "target_author_id": 29,
        "target_author_name": "Hardy",
        "citations": ["Williams (1970)"],
        "notes": "Social realism"
    },
    {
        "name": "Thackeray → Trollope",
        "source_author_id": 9,
        "source_author_name": "Thackeray",
        "target_author_id": 1,
        "target_author_name": "Trollope",
        "citations": ["Skilton (1972)"],
        "notes": "Satirical social novel"
    }
]


def list_available_novels():
    """List all novels/authors in the ELTeC corpus."""
    print("=" * 70)
    print("AVAILABLE AUTHORS IN ELTEC-100 CORPUS")
    print("=" * 70)
    
    conn = sqlite3.connect(MAIN_DB)
    authors_df = pd.read_sql_query("SELECT * FROM authors ORDER BY author_name", conn)
    print(f"\n{len(authors_df)} AUTHORS:\n")
    
    for _, row in authors_df.iterrows():
        print(f"  ID {row['id']:>3}: {row['author_name']}")
    
    # Get novels with chapter counts
    print(f"\n\nNOVELS WITH CHAPTER COUNTS:\n")
    novels_df = pd.read_sql_query("""
        SELECT d.dir, a.author_name, COUNT(t.text_id) as chapters, MIN(t.year) as year
        FROM dirs d
        JOIN all_texts t ON d.id = t.dir
        JOIN authors a ON t.author_id = a.id
        GROUP BY d.id
        ORDER BY a.author_name, d.dir
    """, conn)
    
    current_author = None
    for _, row in novels_df.iterrows():
        if row['author_name'] != current_author:
            current_author = row['author_name']
            print(f"\n  {current_author}:")
        print(f"    {row['dir']} ({row['year']}) - {row['chapters']} chapters")
    
    conn.close()


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
    
    cols_to_drop = ['target_novel', 'target_chapter_num', 'source_novel_name', 'novel', 'number']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    print(f"  Pairs with valid SVM scores: {(~df['svm_score'].isna()).sum():,}")
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


def get_logistic_regression_weights():
    """Return the logistic regression coefficients from the paper."""
    # From the paper: Hapax 79.2%, SVM 10.8%, Alignment 10.1%
    weights = (0.792, 0.101, 0.108)
    print(f"\nUsing logistic regression weights: Hapax={weights[0]:.1%}, Align={weights[1]:.1%}, SVM={weights[2]:.1%}")
    return weights


def compute_influence_scores(df, weights):
    """Compute influence scores using the optimal weights."""
    hap_w, al_w, svm_w = weights
    df['influence_score'] = (
        df['hap_jac_dis'] * hap_w +
        df['al_jac_dis'] * al_w +
        df['svm_score'] * svm_w
    )
    return df


def find_pairs_by_author_id(df, source_author_id, target_author_id):
    """Find all cross-author pairs between two specific authors by ID."""
    mask = (
        (df['source_auth'] == source_author_id) & 
        (df['target_auth'] == target_author_id) &
        (df['same_author'] == 0)
    )
    return df[mask].copy()


def calculate_percentile(pairs_df, all_cross_author_df):
    """Calculate percentile for each pair among all cross-author pairs."""
    if len(pairs_df) == 0:
        return pairs_df
    
    pairs_df = pairs_df.copy()
    pairs_df['percentile'] = pairs_df['influence_score'].apply(
        lambda x: (all_cross_author_df['influence_score'] < x).mean() * 100
    )
    
    return pairs_df.sort_values('influence_score', ascending=False)


def validate_all_pairs(df, weights):
    """Validate all documented influence pairs."""
    print("\n" + "=" * 70)
    print("VALIDATING DOCUMENTED INFLUENCE PAIRS")
    print("=" * 70)
    
    df = compute_influence_scores(df, weights)
    cross_author = df[df['same_author'] == 0]
    total_cross_author = len(cross_author)
    
    print(f"\nTotal cross-author pairs: {total_cross_author:,}")
    
    results = []
    
    for pair_info in INFLUENCE_PAIRS:
        print(f"\n--- {pair_info['name']} ---")
        
        pairs = find_pairs_by_author_id(
            df,
            source_author_id=pair_info['source_author_id'],
            target_author_id=pair_info['target_author_id']
        )
        
        if len(pairs) == 0:
            print(f"  WARNING: No pairs found!")
            results.append({
                'Pair': pair_info['name'],
                'Source': pair_info['source_author_name'],
                'Target': pair_info['target_author_name'],
                'N_Pairs': 0,
                'Max_Pctl': None,
                'Top10_Mean': None,
                'Above_99': 0,
                'Above_95': 0,
                'Citation': pair_info['citations'][0]
            })
            continue
        
        pairs = calculate_percentile(pairs, cross_author)
        
        n_pairs = len(pairs)
        max_pctl = pairs['percentile'].max()
        top_10_mean = pairs.head(10)['percentile'].mean() if n_pairs >= 10 else pairs['percentile'].mean()
        above_99 = (pairs['percentile'] >= 99).sum()
        above_95 = (pairs['percentile'] >= 95).sum()
        
        print(f"  Found {n_pairs:,} chapter pairs")
        print(f"  Max percentile: {max_pctl:.2f}%")
        print(f"  Top 10 mean:    {top_10_mean:.2f}%")
        print(f"  Pairs ≥99th:    {above_99:,}")
        print(f"  Pairs ≥95th:    {above_95:,}")
        
        print(f"  Top 3 pairs:")
        for i, (_, row) in enumerate(pairs.head(3).iterrows(), 1):
            print(f"    {i}. ch{row['source_chapter']:>2} → ch{row['target_chapter']:>2} ({row['percentile']:.2f}%)")
        
        results.append({
            'Pair': pair_info['name'],
            'Source': pair_info['source_author_name'],
            'Target': pair_info['target_author_name'],
            'N_Pairs': n_pairs,
            'Max_Pctl': round(max_pctl, 2),
            'Top10_Mean': round(top_10_mean, 2),
            'Above_99': above_99,
            'Above_95': above_95,
            'Citation': pair_info['citations'][0]
        })
    
    return pd.DataFrame(results)


def query_specific_pair(df, weights, source_query, target_query):
    """Query a specific author pair interactively."""
    print(f"\n" + "=" * 70)
    print(f"QUERYING: {source_query} → {target_query}")
    print("=" * 70)
    
    df = compute_influence_scores(df, weights)
    cross_author = df[df['same_author'] == 0]
    
    source_mask = df['source_author_name'].str.contains(source_query, case=False, na=False)
    target_mask = df['target_author_name'].str.contains(target_query, case=False, na=False)
    cross_mask = df['same_author'] == 0
    
    pairs = df[source_mask & target_mask & cross_mask].copy()
    
    if len(pairs) == 0:
        print(f"\nNo pairs found matching '{source_query}' → '{target_query}'")
        return
    
    pairs = calculate_percentile(pairs, cross_author)
    
    print(f"\nFound {len(pairs):,} chapter pairs")
    print(f"\nTop 20 pairs:\n")
    
    for i, (_, row) in enumerate(pairs.head(20).iterrows(), 1):
        print(f"{i:3}. {row['source_author_name']:<12} ch{row['source_chapter']:>2} → "
              f"{row['target_author_name']:<12} ch{row['target_chapter']:>2}  "
              f"Pctl: {row['percentile']:>6.2f}%")
    
    print(f"\nSummary:")
    print(f"  Max:    {pairs['percentile'].max():.2f}%")
    print(f"  Mean:   {pairs['percentile'].mean():.2f}%")
    print(f"  ≥99th:  {(pairs['percentile'] >= 99).sum()}")
    print(f"  ≥95th:  {(pairs['percentile'] >= 95).sum()}")


def main():
    parser = argparse.ArgumentParser(description='Validate influence pairs with Sextant')
    parser.add_argument('--list-novels', action='store_true', help='List corpus contents')
    parser.add_argument('--validate', action='store_true', help='Validate all pairs')
    parser.add_argument('--query', nargs=2, metavar=('SRC', 'TGT'), help='Query specific pair')
    parser.add_argument('--output', default=OUTPUT_DIR, help='Output directory')
    
    args = parser.parse_args()
    
    if args.list_novels:
        list_available_novels()
        return
    
    if args.validate or args.query:
        df = load_data()
        weights = get_logistic_regression_weights()
    
    if args.validate:
        os.makedirs(args.output, exist_ok=True)
        
        results_df = validate_all_pairs(df, weights)
        
        output_file = os.path.join(args.output, 'influence_validation_results.csv')
        results_df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print("\n" + results_df.to_string(index=False))
        
        # LaTeX table
        tex_file = os.path.join(args.output, 'validation_table.tex')
        with open(tex_file, 'w') as f:
            f.write("\\begin{table}[t]\n\\centering\\small\n")
            f.write("\\caption{Validation of documented influence relationships}\n")
            f.write("\\label{tab:validation}\n")
            f.write("\\begin{tabular}{lrrrrr}\n\\toprule\n")
            f.write("\\textbf{Pairing} & \\textbf{N} & \\textbf{Max} & \\textbf{Top-10} & \\textbf{$\\geq$99} & \\textbf{$\\geq$95} \\\\\n")
            f.write("\\midrule\n")
            for _, row in results_df.iterrows():
                if row['N_Pairs'] > 0:
                    f.write(f"{row['Pair']} & {row['N_Pairs']:,} & {row['Max_Pctl']:.1f} & "
                           f"{row['Top10_Mean']:.1f} & {row['Above_99']} & {row['Above_95']} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        
        print(f"\nResults saved to: {args.output}/")
    
    if args.query:
        query_specific_pair(df, weights, args.query[0], args.query[1])
    
    if not (args.list_novels or args.validate or args.query):
        parser.print_help()


if __name__ == "__main__":
    main()