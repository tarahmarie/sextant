#!/usr/bin/env python3
"""
Score the Shelley-Lovelace corpus using frozen ELTeC ground truth weights.

Fully self-contained -- no dependency on model.py or ELTeC result files.
All parameters hardcoded from the ELTeC-100 v1.0-thesis run (Apr 29, 2026):
  - 5,956,426 pairs, ROC AUC 0.784 (10-fold CV), 0.783 (held-out test)
  - All 8 anchor cases above 99th percentile (avg 99.99%)
  - SHAP: Hapax 84.5%, SVM 14.0%, Alignment 1.5%

Methodology note:
  Percentile rankings are reported against cross-author pairs only, matching
  the ELTeC analysis convention. The ELTeC validation reported its 8 documented
  influence cases as percentiles of the 5,785,953 cross-author pair distribution
  (excluding 170,473 same-author pairs). This script does the same: same-author
  pairs are computed for completeness but excluded from the rank denominator
  used for cross-author influence claims.

Output:
  - Prints to stdout (terminal)
  - Also writes to ./projects/shelley-lovelace/results/score_shelley_lovelace_output.txt

Run from sextant root:
    poetry run python score_shelley_lovelace.py
"""

import re
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = 'shelley-lovelace'
OUTPUT_FILE = Path(f'./projects/{PROJECT}/results/score_shelley_lovelace_output.txt')

# ======================================================================
# ELTeC GROUND TRUTH PARAMETERS (frozen -- do not modify)
# ======================================================================

INTERCEPT = -4.207439184487401

COEFS = {
    'hap': -1.2319672017708805,
    'al':  -0.15314075364182664,
    'svm':  0.18399544222763672,
}

# Z-score normalization (fitted on ELTeC 80% training split only)
MEANS = {
    'hap': 0.945407929605967,
    'al':  0.9999721042109713,
    'svm': 0.32427006589173824,
}
STDS = {
    'hap': 0.011339318040662662,
    'al':  0.00025511513289408944,
    'svm': 0.2573745232415633,
}


class Tee:
    """Write output to both stdout and a file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def extract_author(filename):
    """Extract author identifier from ELTeC-pattern filename.

    Filenames look like:
      1843-ENG18430--Lovelace_Ada-Note_D
      1840-ENG18401--Shelley_Percy-section_07
      1843-ENG18430--Menabrea_Lovelace-translation

    Returns the full Surname_Firstname (or composite) so authors who
    share a surname (e.g. Shelley_Mary vs Shelley_Percy) stay distinct.
    """
    if filename is None:
        return None
    m = re.search(r'--([A-Za-z_]+?)(?:-|$)', filename)
    return m.group(1) if m else None


def load_and_score():
    main_conn = sqlite3.connect(f'./projects/{PROJECT}/db/{PROJECT}.db')
    svm_conn = sqlite3.connect(f'./projects/{PROJECT}/db/svm.db')

    df = pd.read_sql_query("""
        SELECT cj.*,
               t1.source_filename as source_name,
               t2.source_filename as target_name,
               t1.short_name_for_svm as source_svm_name,
               t2.short_name_for_svm as target_svm_name,
               t1.chapter_num as source_chapter,
               t2.chapter_num as target_chapter
        FROM combined_jaccard cj
        JOIN all_texts t1 ON cj.source_text = t1.text_id
        JOIN all_texts t2 ON cj.target_text = t2.text_id
    """, main_conn)

    print(f"Total pairs: {len(df):,}")

    # SVM lookup: "How confident is the SVM that the TARGET text
    # was written by the SOURCE author?" This is the influence signal.
    chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
    svm_conn.close()

    # Vectorized SVM join: melt wide table to long, then merge
    id_cols = ['novel', 'number']
    score_cols = [c for c in chapter_df.columns if c not in id_cols]
    chapter_long = chapter_df.melt(
        id_vars=id_cols,
        value_vars=score_cols,
        var_name='source_svm_name',
        value_name='svm_score'
    )
    chapter_long['number'] = chapter_long['number'].astype(str)
    df['target_chapter'] = df['target_chapter'].astype(str)

    df = df.merge(
        chapter_long,
        left_on=['target_svm_name', 'target_chapter', 'source_svm_name'],
        right_on=['novel', 'number', 'source_svm_name'],
        how='left'
    ).drop(columns=['novel', 'number'])

    n_with = df['svm_score'].notna().sum()
    n_without = df['svm_score'].isna().sum()
    print(f"Pairs with SVM scores: {n_with:,}")
    print(f"Pairs without SVM scores: {n_without:,}")
    df = df.dropna(subset=['svm_score'])

    # Z-score normalize using ELTeC training parameters
    df['hap_z'] = (df['hap_jac_dis'] - MEANS['hap']) / STDS['hap']
    df['al_z']  = (df['al_jac_dis']  - MEANS['al'])  / STDS['al']
    df['svm_z'] = (df['svm_score']   - MEANS['svm']) / STDS['svm']

    # Apply frozen ELTeC coefficients
    df['logit'] = (INTERCEPT
                   + COEFS['hap'] * df['hap_z']
                   + COEFS['al']  * df['al_z']
                   + COEFS['svm'] * df['svm_z'])
    df['prob'] = 1 / (1 + np.exp(-df['logit']))

    # Identify same-author vs cross-author pairs
    df['source_author'] = df['source_name'].apply(extract_author)
    df['target_author'] = df['target_name'].apply(extract_author)
    df['is_same_author'] = df['source_author'] == df['target_author']

    n_same = df['is_same_author'].sum()
    n_cross = (~df['is_same_author']).sum()
    print(f"Same-author pairs: {n_same:,}")
    print(f"Cross-author pairs: {n_cross:,}")

    main_conn.close()
    return df


def rank_pairs(df, cross_author_only=True):
    """Rank pairs by probability.

    If cross_author_only is True (default), restrict to cross-author pairs
    and rank against that distribution. This matches the ELTeC analysis
    convention used in logistic_regression.py.
    """
    if cross_author_only:
        ranked = df[~df['is_same_author']].copy()
    else:
        ranked = df.copy()
    ranked = ranked.sort_values('prob', ascending=False).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)
    return ranked


def print_pair_row(row, N_cross, indent="  "):
    """Standard formatter for printing a pair with all signals."""
    print(f"{indent}rank {row['rank']:6d}/{N_cross}  p={row['prob']:.4f}  "
          f"al={row['al_jac_dis']:.4f}  hap={row['hap_jac_dis']:.4f}  "
          f"svm={row['svm_score']:.4f}")
    print(f"{indent}    {row['source_name']} -> {row['target_name']}")


def main():
    df = load_and_score()

    df_cross = rank_pairs(df, cross_author_only=True)
    df_all = rank_pairs(df, cross_author_only=False)
    N_cross = len(df_cross)
    N_all = len(df_all)

    print(f"\n{'='*80}")
    print(f"=== TOP 25 PAIRS, ALL PAIRS RANKING (includes same-author for context) ===")
    print(f"{'='*80}\n")
    for _, row in df_all.head(25).iterrows():
        sa = '[same]' if row['is_same_author'] else '[cross]'
        print(f"  {row['rank']:5d}. p={row['prob']:.4f}  {sa}  {row['source_name']} -> {row['target_name']}")

    print(f"\n{'='*80}")
    print(f"=== TOP 25 CROSS-AUTHOR PAIRS (the influence-candidate ranking) ===")
    print(f"{'='*80}\n")
    for _, row in df_cross.head(25).iterrows():
        print(f"  {row['rank']:5d}. p={row['prob']:.4f}  {row['source_name']} -> {row['target_name']}")

    # Mary Shelley -> Lovelace (cross-author by construction).
    # Percy Shelley is in the corpus as a comparator and is intentionally
    # excluded here — he and Mary are distinct authors.
    sl = df_cross[df_cross['source_name'].str.contains('Shelley_Mary') &
                  df_cross['target_name'].str.contains('Lovelace')]
    print(f"\n{'='*80}")
    print(f"=== MARY SHELLEY -> LOVELACE PAIRS ({len(sl)} pairs) ===")
    print(f"=== Ranks against {N_cross:,} cross-author pairs ===")
    print(f"{'='*80}\n")
    for _, row in sl.iterrows():
        print(f"  rank {row['rank']:6d}/{N_cross}  p={row['prob']:.4f}  "
              f"hap={row['hap_jac_dis']:.6f}  al={row['al_jac_dis']:.6f}  "
              f"svm={row['svm_score']:.4f}  {row['source_name']} -> {row['target_name']}")

    if len(sl) > 0:
        best = sl['rank'].min()
        print(f"\n  Best rank: {best} of {N_cross} cross-author pairs (top {best/N_cross*100:.2f}%)")
        print(f"  Percentile: {(1 - best/N_cross)*100:.2f}th")

    # All authors -> Lovelace (top 20, cross-author only)
    al_df = df_cross[df_cross['target_name'].str.contains('Lovelace')]
    print(f"\n{'='*80}")
    print(f"=== ALL AUTHORS -> LOVELACE, top 20 (cross-author only) ===")
    print(f"=== Ranks against {N_cross:,} cross-author pairs ===")
    print(f"{'='*80}\n")
    for _, row in al_df.head(20).iterrows():
        print(f"  rank {row['rank']:6d}/{N_cross}  p={row['prob']:.4f}  "
              f"{row['source_name']} -> {row['target_name']}")

    # Somerville -> Lovelace
    som = df_cross[df_cross['source_name'].str.contains('Somerville') &
                   df_cross['target_name'].str.contains('Lovelace')]
    if len(som) > 0:
        print(f"\n{'='*80}")
        print(f"=== SOMERVILLE -> LOVELACE, top 10 ===")
        print(f"=== Ranks against {N_cross:,} cross-author pairs ===")
        print(f"{'='*80}\n")
        for _, row in som.head(10).iterrows():
            print(f"  rank {row['rank']:6d}/{N_cross}  p={row['prob']:.4f}  "
                  f"{row['source_name']} -> {row['target_name']}")
        best_som = som['rank'].min()
        print(f"\n  Best Somerville -> Lovelace rank: {best_som} (top {best_som/N_cross*100:.2f}%)")
        print(f"  Percentile: {(1 - best_som/N_cross)*100:.2f}th")

    # Babbage -> Lovelace
    bab = df_cross[df_cross['source_name'].str.contains('Babbage') &
                   df_cross['target_name'].str.contains('Lovelace')]
    if len(bab) > 0:
        print(f"\n{'='*80}")
        print(f"=== BABBAGE -> LOVELACE, top 10 ===")
        print(f"=== Ranks against {N_cross:,} cross-author pairs ===")
        print(f"{'='*80}\n")
        for _, row in bab.head(10).iterrows():
            print(f"  rank {row['rank']:6d}/{N_cross}  p={row['prob']:.4f}  "
                  f"{row['source_name']} -> {row['target_name']}")
        best_bab = bab['rank'].min()
        print(f"\n  Best Babbage -> Lovelace rank: {best_bab} (top {best_bab/N_cross*100:.2f}%)")
        print(f"  Percentile: {(1 - best_bab/N_cross)*100:.2f}th")

    # Author-level summary table with per-signal medians
    # Reveals which signal drives each author's connection: hapax (rare-vocab),
    # SVM (writing-style), or alignment (shared phrases). Lower hap_dis and al_dis
    # mean more overlap; higher svm_score means more style match.
    print(f"\n{'='*80}")
    print(f"=== AUTHOR -> LOVELACE PERCENTILE SUMMARY ===")
    print(f"=== Best rank, percentile, top probability, and per-signal medians ===")
    print(f"=== (lower hap_dis = more shared rare vocab; lower al_dis = more shared phrases; higher svm = closer style match) ===")
    print(f"{'='*80}\n")
    header = (f"  {'Author':<20} {'Best rank':>10} {'Percentile':>11}  "
              f"{'Top prob':>9}  {'Med hap':>8}  {'Med al':>8}  {'Med svm':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    lovelace_targets = df_cross[df_cross['target_name'].str.contains('Lovelace')]
    author_summary = (
        lovelace_targets
        .groupby('source_author')
        .agg(
            best_rank=('rank', 'min'),
            top_prob=('prob', 'max'),
            med_hap=('hap_jac_dis', 'median'),
            med_al=('al_jac_dis', 'median'),
            med_svm=('svm_score', 'median'),
        )
        .sort_values('best_rank')
    )
    for author, row in author_summary.head(25).iterrows():
        pct = (1 - row['best_rank'] / N_cross) * 100
        print(f"  {author:<20} {int(row['best_rank']):>10} {pct:>10.2f}th  "
              f"{row['top_prob']:>9.4f}  {row['med_hap']:>8.4f}  "
              f"{row['med_al']:>8.4f}  {row['med_svm']:>8.4f}")

    # =================================================================
    # SEQUENCE ALIGNMENT FINDINGS
    # =================================================================
    # Most pairs have al_jac_dis = 1.0 (no shared aligned passages); pairs below
    # 1.0 are where TextPAIR found actual textual overlap. These are candidates
    # for close reading: shared phrases, direct citations, or paraphrases.

    aligned_pairs = df_cross[df_cross['al_jac_dis'] < 1.0].sort_values('al_jac_dis')
    n_aligned = len(aligned_pairs)
    print(f"\n{'='*80}")
    print(f"=== SEQUENCE ALIGNMENT OVERVIEW ===")
    print(f"{'='*80}\n")
    print(f"  Cross-author pairs with shared alignments: {n_aligned:,}")
    print(f"  ({n_aligned / N_cross * 100:.2f}% of {N_cross:,} cross-author pairs)\n")

    # Top general alignments (corpus-wide, not Lovelace-specific)
    print(f"\n{'='*80}")
    print(f"=== TOP CROSS-AUTHOR SEQUENCE ALIGNMENTS (general, all targets) ===")
    print(f"=== Sorted by al_jac_dis ascending (most overlap first) ===")
    print(f"=== Surfaces e.g. Coleridge -> Mary Shelley citation, Percy/Mary cross-references ===")
    print(f"{'='*80}\n")
    for _, row in aligned_pairs.head(30).iterrows():
        print_pair_row(row, N_cross)

    # Top alignments INTO Lovelace (thesis-relevant subset)
    aligned_to_lovelace = aligned_pairs[aligned_pairs['target_name'].str.contains('Lovelace')]
    print(f"\n{'='*80}")
    print(f"=== TOP CROSS-AUTHOR ALIGNMENTS INTO LOVELACE ===")
    print(f"=== Lovelace as target, sorted by al_jac_dis ascending ===")
    print(f"{'='*80}\n")
    print(f"  Cross-author pairs with shared alignments AND Lovelace target: {len(aligned_to_lovelace):,}\n")
    for _, row in aligned_to_lovelace.head(30).iterrows():
        print_pair_row(row, N_cross)

    # =================================================================
    # SVM STYLOMETRY FINDINGS
    # =================================================================
    # SVM score is "P(target was written by source author)" trained on the corpus.
    # High scores mean writing-style match, independent of vocabulary overlap.

    # Top general SVM (corpus-wide, not Lovelace-specific)
    svm_general = df_cross.sort_values('svm_score', ascending=False)
    print(f"\n{'='*80}")
    print(f"=== TOP CROSS-AUTHOR SVM STYLOMETRY (general, all targets) ===")
    print(f"=== Sorted by svm_score descending (highest style match first) ===")
    print(f"=== Surfaces unexpected style matches across the corpus ===")
    print(f"{'='*80}\n")
    for _, row in svm_general.head(30).iterrows():
        print_pair_row(row, N_cross)

    # Top SVM INTO Lovelace (thesis-relevant subset)
    svm_to_lovelace = (
        df_cross[df_cross['target_name'].str.contains('Lovelace')]
        .sort_values('svm_score', ascending=False)
    )
    print(f"\n{'='*80}")
    print(f"=== TOP CROSS-AUTHOR SVM STYLOMETRY INTO LOVELACE ===")
    print(f"=== Lovelace as target, sorted by svm_score descending ===")
    print(f"{'='*80}\n")
    for _, row in svm_to_lovelace.head(30).iterrows():
        print_pair_row(row, N_cross)


if __name__ == '__main__':
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            main()
        finally:
            sys.stdout = original_stdout

    print(f"\n[Output also saved to: {OUTPUT_FILE}]")