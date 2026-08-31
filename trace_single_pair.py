#!/usr/bin/env python3
"""
SEXTANT SINGLE PAIR AUDIT TRACE
================================

This script allows an auditor to trace exactly what happens to a single text pair
as it flows through the entire Sextant pipeline. It answers the question:

    "Show me EXACTLY how Sextant processes Eliot ch79 → Lawrence ch29"

For each pair, it shows:
1. RAW DATA: The source values from each database table
2. HAPAX LEGOMENA: The actual shared rare words and Jaccard calculation
3. SVM STYLOMETRY: The probability score from the trained classifier  
4. SEQUENCE ALIGNMENT: Any detected textual echoes
5. LOGISTIC REGRESSION: How the three signals combine into a final score
6. PERCENTILE RANKING: Where this pair falls among all cross-author pairs

With --audit, it additionally prints AUDIT RECEIPTS: for this pair, what the
pipeline actually did versus what the thesis says it did. Requires
audit_receipts.py in the same directory.

Usage:
    python trace_single_pair.py --source "Eliot" --source-chapter 79 --target "Lawrence" --target-chapter 29
    python trace_single_pair.py --pair-id 12345
    python trace_single_pair.py --pair-id 12345 --audit
    python trace_single_pair.py --interactive

Author: Tarah Wheeler
"""

import argparse
import sqlite3
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from util import get_project_name

# Audit receipts are optional: the trace still runs if the module is absent.
try:
    from audit_receipts import print_all_receipts
    RECEIPTS_AVAILABLE = True
except ImportError:
    RECEIPTS_AVAILABLE = False

# ============================================================================
# DATABASE CONNECTIONS
# ============================================================================

def get_db_connections():
    """Get connections to both main and SVM databases."""
    project_name = get_project_name()
    
    main_db_path = f"./projects/{project_name}/db/{project_name}.db"
    svm_db_path = f"./projects/{project_name}/db/svm.db"
    
    main_conn = sqlite3.connect(main_db_path)
    main_conn.row_factory = sqlite3.Row
    
    svm_conn = sqlite3.connect(svm_db_path)
    svm_conn.row_factory = sqlite3.Row
    
    return main_conn, svm_conn, project_name


# ============================================================================
# PAIR IDENTIFICATION
# ============================================================================

def get_random_cross_author_pair(main_conn):
    """
    Select a random cross-author pair from the database.
    Prioritizes pairs that have some signal (not all zeros).
    """
    cursor = main_conn.cursor()
    
    # Get a random cross-author pair that has some hapax overlap
    # (more interesting for demonstration than a pair with no signal)
    cursor.execute("""
        SELECT 
            cj.pair_id,
            cj.source_text,
            cj.target_text,
            cj.source_auth,
            cj.target_auth,
            cj.source_year,
            cj.target_year,
            cj.hap_jac_dis,
            a1.source_filename as source_name,
            a2.source_filename as target_name,
            a1.chapter_num as source_chapter,
            a2.chapter_num as target_chapter,
            a1.short_name_for_svm as source_svm_code,
            a2.short_name_for_svm as target_svm_code,
            auth1.author_name as source_author_name,
            auth2.author_name as target_author_name
        FROM combined_jaccard cj
        JOIN all_texts a1 ON cj.source_text = a1.text_id
        JOIN all_texts a2 ON cj.target_text = a2.text_id
        JOIN authors auth1 ON cj.source_auth = auth1.id
        JOIN authors auth2 ON cj.target_auth = auth2.id
        WHERE cj.source_auth != cj.target_auth
          AND cj.source_year <= cj.target_year
          AND cj.hap_jac_dis < 0.99
        ORDER BY RANDOM()
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    if result:
        return dict(result)
    
    # Fallback: any cross-author pair
    cursor.execute("""
        SELECT 
            cj.pair_id,
            cj.source_text,
            cj.target_text,
            cj.source_auth,
            cj.target_auth,
            cj.source_year,
            cj.target_year,
            cj.hap_jac_dis,
            a1.source_filename as source_name,
            a2.source_filename as target_name,
            a1.chapter_num as source_chapter,
            a2.chapter_num as target_chapter,
            a1.short_name_for_svm as source_svm_code,
            a2.short_name_for_svm as target_svm_code,
            auth1.author_name as source_author_name,
            auth2.author_name as target_author_name
        FROM combined_jaccard cj
        JOIN all_texts a1 ON cj.source_text = a1.text_id
        JOIN all_texts a2 ON cj.target_text = a2.text_id
        JOIN authors auth1 ON cj.source_auth = auth1.id
        JOIN authors auth2 ON cj.target_auth = auth2.id
        WHERE cj.source_auth != cj.target_auth
        ORDER BY RANDOM()
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    return dict(result) if result else None


def find_pair_by_authors_and_chapters(main_conn, source_author, source_chapter, 
                                       target_author, target_chapter):
    """
    Find a specific text pair by author names and chapter numbers.
    Returns the pair_id and basic metadata.
    """
    cursor = main_conn.cursor()
    
    # First, let's debug: find what author names look like
    cursor.execute("SELECT DISTINCT author_name FROM authors WHERE author_name LIKE ?", 
                  (f"%{source_author}%",))
    source_authors = cursor.fetchall()
    
    cursor.execute("SELECT DISTINCT author_name FROM authors WHERE author_name LIKE ?", 
                  (f"%{target_author}%",))
    target_authors = cursor.fetchall()
    
    print(f"\nDEBUG: Authors matching '{source_author}': {[a[0] for a in source_authors]}")
    print(f"DEBUG: Authors matching '{target_author}': {[a[0] for a in target_authors]}")
    
    # Check chapter_num format
    cursor.execute("""
        SELECT DISTINCT a.chapter_num, a.source_filename 
        FROM all_texts a 
        JOIN authors auth ON a.author_id = auth.id
        WHERE auth.author_name LIKE ? 
        LIMIT 5
    """, (f"%{source_author}%",))
    sample_chapters = cursor.fetchall()
    print(f"DEBUG: Sample chapters for {source_author}: {[(c[0], c[1][:50]) for c in sample_chapters]}")
    
    # Now try the actual query with string chapter numbers
    query = """
    SELECT
        cj.pair_id,
        cj.source_text,
        cj.target_text,
        cj.source_auth,
        cj.target_auth,
        cj.source_year,
        cj.target_year,
        a1.source_filename as source_name,
        a2.source_filename as target_name,
        a1.chapter_num as source_chapter,
        a2.chapter_num as target_chapter,
        a1.short_name_for_svm as source_svm_code,
        a2.short_name_for_svm as target_svm_code,
        auth1.author_name as source_author_name,
        auth2.author_name as target_author_name
    FROM combined_jaccard cj
    JOIN all_texts a1 ON cj.source_text = a1.text_id
    JOIN all_texts a2 ON cj.target_text = a2.text_id
    JOIN authors auth1 ON cj.source_auth = auth1.id
    JOIN authors auth2 ON cj.target_auth = auth2.id
    WHERE auth1.author_name LIKE ?
      AND auth2.author_name LIKE ?
      AND a1.chapter_num = ?
      AND a2.chapter_num = ?
    """
    
    # Try with string chapter numbers
    cursor.execute(query, (f"%{source_author}%", f"%{target_author}%", 
                          str(source_chapter), str(target_chapter)))
    result = cursor.fetchone()
    
    if result is None:
        print(f"\n❌ ERROR: Could not find pair:")
        print(f"   Source: {source_author} chapter {source_chapter}")
        print(f"   Target: {target_author} chapter {target_chapter}")
        
        # Additional debug: see if ANY pairs exist between these authors
        cursor.execute("""
            SELECT COUNT(*) 
            FROM combined_jaccard cj
            JOIN authors auth1 ON cj.source_auth = auth1.id
            JOIN authors auth2 ON cj.target_auth = auth2.id
            WHERE auth1.author_name LIKE ?
              AND auth2.author_name LIKE ?
        """, (f"%{source_author}%", f"%{target_author}%"))
        count = cursor.fetchone()[0]
        print(f"   DEBUG: Total pairs between {source_author} and {target_author}: {count}")
        
        return None
    
    return dict(result)


def find_pair_by_id(main_conn, pair_id):
    """Find a pair by its pair_id."""
    query = """
    SELECT
        cj.pair_id,
        cj.source_text,
        cj.target_text,
        cj.source_auth,
        cj.target_auth,
        cj.source_year,
        cj.target_year,
        a1.source_filename as source_name,
        a2.source_filename as target_name,
        a1.chapter_num as source_chapter,
        a2.chapter_num as target_chapter,
        a1.short_name_for_svm as source_svm_code,
        a2.short_name_for_svm as target_svm_code,
        auth1.author_name as source_author_name,
        auth2.author_name as target_author_name
    FROM combined_jaccard cj
    JOIN all_texts a1 ON cj.source_text = a1.text_id
    JOIN all_texts a2 ON cj.target_text = a2.text_id
    JOIN authors auth1 ON cj.source_auth = auth1.id
    JOIN authors auth2 ON cj.target_auth = auth2.id
    WHERE cj.pair_id = ?
    """
    
    cursor = main_conn.cursor()
    cursor.execute(query, (pair_id,))
    result = cursor.fetchone()
    
    if result is None:
        print(f"\n❌ ERROR: Could not find pair with pair_id = {pair_id}")
        return None
    
    return dict(result)


# ============================================================================
# STEP 1: RAW DATA FROM COMBINED_JACCARD
# ============================================================================

def trace_combined_jaccard(main_conn, pair_id):
    """
    STEP 1: Show the raw values stored in combined_jaccard table.
    This is the final merged table containing all three signals.
    """
    print("\n" + "=" * 70)
    print("STEP 1: RAW DATA FROM combined_jaccard TABLE")
    print("=" * 70)
    
    query = """
    SELECT * FROM combined_jaccard WHERE pair_id = ?
    """
    cursor = main_conn.cursor()
    cursor.execute(query, (pair_id,))
    result = cursor.fetchone()
    
    if result is None:
        print(f"❌ No entry in combined_jaccard for pair_id = {pair_id}")
        return None
    
    data = dict(result)
    
    print(f"\nPair ID: {data['pair_id']}")
    print(f"\nSource text ID: {data['source_text']} (author_id: {data['source_auth']}, year: {data['source_year']})")
    print(f"Target text ID: {data['target_text']} (author_id: {data['target_auth']}, year: {data['target_year']})")
    print(f"\n--- Stored Values ---")
    print(f"  hap_jac_sim (Hapax Jaccard Similarity):  {data.get('hap_jac_sim', 'N/A')}")
    print(f"  hap_jac_dis (Hapax Jaccard Distance):    {data['hap_jac_dis']}")
    print(f"  al_jac_sim  (Alignment Jaccard Sim):     {data.get('al_jac_sim', 'N/A')}")
    print(f"  al_jac_dis  (Alignment Jaccard Dist):    {data['al_jac_dis']}")
    print(f"  source_length: {data['source_length']} words")
    print(f"  target_length: {data['target_length']} words")
    
    return data


# ============================================================================
# STEP 2: HAPAX LEGOMENA TRACE
# ============================================================================

def trace_hapax_legomena(main_conn, pair_info, combined_data):
    """
    STEP 2: Trace the hapax legomena calculation.
    Shows: raw hapax counts, intersection, Jaccard formula.
    """
    print("\n" + "=" * 70)
    print("STEP 2: HAPAX LEGOMENA CALCULATION")
    print("=" * 70)
    
    source_text_id = pair_info['source_text']
    target_text_id = pair_info['target_text']
    
    cursor = main_conn.cursor()
    
    # Get hapax counts for source
    cursor.execute("SELECT hapaxes, hapaxes_count FROM hapaxes WHERE source_filename = ?", 
                  (source_text_id,))
    source_hapax = cursor.fetchone()
    
    # Get hapax counts for target
    cursor.execute("SELECT hapaxes, hapaxes_count FROM hapaxes WHERE source_filename = ?", 
                  (target_text_id,))
    target_hapax = cursor.fetchone()
    
    # Get the hapax overlap for this pair
    cursor.execute("SELECT hapaxes, intersect_length FROM hapax_overlaps WHERE file_pair = ?",
                  (pair_info['pair_id'],))
    overlap = cursor.fetchone()
    
    print(f"\nSource text ({pair_info['source_name']}):")
    if source_hapax:
        print(f"  Total hapax legomena: {source_hapax['hapaxes_count']}")
    else:
        print("  ❌ No hapax data found")
        
    print(f"\nTarget text ({pair_info['target_name']}):")
    if target_hapax:
        print(f"  Total hapax legomena: {target_hapax['hapaxes_count']}")
    else:
        print("  ❌ No hapax data found")
    
    if overlap:
        print(f"\nShared hapax legomena: {overlap['intersect_length']}")
        
        # Parse and show the actual shared words
        try:
            hapax_data = overlap['hapaxes']
            if hapax_data:
                # Try different parsing approaches
                if isinstance(hapax_data, str):
                    if hapax_data.startswith('[') or hapax_data.startswith('{'):
                        shared_words = ast.literal_eval(hapax_data)
                    elif hapax_data.startswith('set('):
                        # Handle set(...) format
                        shared_words = list(ast.literal_eval(hapax_data))
                    else:
                        # Maybe comma-separated or space-separated
                        shared_words = [w.strip() for w in hapax_data.replace(',', ' ').split()]
                else:
                    shared_words = list(hapax_data) if hasattr(hapax_data, '__iter__') else []
                
                if shared_words:
                    print(f"\nThe actual shared rare words ({len(shared_words)} total):")
                    print(f"  {', '.join(str(w) for w in shared_words)}")
                else:
                    print(f"\n  (Shared words list is empty)")
            else:
                print(f"\n  (No shared words data stored)")
        except Exception as e:
            print(f"\n  (Could not parse shared words list: {type(e).__name__}: {e})")
    else:
        print("\n⚠️  No hapax overlap entry found for this pair")
    
    # Show the Jaccard calculation
    # NOTE: Sextant uses jac_sim = intersection / (source + target), NOT the standard
    # Jaccard formula of intersection / union. This is documented in database_ops.py line 274.
    print("\n--- Jaccard Similarity Calculation ---")
    if source_hapax and target_hapax and overlap:
        intersection = overlap['intersect_length']
        denominator = source_hapax['hapaxes_count'] + target_hapax['hapaxes_count']
        
        print(f"  Intersection (shared hapaxes): {intersection}")
        print(f"  Denominator: {source_hapax['hapaxes_count']} + {target_hapax['hapaxes_count']} = {denominator}")
        
        if denominator > 0:
            jac_sim = intersection / denominator
            jac_dis = 1 - jac_sim
            print(f"\n  Jaccard Similarity = {intersection} / {denominator} = {jac_sim:.6f}")
            print(f"  Jaccard Distance   = 1 - {jac_sim:.6f} = {jac_dis:.6f}")
            print(f"\n  ✓ Stored value (hap_jac_dis): {combined_data['hap_jac_dis']:.6f}")
            
            # Verify they match
            if abs(jac_dis - combined_data['hap_jac_dis']) < 0.0001:
                print("  ✓ Calculation matches stored value")
            else:
                print(f"  ⚠️ Calculation differs from stored value by {abs(jac_dis - combined_data['hap_jac_dis']):.6f}")
    
    return overlap


# ============================================================================
# STEP 3: SVM STYLOMETRY TRACE
# ============================================================================

def trace_svm_stylometry(svm_conn, pair_info):
    """
    STEP 3: Trace the SVM stylometry score.
    Shows: which novel the target chapter resembles, and the probability.
    Returns the SVM score for use in later steps.

    NOTE FOR AUDITORS: this function resolves the source novel from
    all_texts.short_name_for_svm. logistic_regression.py resolves it from
    novels_dict, which is keyed by dirs.id and merged on authors.id. If those
    two id spaces do not correspond, this trace and the model read different
    columns. Run with --audit to see both side by side (Receipt 1).
    """
    print("\n" + "=" * 70)
    print("STEP 3: SVM STYLOMETRY SCORE")
    print("=" * 70)
    
    # Use canonical identifiers stored in the database rather than
    # re-deriving them from filenames. The SVM's chapter_assessments table
    # keys on (novel, number) which correspond directly to
    # all_texts.short_name_for_svm and all_texts.chapter_num.
    chapter_num = pair_info['target_chapter']
    novel_name = pair_info['target_svm_code']
    source_novel = pair_info['source_svm_code']
    
    print(f"\nLooking up: How much does {pair_info['target_author_name']}'s chapter")
    print(f"            stylistically resemble {pair_info['source_author_name']}'s novel?")
    
    cursor = svm_conn.cursor()
    
    # Get the chapter assessment row - try case-insensitive match
    cursor.execute("SELECT * FROM chapter_assessments WHERE LOWER(novel) = LOWER(?) AND number = ?",
                  (novel_name, chapter_num))
    result = cursor.fetchone()
    
    # If not found, try partial match
    if result is None:
        cursor.execute("SELECT * FROM chapter_assessments WHERE LOWER(novel) LIKE LOWER(?) AND number = ?",
                      (f"%{novel_name}%", chapter_num))
        result = cursor.fetchone()
    
    svm_score = None  # Will store the extracted score
    
    if result:
        # Get column names
        columns = [description[0] for description in cursor.description]
        row_dict = dict(zip(columns, result))
        
        print(f"\nTarget chapter: {novel_name} chapter {chapter_num}")
        print(f"\n--- SVM Probabilities for this chapter resembling each novel ---")
        
        # Find the source novel column - try multiple matching strategies
        source_novel_col = None
        source_novel_lower = source_novel.lower()
        
        for col in columns:
            col_lower = col.lower()
            # Try exact match first, then partial
            if col_lower == source_novel_lower:
                source_novel_col = col
                break
            elif source_novel_lower in col_lower or col_lower in source_novel_lower:
                source_novel_col = col
                break
        
        if source_novel_col and row_dict.get(source_novel_col) is not None:
            svm_score = row_dict[source_novel_col]
            print(f"\n  ★ P(authored by {pair_info['source_author_name']}) = {svm_score:.4f}")
            
            # Show top 5 other probabilities for context
            probs = [(col, row_dict[col]) for col in columns 
                    if col not in ('novel', 'number') and row_dict[col] is not None]
            probs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n  Top 5 novel resemblances for this chapter:")
            for i, (col, prob) in enumerate(probs[:5], 1):
                marker = " ← SOURCE" if col == source_novel_col else ""
                print(f"    {i}. {col}: {prob:.4f}{marker}")
            
            # Show where source ranks
            source_rank = next((i for i, (col, _) in enumerate(probs, 1) if col == source_novel_col), None)
            if source_rank:
                print(f"\n  Source novel ranks #{source_rank} out of {len(probs)} novels")
        else:
            print(f"\n  ⚠️  Could not find column for source novel '{source_novel}'")
            print(f"  Available columns: {[c for c in columns if c not in ('novel', 'number')][:10]}")
    else:
        print(f"\n❌ No SVM assessment found for {novel_name} chapter {chapter_num}")
    
    return svm_score


# ============================================================================
# STEP 4: SEQUENCE ALIGNMENT TRACE
# ============================================================================

def trace_sequence_alignments(main_conn, pair_info, project_name):
    """
    STEP 4: Trace sequence alignments between the texts.
    Shows: actual aligned passages if any exist.
    """
    print("\n" + "=" * 70)
    print("STEP 4: SEQUENCE ALIGNMENTS")
    print("=" * 70)
    
    cursor = main_conn.cursor()

    # Query the alignments table by pair_id.
    #
    # IMPORTANT: do NOT filter by (source_filename, target_filename) here.
    # The alignments table stores ~37% of rows with source/target orientation
    # flipped relative to combined_jaccard's normalized (source_text,
    # target_text) ordering. TextPAIR records the orientation it found the
    # passage in; combined_jaccard normalizes to a canonical order. Filtering
    # by filename matches the wrong half of the corpus.
    #
    # pair_id is the unambiguous key in both tables and is unique in
    # alignments. We also pull back source_filename/target_filename so we can
    # detect when an alignment is stored flipped and label the passages
    # correctly to the reader.
    cursor.execute("""
        SELECT source_passage, target_passage,
               length_source_passage, length_target_passage,
               source_filename, target_filename
        FROM alignments
        WHERE pair_id = ?
    """, (pair_info['pair_id'],))

    alignments = cursor.fetchall()
    
    if not alignments:
        print(f"\n⚠️  No sequence alignments found between these chapters.")
        print("   This is common - only ~1.7% of pairs have alignments.")
        print("   al_jac_dis defaults to 1.0 (maximum distance) when no alignments exist.")
        return []
    
    print(f"\nFound {len(alignments)} sequence alignment(s)!")
    print("\n--- Aligned Passages (FULL TEXT) ---")

    canonical_source_id = pair_info['source_text']
    canonical_target_id = pair_info['target_text']

    for i, align in enumerate(alignments, 1):
        # Detect whether this alignment row is stored with the same
        # orientation as combined_jaccard or flipped. If flipped, swap the
        # passages back so the printed "Source" matches the canonical source.
        flipped = (align['source_filename'] == canonical_target_id
                   and align['target_filename'] == canonical_source_id)
        if flipped:
            src_passage = (align['target_passage'] or '').strip()
            tgt_passage = (align['source_passage'] or '').strip()
            src_len = align['length_target_passage']
            tgt_len = align['length_source_passage']
            flip_note = "  [stored flipped in DB; re-oriented for display]"
        else:
            src_passage = (align['source_passage'] or '').strip()
            tgt_passage = (align['target_passage'] or '').strip()
            src_len = align['length_source_passage']
            tgt_len = align['length_target_passage']
            flip_note = ""

        print(f"\n  Alignment {i}  (lengths: {src_len} / {tgt_len} words){flip_note}")
        print(f"\n  Source -- {pair_info['source_author_name']}  "
              f"({pair_info['source_name']}):")
        for line in src_passage.splitlines() or [src_passage]:
            print(f"    {line}")
        print(f"\n  Target -- {pair_info['target_author_name']}  "
              f"({pair_info['target_name']}):")
        for line in tgt_passage.splitlines() or [tgt_passage]:
            print(f"    {line}")
    
    # Show the Jaccard calculation for alignments.
    # Filter by pair_id for the same orientation-safety reason as above.
    cursor.execute("""
        SELECT al_jac_sim, al_jac_dis, source_total_words, target_total_words,
               length_source_passage, length_target_passage
        FROM alignments_jaccard
        WHERE pair_id = ?
    """, (pair_info['pair_id'],))
    
    jac_data = cursor.fetchone()
    if jac_data:
        print("\n--- Alignment Jaccard Calculation ---")
        print(f"  Total aligned words: {jac_data['length_source_passage']} + {jac_data['length_target_passage']}")
        print(f"  Total text words: {jac_data['source_total_words']} + {jac_data['target_total_words']}")
        print(f"  Jaccard Similarity: {jac_data['al_jac_sim']:.6f}")
        print(f"  Jaccard Distance: {jac_data['al_jac_dis']:.6f}")
    
    return alignments


# ============================================================================
# STEP 5: LOGISTIC REGRESSION COMBINATION
# ============================================================================

def load_model_coefficients(project_name):
    """
    Load model coefficients.

    For shelley-lovelace, the CANONICAL thesis methodology uses ELTeC-frozen
    weights (trained on the ELTeC-100 corpus, then applied unchanged to
    shelley-lovelace). Those constants live in score_shelley_lovelace.py and
    are the source of truth. We do NOT read influence_coefficients_shap_cv.csv
    for shelley-lovelace because that file holds weights from a model trained
    ON shelley-lovelace itself -- a different model that does not match the
    thesis percentiles.

    SHAP contribution percentages are still loaded from the project's CSV
    because they are diagnostic rather than load-bearing for the score.

    For any other project, fall back to the project's saved CSV.
    """
    if project_name == 'shelley-lovelace':
        try:
            from score_shelley_lovelace import COEFS
            coefficients = {
                'hap_jac_dis': COEFS['hap'],
                'al_jac_dis':  COEFS['al'],
                'svm_score':   COEFS['svm'],
            }
            # SHAP percentages are diagnostic only; load from CSV if present.
            shap_pct = None
            csv_path = f"./projects/{project_name}/results/influence_coefficients_shap_cv.csv"
            try:
                df = pd.read_csv(csv_path)
                shap_pct = {
                    'hap_jac_dis': df[df['variable'] == 'hap_jac_dis']['shap_contribution_pct'].values[0],
                    'al_jac_dis':  df[df['variable'] == 'al_jac_dis']['shap_contribution_pct'].values[0],
                    'svm_score':   df[df['variable'] == 'svm_score']['shap_contribution_pct'].values[0],
                }
            except Exception:
                pass
            return coefficients, shap_pct
        except ImportError as e:
            print(f"  ⚠️  Could not import ELTeC frozen weights: {e}")
            return None, None

    # Non-shelley-lovelace projects: read the project's own CSV.
    coef_path = f"./projects/{project_name}/results/influence_coefficients_shap_cv.csv"
    try:
        df = pd.read_csv(coef_path)
        coefficients = {
            'hap_jac_dis': df[df['variable'] == 'hap_jac_dis']['coefficient'].values[0],
            'al_jac_dis': df[df['variable'] == 'al_jac_dis']['coefficient'].values[0],
            'svm_score': df[df['variable'] == 'svm_score']['coefficient'].values[0],
        }
        shap_pct = {
            'hap_jac_dis': df[df['variable'] == 'hap_jac_dis']['shap_contribution_pct'].values[0],
            'al_jac_dis': df[df['variable'] == 'al_jac_dis']['shap_contribution_pct'].values[0],
            'svm_score': df[df['variable'] == 'svm_score']['shap_contribution_pct'].values[0],
        }
        return coefficients, shap_pct
    except Exception as e:
        print(f"  ⚠️  Could not load coefficients: {e}")
        return None, None


def load_scaler_parameters(project_name):
    """
    Load scaler (mean, std) parameters.

    For shelley-lovelace, use the ELTeC-frozen MEANS/STDS from
    score_shelley_lovelace.py. These were fitted on the ELTeC training split
    only and are the canonical standardization for the thesis pipeline.

    For other projects, fall back to scaler_parameters.csv, or None to let
    the caller compute approximate values from the database.
    """
    if project_name == 'shelley-lovelace':
        try:
            from score_shelley_lovelace import MEANS, STDS
            return {
                'hap_jac_dis': {'mean': MEANS['hap'], 'std': STDS['hap']},
                'al_jac_dis':  {'mean': MEANS['al'],  'std': STDS['al']},
                'svm_score':   {'mean': MEANS['svm'], 'std': STDS['svm']},
            }
        except ImportError:
            pass  # Fall through to CSV path

    scaler_path = f"./projects/{project_name}/results/scaler_parameters.csv"
    try:
        df = pd.read_csv(scaler_path)
        scaling = {}
        for _, row in df.iterrows():
            scaling[row['variable']] = {
                'mean': row['mean'],
                'std': row['std']
            }
        return scaling
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  ⚠️  Could not load scaler parameters: {e}")
        return None


def load_model_intercept(project_name):
    """
    Load model intercept.

    For shelley-lovelace, use the ELTeC-frozen INTERCEPT from
    score_shelley_lovelace.py (the canonical thesis value, -4.207...). The
    intercept stored in model_intercept.txt under projects/shelley-lovelace/
    results/ comes from a DIFFERENT model trained on shelley-lovelace itself
    and is NOT the thesis intercept.

    For other projects, fall back to model_intercept.txt.
    """
    if project_name == 'shelley-lovelace':
        try:
            from score_shelley_lovelace import INTERCEPT
            return INTERCEPT
        except ImportError:
            pass  # Fall through to file path

    intercept_path = f"./projects/{project_name}/results/model_intercept.txt"
    try:
        with open(intercept_path, 'r') as f:
            for line in f:
                if line.startswith('intercept = '):
                    return float(line.split('=')[1].strip())
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  ⚠️  Could not load intercept: {e}")
        return None


def compute_scaling_parameters(main_conn, svm_conn):
    """
    Compute the mean and std for each feature from the full dataset.
    This is a FALLBACK if saved scaler parameters are not available.
    Note: This is approximate since it computes from all data, not just training set.
    """
    cursor = main_conn.cursor()
    
    # Get mean and std for hap_jac_dis and al_jac_dis
    cursor.execute("""
        SELECT AVG(hap_jac_dis), AVG(al_jac_dis) 
        FROM combined_jaccard WHERE source_year <= target_year
    """)
    means = cursor.fetchone()
    hap_mean, al_mean = means[0], means[1]
    
    cursor.execute("""
        SELECT 
            AVG((hap_jac_dis - ?) * (hap_jac_dis - ?)),
            AVG((al_jac_dis - ?) * (al_jac_dis - ?))
        FROM combined_jaccard WHERE source_year <= target_year
    """, (hap_mean, hap_mean, al_mean, al_mean))
    variances = cursor.fetchone()
    hap_std = np.sqrt(variances[0])
    al_std = np.sqrt(variances[1])
    
    # For SVM, use reasonable estimates
    svm_mean = 0.5
    svm_std = 0.25
    
    return {
        'hap_jac_dis': {'mean': hap_mean, 'std': hap_std},
        'al_jac_dis': {'mean': al_mean, 'std': al_std},
        'svm_score': {'mean': svm_mean, 'std': svm_std}
    }


def trace_logistic_regression(main_conn, svm_conn, combined_data, svm_score, project_name):
    """
    STEP 5: Show how the three signals combine in the logistic regression.
    Computes the actual probability using trained model coefficients.
    """
    print("\n" + "=" * 70)
    print("STEP 5: LOGISTIC REGRESSION COMBINATION")
    print("=" * 70)
    
    # Load model coefficients
    coefficients, shap_pct = load_model_coefficients(project_name)
    
    # The three features
    hap_jac_dis = combined_data['hap_jac_dis']
    al_jac_dis = combined_data['al_jac_dis']
    
    print("\n--- Input Features (Raw Values) ---")
    print(f"  hap_jac_dis (Hapax Jaccard Distance):     {hap_jac_dis:.6f}")
    print(f"  al_jac_dis  (Alignment Jaccard Distance): {al_jac_dis:.6f}")
    if svm_score is not None:
        print(f"  svm_score   (Stylometric Probability):    {svm_score:.6f}")
    else:
        print("  svm_score: N/A")
    
    if svm_score is None:
        print("\n⚠️  Cannot compute logistic regression probability without SVM score")
        return None
    
    if coefficients is None:
        print("\n⚠️  Cannot compute probability - model coefficients not found")
        return None
    
    # Try to load saved scaler parameters first
    print("\n--- Standardization Parameters ---")
    scaling = load_scaler_parameters(project_name)

    if scaling:
        if project_name == 'shelley-lovelace':
            print("  ✓ Using ELTeC-frozen MEANS/STDS from score_shelley_lovelace.py")
            print("    (the canonical thesis standardization, fitted on ELTeC training split)")
        else:
            print("  ✓ Loaded from saved scaler_parameters.csv (exact training values)")
    else:
        print("  ⚠️  scaler_parameters.csv not found - computing from database (approximate)")
        print("     Run logistic_regression_shap_tt.py to generate exact parameters")
        scaling = compute_scaling_parameters(main_conn, svm_conn)
    
    print(f"\n  Feature standardization (mean, std):")
    print(f"    hap_jac_dis: μ = {scaling['hap_jac_dis']['mean']:.6f}, σ = {scaling['hap_jac_dis']['std']:.6f}")
    print(f"    al_jac_dis:  μ = {scaling['al_jac_dis']['mean']:.6f}, σ = {scaling['al_jac_dis']['std']:.6f}")
    print(f"    svm_score:   μ = {scaling['svm_score']['mean']:.6f}, σ = {scaling['svm_score']['std']:.6f}")
    
    # Standardize the features (z-score)
    hap_z = (hap_jac_dis - scaling['hap_jac_dis']['mean']) / scaling['hap_jac_dis']['std']
    al_z = (al_jac_dis - scaling['al_jac_dis']['mean']) / scaling['al_jac_dis']['std']
    svm_z = (svm_score - scaling['svm_score']['mean']) / scaling['svm_score']['std']
    
    print(f"\n--- Standardized Features (z-scores) ---")
    print(f"  hap_z = ({hap_jac_dis:.6f} - {scaling['hap_jac_dis']['mean']:.6f}) / {scaling['hap_jac_dis']['std']:.6f} = {hap_z:.4f}")
    print(f"  al_z  = ({al_jac_dis:.6f} - {scaling['al_jac_dis']['mean']:.6f}) / {scaling['al_jac_dis']['std']:.6f} = {al_z:.4f}")
    print(f"  svm_z = ({svm_score:.6f} - {scaling['svm_score']['mean']:.6f}) / {scaling['svm_score']['std']:.6f} = {svm_z:.4f}")
    
    # Model coefficients
    print(f"\n--- Model Coefficients (from training) ---")
    print(f"  β_hap = {coefficients['hap_jac_dis']:.6f}")
    print(f"  β_al  = {coefficients['al_jac_dis']:.6f}")
    print(f"  β_svm = {coefficients['svm_score']:.6f}")
    
    # Try to load saved intercept
    intercept = load_model_intercept(project_name)
    if intercept is not None:
        if project_name == 'shelley-lovelace':
            print(f"  intercept = {intercept:.6f} (ELTeC-frozen, from score_shelley_lovelace.py)")
        else:
            print(f"  intercept = {intercept:.6f} (loaded from model_intercept.txt)")
        intercept_source = "saved"
    else:
        # Estimate intercept based on class imbalance (~2.8% same-author)
        intercept = -4.5
        print(f"  intercept ≈ {intercept} (estimated - run training to save exact value)")
        intercept_source = "estimated"
    
    # Compute log-odds (linear combination)
    log_odds_no_intercept = (coefficients['hap_jac_dis'] * hap_z + 
                             coefficients['al_jac_dis'] * al_z + 
                             coefficients['svm_score'] * svm_z)
    
    print(f"\n--- Log-Odds Calculation ---")
    print(f"  log-odds = β_hap × hap_z + β_al × al_z + β_svm × svm_z + intercept")
    print(f"           = ({coefficients['hap_jac_dis']:.4f} × {hap_z:.4f}) + ({coefficients['al_jac_dis']:.4f} × {al_z:.4f}) + ({coefficients['svm_score']:.4f} × {svm_z:.4f}) + {intercept}")
    print(f"           = {coefficients['hap_jac_dis'] * hap_z:.4f} + {coefficients['al_jac_dis'] * al_z:.4f} + {coefficients['svm_score'] * svm_z:.4f} + {intercept}")
    
    log_odds = log_odds_no_intercept + intercept
    print(f"           = {log_odds:.4f}")

    # ------------------------------------------------------------------
    # Per-pair contribution breakdown
    # ------------------------------------------------------------------
    # The SHAP panel below shows the AVERAGE contribution of each feature
    # across the whole corpus. For a single pair, what we actually want is:
    # "How much did EACH signal push THIS pair's score up or down?" The
    # per-feature logit contribution (β × z) answers that directly.
    contribs = {
        'Hapax Legomena   ': coefficients['hap_jac_dis'] * hap_z,
        'Sequence Alignment': coefficients['al_jac_dis']  * al_z,
        'SVM Stylometry   ': coefficients['svm_score']   * svm_z,
    }
    total_abs = sum(abs(v) for v in contribs.values()) or 1.0  # guard div-by-zero
    print(f"\n--- Per-Pair Contribution Breakdown ---")
    print(f"  (How much each signal pushed THIS pair's logit up or down.)")
    print(f"  {'Signal':<20} {'Logit Δ':>10}  {'|share|':>9}  Direction")
    print(f"  {'-'*20} {'-'*10}  {'-'*9}  {'-'*9}")
    for name, v in contribs.items():
        share = abs(v) / total_abs * 100
        arrow = '↑ raises p' if v > 0 else '↓ lowers p'
        print(f"  {name:<20} {v:>+10.4f}  {share:>8.1f}%  {arrow}")
    print(f"  {'-'*20} {'-'*10}  {'-'*9}")
    print(f"  {'Intercept':<20} {intercept:>+10.4f}  {'':>8}   (baseline)")
    print(f"  {'Final logit':<20} {log_odds:>+10.4f}")

    # Convert to probability using sigmoid function
    probability = 1 / (1 + np.exp(-log_odds))
    
    print(f"\n--- Probability Calculation ---")
    print(f"  P(same-author) = 1 / (1 + exp(-log_odds))")
    print(f"                 = 1 / (1 + exp(-{log_odds:.4f}))")
    print(f"                 = 1 / (1 + {np.exp(-log_odds):.4f})")
    print(f"                 = {probability:.6f}")
    
    print(f"\n--- Interpretation ---")
    if intercept_source == "saved":
        print(f"  ★ Model probability (EXACT): {probability:.4f} ({probability*100:.2f}%)")
    else:
        print(f"  ★ Model probability (approximate): {probability:.4f} ({probability*100:.2f}%)")
        print(f"    (Run logistic_regression_shap_tt.py to get exact intercept)")
    
    print(f"\n  Since this is a CROSS-AUTHOR pair, a high probability indicates:")
    print(f"    → The model 'mistakes' these different authors as the same")
    print(f"    → Their writing is stylistically similar")
    print(f"    → This similarity MAY indicate literary influence")
    
    print(f"\n--- SHAP Feature Contributions (CORPUS-WIDE AVERAGES, not per-pair) ---")
    if shap_pct:
        print(f"  These are the average importance of each signal across all pairs.")
        print(f"  For THIS pair's actual contribution, see the per-pair breakdown above.")
        print(f"  Hapax Legomena:     {shap_pct['hap_jac_dis']:.1f}% of model signal (corpus-wide)")
        print(f"  SVM Stylometry:     {shap_pct['svm_score']:.1f}% of model signal (corpus-wide)")
        print(f"  Sequence Alignment: {shap_pct['al_jac_dis']:.1f}% of model signal (corpus-wide)")
    
    return probability


# ============================================================================
# STEP 6: PERCENTILE RANKING
# ============================================================================

def compute_full_model_ranking(main_conn, svm_conn, pair_info, combined_data,
                               scaling, coefficients, intercept):
    """
    Rank every eligible cross-author pair with the given model, and return
    this pair's rank plus the denominator.

    Project-independent. Pulls hap_jac_dis and al_jac_dis straight from
    combined_jaccard, joins the SVM value out of chapter_assessments the same
    way Step 3 does (target chapter row, source novel column), standardizes
    with the supplied scaler, applies the supplied coefficients, and sorts.

    Pairs with no SVM value are dropped, mirroring the scoring path's dropna.

    Returns (rank, denominator, probability, dropped_count) or None on failure.
    """
    import numpy as np
    import pandas as pd

    # 1. Every eligible cross-author pair, with the novel keys needed for the
    #    SVM lookup.
    pairs = pd.read_sql_query("""
        SELECT cj.pair_id,
               cj.hap_jac_dis,
               cj.al_jac_dis,
               a1.short_name_for_svm AS source_novel,
               a2.short_name_for_svm AS target_novel,
               a2.chapter_num        AS target_chapter
        FROM combined_jaccard cj
        JOIN all_texts a1 ON cj.source_text = a1.text_id
        JOIN all_texts a2 ON cj.target_text = a2.text_id
        WHERE cj.source_auth != cj.target_auth
          AND cj.source_year <= cj.target_year
    """, main_conn)

    if pairs.empty:
        return None

    # 2. chapter_assessments is wide: one row per (novel, chapter), one column
    #    per novel. Melt it into (target_novel, target_chapter, source_novel,
    #    svm_score) so it can be merged.
    ca = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
    id_cols = ['novel', 'number']
    value_cols = [c for c in ca.columns if c not in id_cols]
    svm_long = ca.melt(id_vars=id_cols, value_vars=value_cols,
                       var_name='source_novel', value_name='svm_score')
    svm_long = svm_long.rename(columns={'novel': 'target_novel',
                                        'number': 'target_chapter'})
    svm_long['target_chapter'] = svm_long['target_chapter'].astype(str)
    pairs['target_chapter'] = pairs['target_chapter'].astype(str)

    merged = pairs.merge(svm_long,
                         on=['target_novel', 'target_chapter', 'source_novel'],
                         how='left')

    dropped = int(merged['svm_score'].isna().sum())
    merged = merged.dropna(subset=['svm_score'])
    if merged.empty:
        return None

    # 3. Standardize, apply coefficients, sigmoid.
    hz = (merged['hap_jac_dis'] - scaling['hap_jac_dis']['mean']) / scaling['hap_jac_dis']['std']
    az = (merged['al_jac_dis']  - scaling['al_jac_dis']['mean'])  / scaling['al_jac_dis']['std']
    sz = (merged['svm_score']   - scaling['svm_score']['mean'])   / scaling['svm_score']['std']

    logit = (coefficients['hap_jac_dis'] * hz
             + coefficients['al_jac_dis'] * az
             + coefficients['svm_score']  * sz
             + intercept)
    merged['prob'] = 1.0 / (1.0 + np.exp(-logit))

    # 4. Rank descending by probability; rank 1 is the strongest pair.
    merged = merged.sort_values('prob', ascending=False).reset_index(drop=True)
    merged['rank'] = merged.index + 1

    hit = merged[merged['pair_id'] == pair_info['pair_id']]
    if hit.empty:
        return ('dropped', len(merged), None, dropped)

    row = hit.iloc[0]
    return (int(row['rank']), len(merged), float(row['prob']), dropped)


def trace_percentile_ranking(main_conn, svm_conn, pair_info, combined_data, project_name):
    """
    STEP 6: Show where this pair ranks among all cross-author pairs.

    The CANONICAL ranking is the full-model percentile -- the same number the
    thesis reports. For shelley-lovelace it comes from score_shelley_lovelace,
    which is the thesis source of truth. For every other project it is computed
    inline by compute_full_model_ranking(), using the same scaler, coefficients
    and intercept that Step 5 printed.

    The hapax-only ranking is retained as a secondary diagnostic: it shows
    where this pair sits on the single strongest signal in isolation, which
    is useful when comparing alignment-driven vs hapax-driven hits.
    """
    print("\n" + "=" * 70)
    print("STEP 6: PERCENTILE RANKING")
    print("=" * 70)

    full_model_done = False

    # ------------------------------------------------------------------
    # Path A: shelley-lovelace uses the canonical scorer.
    # ------------------------------------------------------------------
    if project_name == 'shelley-lovelace':
        print("\n--- Full-Model Ranking (canonical) ---")
        print("  Scoring the full corpus with frozen ELTeC weights...")
        print("  (~10-20 seconds on first call.)")
        try:
            from score_shelley_lovelace import load_and_score, rank_pairs

            df = load_and_score()
            df_cross = rank_pairs(df, cross_author_only=True)
            N_cross = len(df_cross)

            match = df_cross[df_cross['pair_id'] == pair_info['pair_id']]
            if len(match) > 0:
                row = match.iloc[0]
                rank_int = int(row['rank'])
                percentile = (1 - rank_int / N_cross) * 100
                top_pct = rank_int / N_cross * 100

                print(f"\n  Cross-author denominator:      {N_cross:,}")
                print(f"  This pair's p (full model):    {row['prob']:.6f}")
                print(f"  Rank:                          {rank_int:,} of {N_cross:,}")
                print(f"  Percentile (higher = better):  {percentile:.4f}th")
                print(f"  Top:                           {top_pct:.4f}%")

                print(f"\n  Per-signal z-scores for this pair:")
                print(f"    hap_z = {row['hap_z']:+.4f}")
                print(f"    al_z  = {row['al_z']:+.4f}")
                print(f"    svm_z = {row['svm_z']:+.4f}")
                print(f"  TextPAIR alignments on this pair: {int(row['n_align'])}")
                full_model_done = True
            else:
                df_all = rank_pairs(df, cross_author_only=False)
                match_all = df_all[df_all['pair_id'] == pair_info['pair_id']]
                if len(match_all) > 0:
                    row = match_all.iloc[0]
                    sa_flag = '[same-author]' if row['is_same_author'] else '[cross]'
                    print(f"\n  Pair found in ALL-PAIRS ranking but NOT in cross-author "
                          f"ranking: {sa_flag}")
                    print(f"  All-pairs rank:                {int(row['rank']):,} of "
                          f"{len(df_all):,}")
                    print(f"  p (full model):                {row['prob']:.6f}")
                    if row['is_same_author']:
                        print(f"\n  Note: same-author pairs are excluded from the")
                        print(f"        cross-author influence-candidate denominator.")
                    full_model_done = True
                else:
                    print(f"\n  pair_id {pair_info['pair_id']} not in scored output.")
                    print(f"  Most likely cause: no SVM score available for this pair")
                    print(f"  (dropped by load_and_score's dropna on svm_score).")
        except ImportError as e:
            print(f"\n  Could not import score_shelley_lovelace: {e}")
        except Exception as e:
            print(f"\n  Full-model ranking failed: {e}")
            print(f"  Falling back to the inline computation below.")

    # ------------------------------------------------------------------
    # Path B: any other project, computed inline.
    # ------------------------------------------------------------------
    if not full_model_done:
        print("\n--- Full-Model Ranking (computed inline) ---")
        print("  Scoring every eligible cross-author pair with this project's")
        print("  scaler, coefficients and intercept. This takes ~30-60 seconds")
        print("  on a corpus the size of ELTeC.")

        scaling = load_scaler_parameters(project_name)
        coefficients, _ = load_model_coefficients(project_name)
        intercept = load_model_intercept(project_name)

        missing = [n for n, v in (('scaler', scaling),
                                  ('coefficients', coefficients),
                                  ('intercept', intercept)) if v is None]
        if missing:
            print(f"\n  Cannot compute: missing {', '.join(missing)}.")
            print(f"  Run the logistic regression for this project to generate them.")
        else:
            try:
                result = compute_full_model_ranking(
                    main_conn, svm_conn, pair_info, combined_data,
                    scaling, coefficients, intercept)
            except Exception as e:
                result = None
                print(f"\n  Inline ranking failed: {type(e).__name__}: {e}")

            if result is None:
                print("\n  No eligible pairs found, or the SVM join produced nothing.")
            elif result[0] == 'dropped':
                _, denom, _, dropped = result
                print(f"\n  Cross-author denominator:      {denom:,}")
                print(f"  This pair is NOT in the scored set. It has no SVM value,")
                print(f"  so it was dropped along with {dropped:,} others.")
                full_model_done = True
            else:
                rank_int, denom, prob, dropped = result
                percentile = (1 - rank_int / denom) * 100
                top_pct = rank_int / denom * 100
                print(f"\n  Cross-author denominator:      {denom:,}")
                if dropped:
                    print(f"  Pairs dropped for no SVM value: {dropped:,}")
                print(f"  This pair's p (full model):    {prob:.6f}")
                print(f"  Rank:                          {rank_int:,} of {denom:,}")
                print(f"  Percentile (higher = better):  {percentile:.4f}th")
                print(f"  Top:                           {top_pct:.4f}%")
                print(f"\n  Note: the SVM value here is resolved the way Step 3 does it,")
                print(f"        from all_texts.short_name_for_svm. logistic_regression.py")
                print(f"        resolves it differently; run with --audit (Receipt 1) to")
                print(f"        see both.")
                full_model_done = True

    # ------------------------------------------------------------------
    # Hapax-only ranking (secondary diagnostic, always shown)
    # ------------------------------------------------------------------
    cursor = main_conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM combined_jaccard
        WHERE source_auth != target_auth
          AND source_year <= target_year
    """)
    total_cross_author = cursor.fetchone()[0]

    hap_jac_dis = combined_data['hap_jac_dis']
    cursor.execute("""
        SELECT COUNT(*) FROM combined_jaccard
        WHERE source_auth != target_auth
          AND source_year <= target_year
          AND hap_jac_dis < ?
    """, (hap_jac_dis,))
    pairs_more_similar = cursor.fetchone()[0]
    hap_percentile = (1 - pairs_more_similar / total_cross_author) * 100

    print(f"\n--- Hapax-Only Ranking (secondary diagnostic) ---")
    print(f"  Total cross-author pairs (DB filter): {total_cross_author:,}")
    print(f"  This pair's hap_jac_dis:              {hap_jac_dis:.6f}")
    print(f"  Pairs with LOWER distance:            {pairs_more_similar:,}")
    print(f"  Hapax-only percentile:                {hap_percentile:.2f}th")
    if full_model_done:
        print(f"\n  Note: the canonical rank is the full-model rank above. The")
        print(f"        hapax-only number is shown so you can see how much of the")
        print(f"        signal is being carried by the hapax channel alone.")
    else:
        print(f"\n  Note: full-model rank above unavailable; this hapax-only number")
        print(f"        is an APPROXIMATE proxy, not the thesis percentile.")

    return None


# ============================================================================
# MAIN TRACE FUNCTION
# ============================================================================

def trace_pair(source_author=None, source_chapter=None, 
               target_author=None, target_chapter=None, pair_id=None,
               random_pair=False, show_receipts=False):
    """
    Main function to trace a single pair through the entire pipeline.

    show_receipts: also print the audit receipt sections from audit_receipts.py,
    which compare what the pipeline did against what the thesis says it did.
    """
    print("\n" + "=" * 70)
    print("SEXTANT SINGLE PAIR AUDIT TRACE")
    print("=" * 70)
    
    main_conn, svm_conn, project_name = get_db_connections()
    
    # Find the pair
    if random_pair:
        print("\nSelecting a random cross-author pair...")
        pair_info = get_random_cross_author_pair(main_conn)
        if pair_info is None:
            print("❌ ERROR: Could not find any cross-author pairs in database")
            return
    elif pair_id:
        pair_info = find_pair_by_id(main_conn, pair_id)
    else:
        pair_info = find_pair_by_authors_and_chapters(
            main_conn, source_author, source_chapter, target_author, target_chapter
        )
    
    if pair_info is None:
        return
    
    print(f"\n" + "=" * 70)
    print("PAIR IDENTIFICATION")
    print("=" * 70)
    print(f"\n✓ Found pair:")
    print(f"  Source: {pair_info['source_author_name']} - {pair_info['source_name']}")
    print(f"  Target: {pair_info['target_author_name']} - {pair_info['target_name']}")
    print(f"  Pair ID: {pair_info['pair_id']}")
    print(f"  Years: {pair_info['source_year']} → {pair_info['target_year']}")
    
    print(f"\n" + "-" * 70)
    print("TO REPLICATE THIS EXACT TRACE, RUN:")
    replicate = f"  python trace_single_pair.py --pair-id {pair_info['pair_id']}"
    if show_receipts:
        replicate += " --audit"
    print(replicate)
    print("-" * 70)
    
    # Step 1: Raw data
    combined_data = trace_combined_jaccard(main_conn, pair_info['pair_id'])
    if combined_data is None:
        return
    
    # Step 2: Hapax legomena
    trace_hapax_legomena(main_conn, pair_info, combined_data)
    
    # Step 3: SVM stylometry
    svm_score = trace_svm_stylometry(svm_conn, pair_info)
    
    # Step 4: Sequence alignments
    trace_sequence_alignments(main_conn, pair_info, project_name)
    
    # Step 5: Logistic regression
    probability = trace_logistic_regression(main_conn, svm_conn, combined_data, svm_score, project_name)
    
    # Step 6: Percentile ranking (full-model + hapax-only diagnostic)
    trace_percentile_ranking(main_conn, svm_conn, pair_info, combined_data, project_name)

    # Audit receipts: what the pipeline did vs what the thesis says it did.
    if show_receipts:
        if RECEIPTS_AVAILABLE:
            print_all_receipts(main_conn, svm_conn, pair_info, combined_data,
                               svm_score, project_name)
        else:
            print("\n⚠️  --audit requested but audit_receipts.py could not be imported.")
            print("   Put audit_receipts.py in the same directory as this script.")
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT TRACE COMPLETE")
    print("=" * 70)
    print("\nThis trace shows exactly how Sextant processed this text pair.")
    print("All values are pulled directly from the database tables created")
    print("during the pipeline execution. The percentile ranking places this")
    print("pair in context among millions of other cross-author comparisons.")
    if not show_receipts and RECEIPTS_AVAILABLE:
        print("\nRe-run with --audit to compare what the pipeline did against")
        print("what the thesis says it did, for this pair.")
    
    main_conn.close()
    svm_conn.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trace a single text pair through the Sextant pipeline",
        epilog="""
Examples:
  python trace_single_pair.py                    # Random cross-author pair
  python trace_single_pair.py --pair-id 4448692  # Specific pair by ID
  python trace_single_pair.py --pair-id 4448692 --audit
  python trace_single_pair.py --source Eliot --source-chapter 79 --target Lawrence --target-chapter 29
        """
    )
    
    parser.add_argument('--source', type=str, help='Source author name (e.g., "Eliot")')
    parser.add_argument('--source-chapter', type=int, help='Source chapter number')
    parser.add_argument('--target', type=str, help='Target author name (e.g., "Lawrence")')
    parser.add_argument('--target-chapter', type=int, help='Target chapter number')
    parser.add_argument('--pair-id', type=int, help='Direct pair_id from database')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--audit', action='store_true',
                        help='Print audit receipt sections (needs audit_receipts.py)')
    
    args = parser.parse_args()
    
    if args.interactive:
        print("\n=== SEXTANT AUDIT TRACE - Interactive Mode ===\n")
        source = input("Source author (e.g., Eliot): ").strip()
        source_ch = int(input("Source chapter number: ").strip())
        target = input("Target author (e.g., Lawrence): ").strip()
        target_ch = int(input("Target chapter number: ").strip())
        
        trace_pair(source, source_ch, target, target_ch, show_receipts=args.audit)
        
    elif args.pair_id:
        trace_pair(pair_id=args.pair_id, show_receipts=args.audit)
        
    elif args.source and args.target:
        trace_pair(args.source, args.source_chapter, args.target, args.target_chapter,
                   show_receipts=args.audit)
        
    else:
        # Default: random cross-author pair
        print("\nNo arguments provided. Selecting a random cross-author pair...")
        print("(Use --pair-id to replicate a specific trace)\n")
        trace_pair(random_pair=True, show_receipts=args.audit)


if __name__ == "__main__":
    main()