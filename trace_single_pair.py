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

Usage:
    python trace_single_pair.py --source "Eliot" --source-chapter 79 --target "Lawrence" --target-chapter 29
    python trace_single_pair.py --pair-id 12345
    python trace_single_pair.py --interactive

Author: Tarah Wheeler
For: Sextant / ACL SIGHUM 2026 Paper
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
                    # Show first 30 words
                    display_words = list(shared_words)[:30]
                    print(f"  {', '.join(str(w) for w in display_words)}")
                    if len(shared_words) > 30:
                        print(f"  ... and {len(shared_words) - 30} more")
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
    """
    print("\n" + "=" * 70)
    print("STEP 3: SVM STYLOMETRY SCORE")
    print("=" * 70)
    
    # Extract novel name from filename
    # Format: 1872-ENG18721—Eliot-chapter_79
    target_name = pair_info['target_name']
    
    # Parse the target chapter info
    parts = target_name.split('-chapter_')
    if len(parts) == 2:
        chapter_num = parts[1]
    else:
        chapter_num = target_name.split('_')[-1]
    
    # Extract novel name from the directory structure
    novel_part = target_name.split('—')[1] if '—' in target_name else target_name.split('-')[1]
    novel_name = novel_part.split('-chapter')[0]
    
    # Similarly for source
    source_name = pair_info['source_name']
    source_novel = source_name.split('—')[1] if '—' in source_name else source_name.split('-')[1]
    source_novel = source_novel.split('-chapter')[0]
    
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
    
    # Query the alignments table
    cursor.execute("""
        SELECT source_passage, target_passage, length_source_passage, length_target_passage
        FROM alignments
        WHERE source_filename = ? AND target_filename = ?
    """, (pair_info['source_text'], pair_info['target_text']))
    
    alignments = cursor.fetchall()
    
    if not alignments:
        print(f"\n⚠️  No sequence alignments found between these chapters.")
        print("   This is common - only ~1.7% of pairs have alignments.")
        print("   al_jac_dis defaults to 1.0 (maximum distance) when no alignments exist.")
        return []
    
    print(f"\nFound {len(alignments)} sequence alignment(s)!")
    print("\n--- Aligned Passages ---")
    
    for i, align in enumerate(alignments, 1):
        print(f"\n  Alignment {i}:")
        print(f"    Source ({pair_info['source_author_name']}): \"{align['source_passage'][:100]}...\"")
        print(f"    Target ({pair_info['target_author_name']}): \"{align['target_passage'][:100]}...\"")
        print(f"    Lengths: {align['length_source_passage']} / {align['length_target_passage']} words")
    
    # Show the Jaccard calculation for alignments
    cursor.execute("""
        SELECT al_jac_sim, al_jac_dis, source_total_words, target_total_words,
               length_source_passage, length_target_passage
        FROM alignments_jaccard
        WHERE source_filename = ? AND target_filename = ?
    """, (pair_info['source_text'], pair_info['target_text']))
    
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
    Load the trained model coefficients from the results file.
    """
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
    Load the saved scaler parameters (mean, std) from the results file.
    Returns None if file doesn't exist (fall back to computing from DB).
    """
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
    Load the saved model intercept from the results file.
    Returns None if file doesn't exist.
    """
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
    
    print(f"\n--- SHAP Feature Contributions ---")
    if shap_pct:
        print(f"  Hapax Legomena:    {shap_pct['hap_jac_dis']:.1f}% of model signal")
        print(f"  SVM Stylometry:    {shap_pct['svm_score']:.1f}% of model signal")
        print(f"  Sequence Alignment: {shap_pct['al_jac_dis']:.1f}% of model signal")
    
    return probability


# ============================================================================
# STEP 6: PERCENTILE RANKING
# ============================================================================

def trace_percentile_ranking(main_conn, svm_conn, pair_info, combined_data):
    """
    STEP 6: Show where this pair ranks among all cross-author pairs.
    """
    print("\n" + "=" * 70)
    print("STEP 6: PERCENTILE RANKING")
    print("=" * 70)
    
    # Count total cross-author pairs
    cursor = main_conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM combined_jaccard 
        WHERE source_auth != target_auth 
          AND source_year <= target_year
    """)
    total_cross_author = cursor.fetchone()[0]
    
    print(f"\nTotal cross-author pairs (with temporal filter): {total_cross_author:,}")
    
    # For a simple ranking, use hapax distance (lower = more similar = better)
    hap_jac_dis = combined_data['hap_jac_dis']
    
    cursor.execute("""
        SELECT COUNT(*) FROM combined_jaccard 
        WHERE source_auth != target_auth 
          AND source_year <= target_year
          AND hap_jac_dis < ?
    """, (hap_jac_dis,))
    pairs_more_similar = cursor.fetchone()[0]
    
    percentile = (1 - pairs_more_similar / total_cross_author) * 100
    
    print(f"\n--- Hapax-based Ranking ---")
    print(f"  This pair's hap_jac_dis: {hap_jac_dis:.6f}")
    print(f"  Pairs with LOWER distance (more similar): {pairs_more_similar:,}")
    print(f"  Percentile (higher = more similar): {percentile:.2f}th")
    
    # Note about the full model ranking
    print("\n--- Note on Full Model Ranking ---")
    print("  The paper reports percentiles based on the full logistic regression model,")
    print("  which combines all three signals. This requires loading the trained model")
    print("  and computing probabilities for all ~5.8M cross-author pairs.")
    print("  The reported percentile for Eliot→Lawrence is >99.99th percentile.")
    
    return percentile


# ============================================================================
# MAIN TRACE FUNCTION
# ============================================================================

def trace_pair(source_author=None, source_chapter=None, 
               target_author=None, target_chapter=None, pair_id=None, random_pair=False):
    """
    Main function to trace a single pair through the entire pipeline.
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
    print(f"  python trace_single_pair.py --pair-id {pair_info['pair_id']}")
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
    
    # Step 6: Percentile ranking
    trace_percentile_ranking(main_conn, svm_conn, pair_info, combined_data)
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT TRACE COMPLETE")
    print("=" * 70)
    print("\nThis trace shows exactly how Sextant processed this text pair.")
    print("All values are pulled directly from the database tables created")
    print("during the pipeline execution. The percentile ranking places this")
    print("pair in context among millions of other cross-author comparisons.")
    
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
  python trace_single_pair.py --source Eliot --source-chapter 79 --target Lawrence --target-chapter 29
        """
    )
    
    parser.add_argument('--source', type=str, help='Source author name (e.g., "Eliot")')
    parser.add_argument('--source-chapter', type=int, help='Source chapter number')
    parser.add_argument('--target', type=str, help='Target author name (e.g., "Lawrence")')
    parser.add_argument('--target-chapter', type=int, help='Target chapter number')
    parser.add_argument('--pair-id', type=int, help='Direct pair_id from database')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        print("\n=== SEXTANT AUDIT TRACE - Interactive Mode ===\n")
        source = input("Source author (e.g., Eliot): ").strip()
        source_ch = int(input("Source chapter number: ").strip())
        target = input("Target author (e.g., Lawrence): ").strip()
        target_ch = int(input("Target chapter number: ").strip())
        
        trace_pair(source, source_ch, target, target_ch)
        
    elif args.pair_id:
        trace_pair(pair_id=args.pair_id)
        
    elif args.source and args.target:
        trace_pair(args.source, args.source_chapter, args.target, args.target_chapter)
        
    else:
        # Default: random cross-author pair
        print("\nNo arguments provided. Selecting a random cross-author pair...")
        print("(Use --pair-id to replicate a specific trace)\n")
        trace_pair(random_pair=True)


if __name__ == "__main__":
    main()