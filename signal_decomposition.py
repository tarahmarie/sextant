#!/usr/bin/env python3
"""Recompute signal decomposition from corrected Final Run (500w canonical).

Uses combined_jaccard (main db) for hapax + alignment, and
chapter_assessments (svm.db) for SVM scores. Distinguishes
Mary/Percy Shelley by source_filename.
"""

import sqlite3
import re
import numpy as np

MAIN_DB = "projects/shelley-lovelace-final-500w/db/shelley-lovelace.db"
SVM_DB = "projects/shelley-lovelace-final-500w/db/svm.db"

# Frozen ELTeC z-score parameters
HAP_MEAN, HAP_STD = 0.9498, 0.00991
SVM_MEAN, SVM_STD = 0.3241, 0.2570
AL_MEAN, AL_STD = 0.9999707, 0.000261

# CHAPTER_PATTERN: extract number after keyword prefix (matches do_svm.py fix)
CHAPTER_PATTERN = re.compile(
    r'^(?:chapter|section|letter|part|book|canto|act|scene|note|volume|vol)_(.+)$',
    re.IGNORECASE
)

def extract_chapter_num(chapter_str):
    if not chapter_str:
        return None
    m = CHAPTER_PATTERN.match(chapter_str)
    return m.group(1) if m else chapter_str

# --- Load text metadata from main db ---
conn = sqlite3.connect(MAIN_DB)
cur = conn.cursor()

cur.execute("SELECT text_id, source_filename, author_id, short_name_for_svm, chapter_num FROM all_texts")
all_texts = {}
for text_id, filename, author_id, svm_name, chapter_num in cur.fetchall():
    all_texts[text_id] = {
        'filename': filename,
        'author_id': author_id,
        'svm_name': svm_name,
        'chapter_num': chapter_num,
        'is_mary': 'Shelley_Mary' in (filename or ''),
        'is_percy': 'Shelley_Percy' in (filename or ''),
    }

target_ids = set(tid for tid, t in all_texts.items() if t['author_id'] in (7, 13))
print(f"Target texts (Lovelace + Menabrea): {len(target_ids)}")

# --- Load SVM scores from svm.db ---
svm_conn = sqlite3.connect(SVM_DB)
svm_cur = svm_conn.cursor()

svm_cur.execute("PRAGMA table_info(chapter_assessments)")
svm_columns = [row[1] for row in svm_cur.fetchall()]
work_columns = [c for c in svm_columns if c not in ('novel', 'number')]

svm_cur.execute("SELECT * FROM chapter_assessments")
svm_data = {}
for row in svm_cur.fetchall():
    novel = row[0]
    number = str(row[1])
    scores = {work_columns[i]: row[i+2] for i in range(len(work_columns))}
    svm_data[(novel, number)] = scores

svm_conn.close()
print(f"SVM assessments loaded: {len(svm_data)} rows, {len(work_columns)} work columns")

def get_svm_score(source_tid, target_tid):
    src = all_texts.get(source_tid)
    tgt = all_texts.get(target_tid)
    if not src or not tgt:
        return None
    src_novel = src['svm_name']
    src_num = extract_chapter_num(src['chapter_num'])
    tgt_work = tgt['svm_name']
    if not src_novel or not src_num or not tgt_work:
        return None
    if tgt_work not in work_columns:
        return None
    key = (src_novel, src_num)
    if key not in svm_data:
        return None
    return svm_data[key].get(tgt_work)

# --- Define author groups ---
authors = {
    'Babbage': lambda t: t['author_id'] == 3,
    'Somerville': lambda t: t['author_id'] == 20,
    'Mary Shelley': lambda t: t['author_id'] == 14 and t['is_mary'],
    'Percy Shelley': lambda t: t['author_id'] == 14 and t['is_percy'],
    'Wollstonecraft': lambda t: t['author_id'] == 19,
    'Godwin': lambda t: t['author_id'] == 4,
    'Volney': lambda t: t['author_id'] == 6,
    'Goldsmith': lambda t: t['author_id'] == 10,
    'Byron': lambda t: t['author_id'] == 18,
    'Milton': lambda t: t['author_id'] == 23,
}

author_tids = {}
for name, pred in authors.items():
    author_tids[name] = set(tid for tid, t in all_texts.items() if pred(t))

# --- Pull pairs and decompose ---
print()
print(f"{'Author':<20} {'Pairs':>6} {'w/SVM':>6}  {'Hapax z':>8}  {'SVM z':>8}  {'Align z':>8}  {'Profile'}")
print("-" * 80)

for author_name in authors:
    src_ids = author_tids[author_name]
    if not src_ids:
        print(f"{author_name:<20} NO TEXTS")
        continue

    src_str = ','.join(str(x) for x in src_ids)
    tgt_str = ','.join(str(x) for x in target_ids)

    cur.execute(f"""
        SELECT source_text, target_text, hap_jac_dis, al_jac_dis
        FROM combined_jaccard
        WHERE (source_text IN ({src_str}) AND target_text IN ({tgt_str}))
           OR (source_text IN ({tgt_str}) AND target_text IN ({src_str}))
    """)
    rows = cur.fetchall()

    hap_vals = []
    svm_vals = []
    al_vals = []
    total_pairs = len(rows)

    for src_t, tgt_t, hap, al in rows:
        if src_t in target_ids:
            src_t, tgt_t = tgt_t, src_t
        svm = get_svm_score(src_t, tgt_t)
        hap_vals.append(hap)
        al_vals.append(al if al is not None else 1.0)
        if svm is not None:
            svm_vals.append(svm)

    if not hap_vals:
        print(f"{author_name:<20} NO PAIRS")
        continue

    hap_z = np.mean([(h - HAP_MEAN) / HAP_STD for h in hap_vals])
    al_z = np.mean([(a - AL_MEAN) / AL_STD for a in al_vals])
    svm_z = np.mean([(s - SVM_MEAN) / SVM_STD for s in svm_vals]) if svm_vals else float('nan')

    parts = []
    if hap_z < -0.5:
        parts.append("hapax+")
    elif hap_z > 0.5:
        parts.append("hapax-")
    if not np.isnan(svm_z):
        if svm_z > 0.3:
            parts.append("SVM+")
        elif svm_z < -0.3:
            parts.append("SVM-")
    profile = ", ".join(parts) if parts else "neutral"

    print(f"{author_name:<20} {total_pairs:>6} {len(svm_vals):>6}  {hap_z:>+8.2f}  {svm_z:>+8.2f}  {al_z:>+8.2f}  {profile}")

conn.close()
print()
print("Negative hapax z = closer vocabulary (good). Positive SVM z = more similar style (good).")
print("Alignment z near 0 = no sequence matches (expected for most pairs).")