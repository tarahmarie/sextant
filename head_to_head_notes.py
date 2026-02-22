#!/usr/bin/env python3
"""Percy vs Mary head-to-head by Note (corrected Final Run).

For each Lovelace Note (A, B, D, E, F, G), find the best-scoring
Percy pair and best-scoring Mary pair.
"""

import sqlite3
import re
import numpy as np
from scipy.special import expit

MAIN_DB = "projects/shelley-lovelace-final-500w/db/shelley-lovelace.db"
SVM_DB = "projects/shelley-lovelace-final-500w/db/svm.db"

# Frozen ELTeC parameters
INTERCEPT = -4.2041
B_HAP = -1.2328
B_AL = -0.1581
B_SVM = 0.1687
HAP_MEAN, HAP_STD = 0.9498, 0.00991
SVM_MEAN, SVM_STD = 0.3241, 0.2570
AL_MEAN, AL_STD = 0.9999707, 0.000261

CHAPTER_PATTERN = re.compile(
    r'^(?:chapter|section|letter|part|book|canto|act|scene|note|volume|vol)_(.+)$',
    re.IGNORECASE
)

def extract_chapter_num(chapter_str):
    if not chapter_str:
        return None
    m = CHAPTER_PATTERN.match(chapter_str)
    return m.group(1) if m else chapter_str

def score(hap, svm, al):
    z_hap = (hap - HAP_MEAN) / HAP_STD
    z_svm = (svm - SVM_MEAN) / SVM_STD
    z_al = ((al if al else 1.0) - AL_MEAN) / AL_STD
    logit = INTERCEPT + B_HAP * z_hap + B_SVM * z_svm + B_AL * z_al
    return expit(logit)

# --- Load metadata ---
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
    }

# Find Notes by filename pattern "-Note_X" or chapter_num == single letter
note_texts = {}
for tid, t in all_texts.items():
    fn = t['filename'] or ''
    ch = t['chapter_num'] or ''
    for note in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        if f'-Note_{note}' in fn or ch == note:
            note_texts.setdefault(note, []).append(tid)

# Also include Menabrea sections as targets
menabrea_tids = [tid for tid, t in all_texts.items() if t['author_id'] == 13]

print("Note texts found:")
for note in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    tids = note_texts.get(note, [])
    if tids:
        fns = [all_texts[t]['filename'] for t in tids]
        print(f"  Note {note}: {fns}")
    else:
        print(f"  Note {note}: NOT FOUND (below 500w?)")

mary_ids = set(tid for tid, t in all_texts.items() if 'Shelley_Mary' in (t['filename'] or ''))
percy_ids = set(tid for tid, t in all_texts.items() if 'Shelley_Percy' in (t['filename'] or ''))
print(f"\nMary texts: {len(mary_ids)}, Percy texts: {len(percy_ids)}")

# --- Load SVM ---
svm_conn = sqlite3.connect(SVM_DB)
svm_cur = svm_conn.cursor()
svm_cur.execute("PRAGMA table_info(chapter_assessments)")
svm_columns = [row[1] for row in svm_cur.fetchall()]
work_columns = [c for c in svm_columns if c not in ('novel', 'number')]

svm_cur.execute("SELECT * FROM chapter_assessments")
svm_data = {}
for row in svm_cur.fetchall():
    svm_data[(row[0], str(row[1]))] = {work_columns[i]: row[i+2] for i in range(len(work_columns))}
svm_conn.close()

def get_svm(src_tid, tgt_tid):
    src = all_texts.get(src_tid)
    tgt = all_texts.get(tgt_tid)
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

# --- Head to head ---
print()
print(f"{'Note':<6} {'Percy p':>8} {'Percy Source':<40} {'Mary p':>8} {'Mary Source':<40} {'Winner'}")
print("-" * 120)

for note in ['A', 'B', 'D', 'E', 'F', 'G']:
    tids = note_texts.get(note, [])
    if not tids:
        print(f"{note:<6} Note not found in database")
        continue

    # Percy's best pair to this Note
    percy_best_p = -1
    percy_best_src = ""
    percy_best_hap = 0
    percy_best_svm = 0
    for pid in percy_ids:
        for ntid in tids:
            cur.execute("""
                SELECT hap_jac_dis, al_jac_dis FROM combined_jaccard
                WHERE (source_text = ? AND target_text = ?)
                   OR (source_text = ? AND target_text = ?)
            """, (pid, ntid, ntid, pid))
            for hap, al in cur.fetchall():
                svm = get_svm(pid, ntid)
                if svm is None:
                    continue
                p = score(hap, svm, al)
                if p > percy_best_p:
                    percy_best_p = p
                    percy_best_src = all_texts[pid]['filename']
                    percy_best_hap = hap
                    percy_best_svm = svm

    # Mary's best pair to this Note
    mary_best_p = -1
    mary_best_src = ""
    mary_best_hap = 0
    mary_best_svm = 0
    for mid in mary_ids:
        for ntid in tids:
            cur.execute("""
                SELECT hap_jac_dis, al_jac_dis FROM combined_jaccard
                WHERE (source_text = ? AND target_text = ?)
                   OR (source_text = ? AND target_text = ?)
            """, (mid, ntid, ntid, mid))
            for hap, al in cur.fetchall():
                svm = get_svm(mid, ntid)
                if svm is None:
                    continue
                p = score(hap, svm, al)
                if p > mary_best_p:
                    mary_best_p = p
                    mary_best_src = all_texts[mid]['filename']
                    mary_best_hap = hap
                    mary_best_svm = svm

    def short(fn):
        if not fn:
            return "NONE"
        return fn.split('--')[-1][:38] if '--' in fn else fn[:38]

    winner = "PERCY" if percy_best_p > mary_best_p else "MARY" if mary_best_p > percy_best_p else "TIE"
    margin = abs(percy_best_p - mary_best_p)
    if margin > 0.05:
        winner += " (large)"
    elif margin < 0.01:
        winner += " (barely)"

    print(f"{note:<6} {percy_best_p:>8.4f} {short(percy_best_src):<40} {mary_best_p:>8.4f} {short(mary_best_src):<40} {winner}")

# Also show Menabrea sections
print()
print("--- Also: Percy vs Mary on Menabrea sections ---")
print(f"{'Sec':<6} {'Percy p':>8} {'Percy Source':<40} {'Mary p':>8} {'Mary Source':<40} {'Winner'}")
print("-" * 120)

for mtid in sorted(menabrea_tids):
    sec = all_texts[mtid]['chapter_num']

    percy_best_p = -1
    percy_best_src = ""
    for pid in percy_ids:
        cur.execute("""
            SELECT hap_jac_dis, al_jac_dis FROM combined_jaccard
            WHERE (source_text = ? AND target_text = ?)
               OR (source_text = ? AND target_text = ?)
        """, (pid, mtid, mtid, pid))
        for hap, al in cur.fetchall():
            svm = get_svm(pid, mtid)
            if svm is None:
                continue
            p = score(hap, svm, al)
            if p > percy_best_p:
                percy_best_p = p
                percy_best_src = all_texts[pid]['filename']

    mary_best_p = -1
    mary_best_src = ""
    for mid in mary_ids:
        cur.execute("""
            SELECT hap_jac_dis, al_jac_dis FROM combined_jaccard
            WHERE (source_text = ? AND target_text = ?)
               OR (source_text = ? AND target_text = ?)
        """, (mid, mtid, mtid, mid))
        for hap, al in cur.fetchall():
            svm = get_svm(mid, mtid)
            if svm is None:
                continue
            p = score(hap, svm, al)
            if p > mary_best_p:
                mary_best_p = p
                mary_best_src = all_texts[mid]['filename']

    def short(fn):
        if not fn:
            return "NONE"
        return fn.split('--')[-1][:38] if '--' in fn else fn[:38]

    winner = "PERCY" if percy_best_p > mary_best_p else "MARY" if mary_best_p > percy_best_p else "TIE"
    margin = abs(percy_best_p - mary_best_p)
    if margin > 0.05:
        winner += " (large)"
    elif margin < 0.01:
        winner += " (barely)"

    print(f"s{sec:<5} {percy_best_p:>8.4f} {short(percy_best_src):<40} {mary_best_p:>8.4f} {short(mary_best_src):<40} {winner}")

conn.close()