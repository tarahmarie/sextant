#!/usr/bin/env python3
"""
Report: Every author in the corpus compared to Lovelace.
Run from the sextant directory.
"""
import sqlite3, numpy as np, pandas as pd

project = 'shelley-lovelace'
main_conn = sqlite3.connect(f'./projects/{project}/db/{project}.db')
svm_conn = sqlite3.connect(f'./projects/{project}/db/svm.db')

# ELTeC scaler parameters
means = {'hap': 0.9498, 'al': 0.9999707, 'svm': 0.3243}
stds = {'hap': 0.009913, 'al': 0.0002607, 'svm': 0.2574}
intercept = -4.200229
coefs = {'hap': -1.2332, 'al': -0.1580, 'svm': 0.1673}

df = pd.read_sql_query("""
    SELECT cj.*, t1.source_filename as source_name, t2.source_filename as target_name,
           t1.short_name_for_svm as source_svm_name, t2.short_name_for_svm as target_svm_name,
           t1.chapter_num as source_chapter, t2.chapter_num as target_chapter
    FROM combined_jaccard cj
    JOIN all_texts t1 ON cj.source_text = t1.text_id
    JOIN all_texts t2 ON cj.target_text = t2.text_id
""", main_conn)

chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
svm_conn.close()

svm_scores = []
for _, row in df.iterrows():
    source_novel = row['source_svm_name']
    target_novel = row['target_svm_name']
    target_chapter = str(row['target_chapter'])
    match = chapter_df[(chapter_df['novel'] == target_novel) & 
                       (chapter_df['number'] == target_chapter)]
    if len(match) > 0 and source_novel in match.columns:
        svm_scores.append(match[source_novel].iloc[0])
    else:
        svm_scores.append(np.nan)

df['svm_score'] = svm_scores
df = df.dropna(subset=['svm_score'])

df['hap_z'] = (df['hap_jac_dis'] - means['hap']) / stds['hap']
df['al_z'] = (df['al_jac_dis'] - means['al']) / stds['al']
df['svm_z'] = (df['svm_score'] - means['svm']) / stds['svm']
df['logit'] = intercept + coefs['hap'] * df['hap_z'] + coefs['al'] * df['al_z'] + coefs['svm'] * df['svm_z']
df['prob'] = 1 / (1 + np.exp(-df['logit']))
df = df.sort_values('prob', ascending=False).reset_index(drop=True)
df['rank'] = range(1, len(df) + 1)
N = len(df)

# ---- Extract author and work from source_name ----
def parse_source(name):
    # e.g. 1840-ENG18400--Shelley_Percy-section_7
    parts = name.split('--')
    if len(parts) < 2:
        return name, name, name
    year = name[:4]
    rest = parts[1]  # Shelley_Percy-section_7
    # split on first hyphen after author
    author_work = rest.split('-', 1)
    author = author_work[0].replace('_', ' ')
    return author, year, name

# ---- Filter to -> Lovelace only, exclude Lovelace self ----
to_lace = df[df['target_name'].str.contains('Lovelace')].copy()
to_lace_ext = to_lace[~to_lace['source_name'].str.contains('Lovelace')].copy()

to_lace_ext['author'] = to_lace_ext['source_name'].apply(lambda x: parse_source(x)[0])
to_lace_ext['year'] = to_lace_ext['source_name'].apply(lambda x: parse_source(x)[1])

# Identify distinct works (by year+author combo)
to_lace_ext['work_key'] = to_lace_ext['source_name'].apply(
    lambda x: '-'.join(x.split('--')[0:2]).rsplit('-', 1)[0] if '--' in x else x
)
# Simpler: use everything before the chapter/section/note/letter identifier
import re
def get_work_id(name):
    # strip chapter_X, section_X, Note_X, letter_X from end
    m = re.match(r'(.+?)[-_](?:chapter|section|Note|letter|part|book)_.*$', name)
    if m:
        return m.group(1)
    return name

to_lace_ext['work_id'] = to_lace_ext['source_name'].apply(get_work_id)

# Also get Lovelace target note
to_lace_ext['target_note'] = to_lace_ext['target_name'].apply(
    lambda x: x.split('Note_')[-1] if 'Note_' in x else 
              (x.split('letter_')[-1] if 'letter_' in x else x))

# ====================================================================
print('=' * 100)
print('EVERY AUTHOR -> LOVELACE: COMPREHENSIVE REPORT')
print(f'Total scored pairs in corpus: {N:,}')
print(f'Total pairs targeting Lovelace (excl. self): {len(to_lace_ext):,}')
print('=' * 100)

# ---- SECTION 1: Author-level summary ----
print('\n')
print('=' * 100)
print('SECTION 1: AUTHOR-LEVEL SUMMARY (sorted by best rank)')
print('=' * 100)
print(f'{"Author":<30} {"Pairs":>6} {"Best Rank":>12} {"Pctile":>8} {"Best P":>8} {"Mean P":>10} {"Med Rank":>10} {"Best SVM":>9}')
print('-' * 100)

author_stats = []
for author in sorted(to_lace_ext['author'].unique()):
    sub = to_lace_ext[to_lace_ext['author'] == author]
    best_row = sub.loc[sub['prob'].idxmax()]
    author_stats.append({
        'author': author,
        'pairs': len(sub),
        'best_rank': sub['rank'].min(),
        'pctile': (1 - sub['rank'].min() / N) * 100,
        'best_prob': sub['prob'].max(),
        'mean_prob': sub['prob'].mean(),
        'median_rank': sub['rank'].median(),
        'best_svm': sub['svm_score'].max(),
    })

author_stats.sort(key=lambda x: x['best_rank'])
for s in author_stats:
    print(f'{s["author"]:<30} {s["pairs"]:>6} {s["best_rank"]:>8}/{N}  {s["pctile"]:>6.1f}% {s["best_prob"]:>8.4f} {s["mean_prob"]:>10.6f} {s["median_rank"]:>8.0f}/{N} {s["best_svm"]:>8.4f}')

# ---- SECTION 2: Work-level summary ----
print('\n')
print('=' * 100)
print('SECTION 2: WORK-LEVEL SUMMARY (sorted by best rank)')
print('=' * 100)
print(f'{"Work":<65} {"Pairs":>5} {"Best Rank":>12} {"Pctile":>8} {"Best P":>8} {"Mean SVM":>9}')
print('-' * 100)

work_stats = []
for wid in sorted(to_lace_ext['work_id'].unique()):
    sub = to_lace_ext[to_lace_ext['work_id'] == wid]
    work_stats.append({
        'work_id': wid,
        'pairs': len(sub),
        'best_rank': sub['rank'].min(),
        'pctile': (1 - sub['rank'].min() / N) * 100,
        'best_prob': sub['prob'].max(),
        'mean_svm': sub['svm_score'].mean(),
    })

work_stats.sort(key=lambda x: x['best_rank'])
for s in work_stats:
    print(f'{s["work_id"]:<65} {s["pairs"]:>5} {s["best_rank"]:>8}/{N}  {s["pctile"]:>6.1f}% {s["best_prob"]:>8.4f} {s["mean_svm"]:>8.4f}')

# ---- SECTION 3: Per-author best pair to each Note ----
print('\n')
print('=' * 100)
print('SECTION 3: EACH AUTHOR\'S BEST CONNECTION TO EACH LOVELACE NOTE')
print('=' * 100)

notes = ['A', 'B', 'D', 'E', 'F', 'G']
# Also check if any letters appear as targets
letter_targets = to_lace_ext[to_lace_ext['target_name'].str.contains('letter')]
if len(letter_targets) > 0:
    print(f'\n  [NOTE: {len(letter_targets)} pairs target Lovelace letters, not just Notes]')

for author in [s['author'] for s in author_stats]:  # already sorted by best rank
    sub = to_lace_ext[to_lace_ext['author'] == author]
    print(f'\n  {author} ({len(sub)} pairs)')
    for note in notes:
        note_sub = sub[sub['target_name'].str.contains(f'Note_{note}$', regex=True)]
        if len(note_sub) > 0:
            best = note_sub.loc[note_sub['prob'].idxmax()]
            src_short = best['source_name'].split('--')[1] if '--' in best['source_name'] else best['source_name']
            print(f'    Note {note}: p={best["prob"]:.4f}  rank {best["rank"]:>6}/{N}  svm={best["svm_score"]:.4f}  hap={best["hap_jac_dis"]:.4f}  ({src_short})')
    # Check letters too
    let_sub = sub[sub['target_name'].str.contains('letter')]
    if len(let_sub) > 0:
        best = let_sub.loc[let_sub['prob'].idxmax()]
        src_short = best['source_name'].split('--')[1] if '--' in best['source_name'] else best['source_name']
        tgt_short = best['target_name'].split('--')[1] if '--' in best['target_name'] else best['target_name']
        print(f'    Letters: best p={best["prob"]:.4f}  rank {best["rank"]:>6}/{N}  ({src_short} -> {tgt_short})')

# ---- SECTION 4: Signal decomposition per author ----
print('\n')
print('=' * 100)
print('SECTION 4: SIGNAL DECOMPOSITION (mean values per author -> Lovelace)')
print('=' * 100)
print(f'{"Author":<30} {"Mean Hapax":>11} {"Mean Align":>11} {"Mean SVM":>9} {"Hapax Contrib":>14} {"SVM Contrib":>12}')
print('-' * 100)

for s in author_stats:
    sub = to_lace_ext[to_lace_ext['author'] == s['author']]
    mh = sub['hap_jac_dis'].mean()
    ma = sub['al_jac_dis'].mean()
    ms = sub['svm_score'].mean()
    # z-scored contributions
    hap_contrib = coefs['hap'] * (mh - means['hap']) / stds['hap']
    svm_contrib = coefs['svm'] * (ms - means['svm']) / stds['svm']
    print(f'{s["author"]:<30} {mh:>11.6f} {ma:>11.6f} {ms:>9.4f} {hap_contrib:>+14.4f} {svm_contrib:>+12.4f}')

# ---- SECTION 5: Where do Lovelace letters appear as targets? ----
print('\n')
print('=' * 100)
print('SECTION 5: LOVELACE LETTERS AS TARGETS')
print('=' * 100)
letter_tgts = to_lace_ext[to_lace_ext['target_name'].str.contains('letter')]
if len(letter_tgts) == 0:
    # Check in full to_lace (including self)
    letter_tgts_all = to_lace[to_lace['target_name'].str.contains('letter')]
    if len(letter_tgts_all) > 0:
        print(f'  {len(letter_tgts_all)} pairs target Lovelace letters (including self-comparisons)')
        for _, row in letter_tgts_all.sort_values('prob', ascending=False).head(20).iterrows():
            src = row['source_name'].split('--')[1] if '--' in row['source_name'] else row['source_name']
            tgt = row['target_name'].split('--')[1] if '--' in row['target_name'] else row['target_name']
            print(f'    rank {row["rank"]:>6}/{N}  p={row["prob"]:.4f}  {src} -> {tgt}')
    else:
        print('  No Lovelace letters appear as targets in any scored pairs.')
        # Check if letters are in the texts table at all
        letter_texts = pd.read_sql_query(
            "SELECT text_id, source_filename, short_name_for_svm FROM all_texts WHERE source_filename LIKE '%letter%'",
            main_conn)
        if len(letter_texts) > 0:
            print(f'  But {len(letter_texts)} letter files ARE in all_texts:')
            for _, lt in letter_texts.iterrows():
                print(f'    {lt["source_filename"]}  (svm_name: {lt["short_name_for_svm"]})')
        else:
            print('  No letter files found in all_texts table either.')
else:
    print(f'  {len(letter_tgts)} pairs target Lovelace letters from external authors')
    for _, row in letter_tgts.sort_values('prob', ascending=False).head(20).iterrows():
        src = row['source_name'].split('--')[1] if '--' in row['source_name'] else row['source_name']
        tgt = row['target_name'].split('--')[1] if '--' in row['target_name'] else row['target_name']
        print(f'    rank {row["rank"]:>6}/{N}  p={row["prob"]:.4f}  {src} -> {tgt}')

main_conn.close()
print('\n[DONE]')