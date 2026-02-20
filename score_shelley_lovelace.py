#!/usr/bin/env python3
import sqlite3
import numpy as np
import pandas as pd

project = 'shelley-lovelace'
main_conn = sqlite3.connect(f'./projects/{project}/db/{project}.db')
svm_conn = sqlite3.connect(f'./projects/{project}/db/svm.db')

# ELTeC scaler parameters
means = {'hap': 0.9498, 'al': 0.9999707, 'svm': 0.3243}
stds = {'hap': 0.009913, 'al': 0.0002607, 'svm': 0.2574}

# ELTeC logistic regression coefficients
intercept = -4.200229
coefs = {'hap': -1.2332, 'al': -0.1580, 'svm': 0.1673}

# Load combined_jaccard with text names
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

# Load SVM chapter assessments
chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
svm_conn.close()

# For each pair, get the SVM score: how much does the target chapter
# look like it was written by the source's author?
# The column name in chapter_assessments is the source's short_name_for_svm
svm_scores = []
for _, row in df.iterrows():
    source_novel = row['source_svm_name']
    target_novel = row['target_svm_name']
    target_chapter = str(row['target_chapter'])
    
    # Find the target chapter's SVM assessment
    match = chapter_df[(chapter_df['novel'] == target_novel) & 
                       (chapter_df['number'] == target_chapter)]
    
    if len(match) > 0 and source_novel in match.columns:
        svm_scores.append(match[source_novel].iloc[0])
    else:
        svm_scores.append(np.nan)

df['svm_score'] = svm_scores
print(f"Pairs with SVM scores: {df['svm_score'].notna().sum():,}")
df = df.dropna(subset=['svm_score'])

# Apply ELTeC weights
df['hap_z'] = (df['hap_jac_dis'] - means['hap']) / stds['hap']
df['al_z'] = (df['al_jac_dis'] - means['al']) / stds['al']
df['svm_z'] = (df['svm_score'] - means['svm']) / stds['svm']

df['logit'] = intercept + coefs['hap'] * df['hap_z'] + coefs['al'] * df['al_z'] + coefs['svm'] * df['svm_z']
df['prob'] = 1 / (1 + np.exp(-df['logit']))

df = df.sort_values('prob', ascending=False).reset_index(drop=True)
df['rank'] = range(1, len(df) + 1)

print(f"\n{'='*80}")
print(f"=== TOP 25 PAIRS BY INFLUENCE PROBABILITY ===")
print(f"{'='*80}\n")
for i, row in df.head(25).iterrows():
    print(f"{row['rank']:5d}. p={row['prob']:.4f}  {row['source_name']} -> {row['target_name']}")

# Shelley -> Lovelace
sl = df[df['source_name'].str.contains('Shelley') & df['target_name'].str.contains('Lovelace')]
print(f"\n{'='*80}")
print(f"=== SHELLEY -> LOVELACE PAIRS ({len(sl)} pairs) ===")
print(f"{'='*80}\n")
for _, row in sl.iterrows():
    print(f"  rank {row['rank']:5d}/{len(df)}  p={row['prob']:.4f}  hap={row['hap_jac_dis']:.6f}  al={row['al_jac_dis']:.6f}  svm={row['svm_score']:.4f}  {row['source_name']} -> {row['target_name']}")

if len(sl) > 0:
    best = sl['rank'].min()
    print(f"\n  Best rank: {best} of {len(df)} (top {best/len(df)*100:.1f}%)")
    print(f"  Percentile: {(1 - best/len(df))*100:.1f}th")

# Also show any -> Lovelace
al = df[df['target_name'].str.contains('Lovelace')]
print(f"\n{'='*80}")
print(f"=== ALL AUTHORS -> LOVELACE (top 20) ===")
print(f"{'='*80}\n")
for _, row in al.head(20).iterrows():
    print(f"  rank {row['rank']:5d}/{len(df)}  p={row['prob']:.4f}  {row['source_name']} -> {row['target_name']}")

main_conn.close()