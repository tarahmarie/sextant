import sqlite3
import pandas as pd

PROJECT = 'shelley-lovelace'
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
print(f"After SQL join: {len(df):,}")

chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
print(f"chapter_assessments: {len(chapter_df):,}")

id_cols = ['novel', 'number']
score_cols = [c for c in chapter_df.columns if c not in id_cols]
chapter_long = chapter_df.melt(
    id_vars=id_cols, value_vars=score_cols,
    var_name='source_svm_name', value_name='svm_score'
)
print(f"chapter_long after melt: {len(chapter_long):,}")

dupe_mask = chapter_long.duplicated(subset=['novel', 'number', 'source_svm_name'], keep=False)
print(f"Duplicate right-side keys: {dupe_mask.sum():,}")

left_dupes = df.duplicated(subset=['target_svm_name', 'target_chapter', 'source_svm_name'], keep=False)
print(f"Duplicate left-side keys: {left_dupes.sum():,}")

df['target_chapter'] = df['target_chapter'].astype(str)
chapter_long['number'] = chapter_long['number'].astype(str)
merged = df.merge(
    chapter_long,
    left_on=['target_svm_name', 'target_chapter', 'source_svm_name'],
    right_on=['novel', 'number', 'source_svm_name'],
    how='left'
)
print(f"After merge: {len(merged):,}")
print(f"Inflation: {len(merged) - len(df):,} extra rows")
print(f"SVM not-null: {merged['svm_score'].notna().sum():,}")
print(f"SVM null: {merged['svm_score'].isna().sum():,}")
