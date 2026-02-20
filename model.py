#!/usr/bin/env python3
"""
Shared model loading and scoring pipeline for sextant.

Loads ELTeC-trained logistic regression parameters from CSV files
and applies them to any project's database. This is the ONLY place
model parameters are read or applied.
"""
import re
import sqlite3
import numpy as np
import pandas as pd


def load_eltec_model(eltec_path='./projects/eltec-100/results'):
    """Load trained model parameters from ELTeC result files.
    
    Returns dict with keys: means, stds, intercept, coefs
    """
    scaler = pd.read_csv(f'{eltec_path}/scaler_parameters.csv')
    coeffs = pd.read_csv(f'{eltec_path}/influence_coefficients_shap_cv.csv')
    
    with open(f'{eltec_path}/model_intercept.txt', 'r') as f:
        for line in f:
            if line.startswith('intercept'):
                intercept = float(line.split('=')[1].strip())
                break
    
    means = {
        'hap': scaler.loc[scaler['variable'] == 'hap_jac_dis', 'mean'].iloc[0],
        'al':  scaler.loc[scaler['variable'] == 'al_jac_dis',  'mean'].iloc[0],
        'svm': scaler.loc[scaler['variable'] == 'svm_score',   'mean'].iloc[0],
    }
    stds = {
        'hap': scaler.loc[scaler['variable'] == 'hap_jac_dis', 'std'].iloc[0],
        'al':  scaler.loc[scaler['variable'] == 'al_jac_dis',  'std'].iloc[0],
        'svm': scaler.loc[scaler['variable'] == 'svm_score',   'std'].iloc[0],
    }
    coefs = {
        'hap': coeffs.loc[coeffs['variable'] == 'hap_jac_dis', 'coefficient'].iloc[0],
        'al':  coeffs.loc[coeffs['variable'] == 'al_jac_dis',  'coefficient'].iloc[0],
        'svm': coeffs.loc[coeffs['variable'] == 'svm_score',   'coefficient'].iloc[0],
    }
    
    return {'means': means, 'stds': stds, 'intercept': intercept, 'coefs': coefs}


def load_and_score(project):
    """Load project data, join SVM scores, apply ELTeC model, return ranked DataFrame.
    
    Returns (df, N) where df has columns including prob, rank, source_name, target_name,
    and N is the total number of scored pairs.
    """
    model = load_eltec_model()
    means = model['means']
    stds = model['stds']
    intercept = model['intercept']
    coefs = model['coefs']
    
    main_conn = sqlite3.connect(f'./projects/{project}/db/{project}.db')
    svm_conn = sqlite3.connect(f'./projects/{project}/db/svm.db')
    
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
    print(f"Pairs with SVM scores: {df['svm_score'].notna().sum():,}")
    df = df.dropna(subset=['svm_score'])
    
    # Apply ELTeC model
    df['hap_z'] = (df['hap_jac_dis'] - means['hap']) / stds['hap']
    df['al_z'] = (df['al_jac_dis'] - means['al']) / stds['al']
    df['svm_z'] = (df['svm_score'] - means['svm']) / stds['svm']
    df['logit'] = intercept + coefs['hap'] * df['hap_z'] + coefs['al'] * df['al_z'] + coefs['svm'] * df['svm_z']
    df['prob'] = 1 / (1 + np.exp(-df['logit']))
    
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    N = len(df)
    
    # Stash model params on the dataframe for downstream decomposition
    df.attrs['model'] = model
    
    main_conn.close()
    return df, N


def parse_source(name):
    """Extract author name and year from a source filename.
    
    e.g. '1820-ENG18201--Shelley_Percy-section_7' -> ('Shelley Percy', '1820', <original>)
    """
    parts = name.split('--')
    if len(parts) < 2:
        return name, name, name
    year = name[:4]
    rest = parts[1]
    author_work = rest.split('-', 1)
    author = author_work[0].replace('_', ' ')
    return author, year, name


def get_work_id(name):
    """Strip chapter/section/Note/letter suffix to get work identifier."""
    m = re.match(r'(.+?)[-_](?:chapter|section|Note|letter|part|book)_.*$', name)
    if m:
        return m.group(1)
    return name