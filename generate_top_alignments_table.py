#!/usr/bin/env python3
"""
Generate a LaTeX table of top TextPAIR alignments among the highest-ranked
cross-author pairs in the Shelley-Lovelace corpus.

READ-ONLY against the existing scored output and database.  The only write
is the .tex file and stdout echo.

Run from sextant root:
    poetry run python generate_top_alignments_table.py
"""

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ======================================================================
# Config
# ======================================================================

PROJECT = 'shelley-lovelace'
THESIS = Path(__file__).resolve().parent.parent / 'thesis'
OUT_TEX = THESIS / 'dissertation/Document/sections/5_results/tables/top_alignments.tex'

# Same frozen ELTeC parameters as score_shelley_lovelace.py
INTERCEPT = -4.207439184487401
COEFS = {'hap': -1.2319672017708805, 'al': -0.15314075364182664, 'svm': 0.18399544222763672}
MEANS = {'hap': 0.945407929605967, 'al': 0.9999721042109713, 'svm': 0.32427006589173824}
STDS  = {'hap': 0.011339318040662662, 'al': 0.00025511513289408944, 'svm': 0.2573745232415633}

MAX_ROWS = 15
RANK_ROWS = 12
PASSAGE_MAXLEN = 90

# ======================================================================
# Helpers
# ======================================================================

def extract_author(filename):
    if filename is None:
        return None
    m = re.search(r'--([A-Za-z_]+?)(?:-|$)', filename)
    return m.group(1) if m else None


def latex_escape(s):
    for ch in ['&', '%', '$', '#', '_', '{', '}']:
        s = s.replace(ch, '\\' + ch)
    s = s.replace('~', '\\textasciitilde{}')
    s = s.replace('^', '\\textasciicircum{}')
    return s


def truncate_passage(s, maxlen=PASSAGE_MAXLEN):
    if len(s) <= maxlen:
        return s
    cut = s[:maxlen].rsplit(' ', 1)[0]
    return cut + r'\ldots'


def format_rank(rank):
    return f'{rank:,}'.replace(',', '{,}')


def human_name(filename):
    """Convert ELTeC-pattern filename to thesis display name.

    Examples:
        1835-ENG18350--Shelley_Mary-lodore-v2_chapter_18
          -> M.\ Shelley \emph{Lodore} v2 ch.\ 18 (1835)
        1843-ENG18430--Lovelace_Ada-Note_D
          -> Lovelace Note D (1843)
        1798-ENG17981--Coleridge_Samuel-mariner-part_6
          -> Coleridge \emph{Rime} part 6 (1798)
    """
    year = filename[:4]
    m = re.match(r'\d{4}-[A-Z0-9]+--(.*)', filename)
    if not m:
        return latex_escape(filename)
    rest = m.group(1)

    author_map = {
        'Shelley_Mary': "M.\\ Shelley",
        'Shelley_Percy': "P.\\ Shelley",
        'Lovelace_Ada': "Lovelace",
        'Babbage_Charles': "Babbage",
        'Somerville_Mary': "Somerville",
        'Whewell_William': "Whewell",
        'Chambers_Robert': "Chambers",
        'Coleridge_Samuel': "Coleridge",
        'Volney': "Volney",
        'Goldsmith_Oliver': "Goldsmith",
        'Godwin_William': "Godwin",
        'Byron_George': "Byron",
        'Locke_John': "Locke",
        'Menabrea_Lovelace': "Menabrea/Lovelace",
        'Darwin_Erasmus': "E.\\ Darwin",
        'Wollstonecraft_Mary': "Wollstonecraft",
        'Eliot_George': "Eliot",
        'Poe_Edgar': "Poe",
        'Milton_John': "Milton",
        'Hartley_David': "Hartley",
        'Ure_Andrew': "Ure",
        'Faraday_Michael': "Faraday",
        'Humboldt_Alexander': "Humboldt",
        'Lyell_Charles': "Lyell",
        'Stoker_Bram': "Stoker",
        'Polidori_John': "Polidori",
        'Keats_John': "Keats",
    }

    work_map = {
        'frankenstein': 'Frankenstein',
        'last_man': 'Last Man',
        'lodore': 'Lodore',
        'valperga': 'Valperga',
        'falkner': 'Falkner',
        'mathilda': 'Mathilda',
        'perkin_warbeck': 'Perkin Warbeck',
        'rambles': 'Rambles',
        'lives_france': 'Lives (France)',
        'lives_italy': 'Lives (Italy)',
        'notes_percy': 'Notes on Percy',
        'prometheus': 'Prometheus Unbound',
        'essays': 'Defence of Poetry',
        'economy': 'Economy of Manufactures',
        'decline': 'Decline of Science',
        'ninth_bridgewater': 'Ninth Bridgewater',
        'passages': 'Passages',
        'connexion': 'Connexion',
        'bridgewater_iii': 'Bridgewater III',
        'vestiges': 'Vestiges',
        'mariner': 'Rime',
        'ruins': 'Ruins',
        'vicar': 'Vicar of Wakefield',
        'caleb_williams': 'Caleb Williams',
        'middlemarch': 'Middlemarch',
        'essay_understanding': 'Essay on Understanding',
        'translation': 'Sketch of the Analytical Engine',
        'scientific_ideas': 'Philosophy of Inductive Sciences',
        'inductive_sciences': 'History of Inductive Sciences',
        'plurality': 'Plurality of Worlds',
        'preliminary': 'Preliminary Dissertation',
        'zoonomia': 'Zoonomia',
        'temple': 'Temple of Nature',
        'personal_narrative': 'Personal Narrative',
        'cosmos': 'Cosmos',
        'principles': 'Principles of Geology',
        'vindication': 'Vindication',
        'maria': 'Maria',
        'dracula': 'Dracula',
        'vampyre': 'Vampyre',
        'candle': 'Chemical History of a Candle',
        'paradise_lost': 'Paradise Lost',
        'manufactures': 'Philosophy of Manufactures',
        'observations': 'Observations on Man',
        'st_leon': 'St.\\ Leon',
    }

    # Extract author key
    author_key = rest.split('-', 1)[0] if '-' in rest else rest
    author_display = author_map.get(author_key, author_key.replace('_', ' '))

    after_author = rest[len(author_key)+1:] if '-' in rest else ''

    # Lovelace targets: special handling
    if author_key == 'Lovelace_Ada':
        if after_author.startswith('Note_'):
            note = after_author.replace('Note_', 'Note ')
            return f"Lovelace {note} ({year})"
        if after_author.startswith('letter_faraday_'):
            fid = after_author.replace('letter_faraday_', '')
            return f"Lovelace to Faraday, {fid} ({year})"
        if after_author.startswith('letter_ladybyron_'):
            parts = after_author.replace('letter_ladybyron_', '').split('_')
            box = parts[0] if parts else ''
            folio = parts[1] if len(parts) > 1 else ''
            return f"Lovelace to Lady Byron, {box} f.\\ {folio} ({year})"
        if after_author.startswith('letter_'):
            lid = after_author.replace('letter_', '')
            lid_escaped = lid.replace('_', '\\_')
            return f"Lovelace letter {lid_escaped} ({year})"
        return f"Lovelace ({year})"

    # General case: find work name and structural part
    # Pattern: workname-vN_chapter_M or workname-section_N etc.
    work_name = None
    struct_part = ''

    # Try to match work-structural pattern
    wm = re.match(r'([a-z_]+?)(?:-(v\d+))?[-_](?:chapter|section|part|book|letter|canto|Note|tei_chapter)_(.+)$', after_author)
    if wm:
        raw_work = wm.group(1)
        vol = wm.group(2) or ''
        num = wm.group(3)
        work_name = work_map.get(raw_work, raw_work.replace('_', ' ').title())

        if 'section' in after_author:
            struct_part = f"\\S {num}"
        elif 'chapter' in after_author or 'tei_chapter' in after_author:
            struct_part = f"ch.\\ {num}"
        elif 'part' in after_author:
            struct_part = f"part {num}"
        elif 'book' in after_author:
            struct_part = f"book {num}"
        elif 'letter' in after_author:
            struct_part = f"letter {num}"
        elif 'canto' in after_author:
            struct_part = f"canto {num}"
        elif 'Note' in after_author:
            struct_part = f"Note {num}"

        if vol:
            struct_part = f"{vol} {struct_part}"
    else:
        # No structural part found, just use after_author as work
        raw_work = after_author.rstrip('-_')
        work_name = work_map.get(raw_work, raw_work.replace('_', ' ').title())

    result = f"{author_display} \\emph{{{work_name}}}"
    if struct_part:
        result += f" {struct_part}"
    result += f" ({year})"
    return result


# ======================================================================
# Main
# ======================================================================

def main():
    db_path = f'./projects/{PROJECT}/db/{PROJECT}.db'
    svm_path = f'./projects/{PROJECT}/db/svm.db'

    main_conn = sqlite3.connect(db_path)
    svm_conn = sqlite3.connect(svm_path)

    # --- Load and score (same logic as score_shelley_lovelace.py) ---
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

    chapter_df = pd.read_sql_query("SELECT * FROM chapter_assessments", svm_conn)
    svm_conn.close()

    id_cols = ['novel', 'number']
    score_cols = [c for c in chapter_df.columns if c not in id_cols]
    chapter_long = chapter_df.melt(id_vars=id_cols, value_vars=score_cols,
                                   var_name='source_svm_name', value_name='svm_score')
    chapter_long['number'] = chapter_long['number'].astype(str)
    df['target_chapter'] = df['target_chapter'].astype(str)

    df = df.merge(chapter_long,
                  left_on=['target_svm_name', 'target_chapter', 'source_svm_name'],
                  right_on=['novel', 'number', 'source_svm_name'],
                  how='left').drop(columns=['novel', 'number'])
    df = df.dropna(subset=['svm_score'])

    df['hap_z'] = (df['hap_jac_dis'] - MEANS['hap']) / STDS['hap']
    df['al_z']  = (df['al_jac_dis']  - MEANS['al'])  / STDS['al']
    df['svm_z'] = (df['svm_score']   - MEANS['svm']) / STDS['svm']
    df['logit'] = INTERCEPT + COEFS['hap']*df['hap_z'] + COEFS['al']*df['al_z'] + COEFS['svm']*df['svm_z']
    df['prob'] = 1 / (1 + np.exp(-df['logit']))

    df['source_author'] = df['source_name'].apply(extract_author)
    df['target_author'] = df['target_name'].apply(extract_author)
    df['is_same_author'] = df['source_author'] == df['target_author']

    cross = df[~df['is_same_author']].copy()
    cross = cross.sort_values('prob', ascending=False).reset_index(drop=True)
    cross['rank'] = cross.index + 1

    n_cross = len(cross)
    print(f"Cross-author pairs: {n_cross:,}")

    # --- Load alignment passages ---
    align_df = pd.read_sql_query("""
        SELECT a.pair_id, a.source_passage, a.target_passage,
               src.source_filename as align_src_name,
               tgt.source_filename as align_tgt_name
        FROM alignments a
        JOIN all_texts src ON a.source_filename = src.text_id
        JOIN all_texts tgt ON a.target_filename = tgt.text_id
    """, main_conn)
    main_conn.close()

    # Join alignments onto cross-author ranked pairs
    merged = cross.merge(align_df, on='pair_id', how='inner')
    print(f"Cross-author alignments (rows): {len(merged):,}")
    print(f"Cross-author pairs with alignments: {merged['pair_id'].nunique():,}")

    # --- Verification: Shelley-to-Lovelace alignment count ---
    shelley_to_lovelace = merged[
        (merged['source_name'].str.contains('Shelley')) &
        (merged['target_name'].str.contains('Lovelace_Ada'))
    ]
    print(f"\nShelley-to-Lovelace alignments (cross-author, scored pairs): {len(shelley_to_lovelace)}")
    if len(shelley_to_lovelace) != 18:
        print(f"  WARNING: chapter text claims 18; found {len(shelley_to_lovelace)}.")
        print("  Rows found:")
        for _, r in shelley_to_lovelace.iterrows():
            print(f"    rank {r['rank']:,}: {r['source_name']} -> {r['target_name']}: \"{r['source_passage']}\"")

    # --- Selection logic ---
    # 1. Top 12 by parent-pair rank
    top_by_rank = merged.nsmallest(RANK_ROWS, 'rank')

    # 2. Force-include: Coleridge Rime -> Frankenstein
    coleridge = merged[
        merged['source_name'].str.contains('Coleridge') &
        merged['target_name'].str.contains('Frankenstein')
    ]

    # 3. Force-include: Babbage top 2 Babbage-to-Lovelace pairs
    babbage_to_lovelace = cross[
        (cross['source_author'] == 'Babbage_Charles') &
        (cross['target_name'].str.contains('Lovelace_Ada'))
    ].nsmallest(2, 'rank')
    babbage_forced = merged[merged['pair_id'].isin(babbage_to_lovelace['pair_id'])]

    # 4. Force-include: two specific Shelley-to-Lovelace alignments
    shelley_forced = merged[
        (
            merged['source_name'].str.contains('rambles-v1_letter_04') &
            merged['target_name'].str.contains('letter_166_95-98')
        ) | (
            merged['source_name'].str.contains('lodore-v2_chapter_06') &
            merged['target_name'].str.contains('letter_166_159-166')
        )
    ]

    # Combine, deduplicate, cap at 15
    forced = pd.concat([coleridge, babbage_forced, shelley_forced]).drop_duplicates(subset=['pair_id', 'source_passage'])
    forced_ids = set(zip(forced['pair_id'], forced['source_passage']))

    # Start with forced rows, then fill from top_by_rank
    result_rows = []
    seen = set()

    # Add forced rows first
    for _, row in forced.sort_values('rank').iterrows():
        key = (row['pair_id'], row['source_passage'])
        if key not in seen:
            result_rows.append(row)
            seen.add(key)

    # Fill from top_by_rank
    for _, row in top_by_rank.sort_values('rank').iterrows():
        key = (row['pair_id'], row['source_passage'])
        if key not in seen and len(result_rows) < MAX_ROWS:
            result_rows.append(row)
            seen.add(key)

    # Sort final by rank
    result_rows.sort(key=lambda r: r['rank'])

    # --- Verification: Coleridge rank ---
    coleridge_ranks = [r['rank'] for r in result_rows if 'Coleridge' in r['source_name']]
    print(f"\nColeridge Rime alignment parent rank(s): {coleridge_ranks}")
    print("  (top_cross_author.tex shows rank 4)")

    # --- Verification: Babbage ranks ---
    babbage_ranks = [r['rank'] for r in result_rows if 'Babbage' in r['source_name'] and 'Lovelace' in r['target_name']]
    print(f"Babbage-to-Lovelace alignment parent rank(s): {babbage_ranks}")
    print("  (babbage_lovelace.tex shows ranks 256 and 491)")

    # --- Identify Shelley-to-Lovelace forced rows for shading ---
    shelley_forced_ids = set(zip(shelley_forced['pair_id'], shelley_forced['source_passage']))

    # --- Generate LaTeX ---
    lines = []
    lines.append(r"% =====================================================================")
    lines.append(r"% Top TextPAIR alignments among highest-ranked cross-author pairs")
    lines.append(r"% v1.2-thesis (21 May 2026, corpus-complete run)")
    lines.append(r"% Source: shelley-lovelace.db alignments table + score_shelley_lovelace.py ranking")
    lines.append(f"% Cross-author denominator: {n_cross:,} pairs")
    lines.append(r"% Generated by: generate_top_alignments_table.py")
    lines.append(r"% =====================================================================")
    lines.append(r"")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabularx}{\textwidth}{r >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}X >{\raggedright\itshape\arraybackslash}X}")
    lines.append(r"\toprule")
    lines.append(r"Rank & Source & Target & Aligned passage \\")
    lines.append(r"\midrule")

    for row in result_rows:
        rank_str = format_rank(int(row['rank']))
        src_display = human_name(row['source_name'])
        tgt_display = human_name(row['target_name'])
        passage = truncate_passage(latex_escape(row['source_passage']), PASSAGE_MAXLEN)

        key = (row['pair_id'], row['source_passage'])
        shade = r"\rowcolor{gray!15}" + "\n" if key in shelley_forced_ids else ""
        lines.append(f"{shade}{rank_str} & {src_display} & {tgt_display} & {passage} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\caption[Top TextPAIR alignments]{TextPAIR sequence alignments among the top-ranked cross-author pairs. Alignments occur in only a small fraction of the " + format_rank(n_cross) + r" cross-author pairs and contribute 2\% of model weight; where they occur, they are the most directly interpretable signal, as the aligned text is itself readable. Passages are truncated for display; full alignments are reproducible via the pipeline (Appendix~B). Shaded rows are the two Shelley-to-Lovelace alignments discussed in the text.}")
    lines.append(r"\label{tab:top-alignments}")
    lines.append(r"\end{table}")

    tex = '\n'.join(lines)

    # Print to stdout for review
    print("\n" + "="*72)
    print("GENERATED LaTeX:")
    print("="*72)
    print(tex)

    # Write file
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex, encoding='utf-8')
    print(f"\nWritten to: {OUT_TEX}")


if __name__ == '__main__':
    main()
