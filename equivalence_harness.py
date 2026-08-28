#!/usr/bin/env python3
"""Equivalence harness: prove the language-profile refactor is a no-op for English.

Captures a golden record of every language-dependent transformation in the
pipeline, applied to every file in the named projects' splits/ directories:

  P1  extract_hapaxes.tokenize     (close-reading hapax tokeniser)
  P2  hapaxes_1tM.compute_hapaxes  (pipeline hapax path)
  P3  do_svm.preprocess_text       (SVM preprocessing path)
  NK  the NFKD novel-key normalisation from do_svm.prepare_chapter_data

Each file's output under P1-P3 is hashed (order-sensitive, so this asserts
byte-identity of the token streams, not just set equality). Novel keys are
stored verbatim. A small verbatim sample is kept per project for debugging.

Usage:
    python3 equivalence_harness.py capture   # run BEFORE refactoring
    python3 equivalence_harness.py verify    # run AFTER refactoring

`verify` recomputes everything with the current code and fails loudly on any
difference. Goldens live in equivalence/goldens-<project>.json.gz.
"""

import gzip
import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from multiprocessing import Pool, cpu_count

import extract_hapaxes
from hapaxes_1tM import compute_hapaxes
from do_svm import preprocess_text
from tei import strip_tei
from util import getListOfFiles

PROJECTS = ['eltec-100', 'shelley-lovelace']
GOLDEN_DIR = './equivalence'
SAMPLE_FILES_PER_PROJECT = 2


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def process_one_file(file_path):
    """Compute the three path outputs for a single split file."""
    with open(file_path, 'r') as f:
        raw = f.read()

    # P1: extract_hapaxes tokeniser, applied the way the tool applies it
    text_p1 = extract_hapaxes.load_text(file_path, strip_xml=True)
    tokens_p1 = extract_hapaxes.tokenize(text_p1)
    h1 = _sha('\n'.join(f'{tok}\t{off}' for tok, off in tokens_p1))

    # P2: pipeline hapax path, applied the way load_hapaxes.py applies it
    hapaxes = compute_hapaxes(strip_tei(raw))
    h2 = _sha('\n'.join(hapaxes))

    # P3: SVM preprocessing, applied the way do_svm.py applies it
    svm_text = preprocess_text(raw)
    h3 = _sha(svm_text)

    return (file_path, h1, h2, h3)


def sample_one_file(file_path):
    """Verbatim (truncated) outputs for one file, for debugging mismatches."""
    with open(file_path, 'r') as f:
        raw = f.read()
    text_p1 = extract_hapaxes.load_text(file_path, strip_xml=True)
    return {
        'p1_tokens': extract_hapaxes.tokenize(text_p1)[:50],
        'p2_hapaxes': compute_hapaxes(strip_tei(raw))[:50],
        'p3_text': preprocess_text(raw)[:600],
    }


def novel_keys_for_project(project):
    """Reproduce do_svm's dir-name -> NFKD novel-key mapping."""
    splits_dir = f'./projects/{project}/splits'
    keys = {}
    for entry in sorted(os.listdir(splits_dir)):
        if entry.startswith('.') or not os.path.isdir(os.path.join(splits_dir, entry)):
            continue
        title = entry.split('-')[1]
        keys[entry] = unicodedata.normalize('NFKD', title)
    return keys


def compute_project(project):
    files = sorted(getListOfFiles(f'./projects/{project}/splits'))
    if not files:
        sys.exit(f'No files found for project {project}')

    results = {}
    with Pool(processes=cpu_count()) as pool:
        for i, (path, h1, h2, h3) in enumerate(
            pool.imap(process_one_file, files, chunksize=16)
        ):
            results[path] = {'p1': h1, 'p2': h2, 'p3': h3}
            if (i + 1) % 500 == 0:
                print(f'  {project}: {i + 1}/{len(files)}', flush=True)

    samples = {path: sample_one_file(path) for path in files[:SAMPLE_FILES_PER_PROJECT]}

    return {
        'files': results,
        'samples': samples,
        'novel_keys': novel_keys_for_project(project),
    }


def golden_path(project):
    return os.path.join(GOLDEN_DIR, f'goldens-{project}.json.gz')


def capture(projects):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    git_head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True
    ).stdout.strip()
    for project in projects:
        print(f'Capturing {project}...', flush=True)
        data = compute_project(project)
        data['meta'] = {
            'git_head': git_head,
            'python': sys.version.split()[0],
            'file_count': len(data['files']),
        }
        with gzip.open(golden_path(project), 'wt', encoding='utf-8') as f:
            json.dump(data, f, sort_keys=True)
        print(f'  wrote {golden_path(project)} ({len(data["files"])} files)')


def verify(projects):
    all_ok = True
    for project in projects:
        with gzip.open(golden_path(project), 'rt', encoding='utf-8') as f:
            golden = json.load(f)
        print(f'Verifying {project} against {golden["meta"]["git_head"][:12]} '
              f'({golden["meta"]["file_count"]} files)...', flush=True)
        current = compute_project(project)

        mismatches = []
        golden_files = golden['files']
        current_files = current['files']
        for path in sorted(set(golden_files) | set(current_files)):
            if path not in golden_files:
                mismatches.append((path, 'file not in golden'))
            elif path not in current_files:
                mismatches.append((path, 'file missing from tree'))
            else:
                bad = [p for p in ('p1', 'p2', 'p3')
                       if golden_files[path][p] != current_files[path][p]]
                if bad:
                    mismatches.append((path, 'differs on ' + ', '.join(bad)))

        if golden['novel_keys'] != current['novel_keys']:
            mismatches.append(('<novel_keys>', 'novel-key mapping differs'))

        if mismatches:
            all_ok = False
            print(f'  FAIL: {len(mismatches)} mismatches')
            for path, why in mismatches[:10]:
                print(f'    {why}: {path}')
        else:
            print(f'  OK: {len(current["files"])} files byte-identical on all three paths')

    if not all_ok:
        sys.exit(1)
    print('\nEquivalence verified: English behaviour is unchanged.')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('capture', 'verify'):
        sys.exit(__doc__)
    projects = sys.argv[2:] or PROJECTS
    if sys.argv[1] == 'capture':
        capture(projects)
    else:
        verify(projects)
