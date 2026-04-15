#!/usr/bin/env python3
"""
extract_hapaxes.py -- Shared hapax legomena extraction for close reading.

Given two text files (TEI XML or plain text), finds hapax legomena in each,
computes the shared hapaxes, and displays keyword-in-context (KWIC) for each
shared hapax in both texts. Reports Jaccard distance for comparison with
Sextant pipeline output.

Usage:
    python3 extract_hapaxes.py FILE_A FILE_B [options]

    python3 extract_hapaxes.py --batch PAIRS_FILE [options]
        where PAIRS_FILE is a TSV with two file paths per line

Options:
    --context N       Words of context on each side (default: 8)
    --no-strip        Skip TEI/XML tag stripping (plain text input)
    --stopwords       Remove English stopwords before hapax computation
    --show-all        Also list non-shared hapaxes for each text
    --csv OUTPUT      Write shared hapaxes to CSV file
    --sort {alpha,freq,position}
                      Sort shared hapaxes (default: position in FILE_A)

Examples:
    python3 extract_hapaxes.py splits/Shelley_Percy/Defence_section_7.xml \\
                                splits/Lovelace_Ada/Note_D.xml

    python3 extract_hapaxes.py --batch pairs.tsv --csv hapax_report.csv
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter


def strip_tei(text):
    """Remove TEI/XML tags and header content."""
    # Remove everything up to and including </teiHeader> if present
    text = re.sub(r'.*?</teiHeader>', '', text, flags=re.DOTALL)
    # Remove all remaining XML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text):
    """Simple whitespace + punctuation tokenizer. Returns list of (token, position) tuples
    where position is the character offset in the original text."""
    tokens = []
    for match in re.finditer(r"[a-zA-Z\u00C0-\u00FF]+(?:'[a-zA-Z]+)?", text):
        tokens.append((match.group().lower(), match.start()))
    return tokens


ENGLISH_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'shall', 'should', 'may', 'might', 'can', 'could', 'not', 'no', 'nor',
    'so', 'if', 'than', 'that', 'this', 'these', 'those', 'it', 'its',
    'he', 'she', 'his', 'her', 'they', 'them', 'their', 'we', 'our', 'you',
    'your', 'i', 'me', 'my', 'as', 'up', 'out', 'about', 'into', 'through',
    'then', 'there', 'here', 'when', 'where', 'which', 'what', 'who', 'whom',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'own', 'same', 'very', 'just', 'because',
    'between', 'after', 'before', 'during', 'above', 'below', 'any', 'once',
    'again', 'further', 'also', 'too', 'much', 'many', 'well', 'upon',
}


def load_text(filepath, strip_xml=True):
    """Load a text file, optionally stripping TEI/XML."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    if strip_xml:
        text = strip_tei(text)
    return text


def find_hapaxes(tokens, use_stopwords=False):
    """Find hapax legomena (words appearing exactly once).
    Returns set of hapax types."""
    words = [t for t, _ in tokens]
    if use_stopwords:
        words = [w for w in words if w not in ENGLISH_STOPWORDS]
    counts = Counter(words)
    return {word for word, count in counts.items() if count == 1}


def kwic(tokens, target_word, context_size=8):
    """Keyword-in-context: find target_word in token list and return context strings."""
    words = [t for t, _ in tokens]
    results = []
    for i, word in enumerate(words):
        if word == target_word:
            left = words[max(0, i - context_size):i]
            right = words[i + 1:i + 1 + context_size]
            results.append({
                'left': ' '.join(left),
                'keyword': word,
                'right': ' '.join(right),
                'position': i,
            })
    return results


def jaccard_distance(set_a, set_b):
    """Jaccard distance between two sets. Returns 1.0 if both empty."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return 1.0 - (len(intersection) / len(union))


def format_kwic_line(hit, context_size=8):
    """Format a single KWIC hit for display."""
    left = hit['left'].rjust(context_size * 7)
    keyword = hit['keyword'].upper()
    right = hit['right']
    return f"  {left}  [{keyword}]  {right}"


def analyze_pair(file_a, file_b, args):
    """Analyze shared hapaxes between two files. Returns results dict."""
    text_a = load_text(file_a, strip_xml=not args.no_strip)
    text_b = load_text(file_b, strip_xml=not args.no_strip)

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)

    hapaxes_a = find_hapaxes(tokens_a, use_stopwords=args.stopwords)
    hapaxes_b = find_hapaxes(tokens_b, use_stopwords=args.stopwords)

    shared = hapaxes_a & hapaxes_b
    jac_dist = jaccard_distance(hapaxes_a, hapaxes_b)

    # Gather KWIC for shared hapaxes
    shared_details = []
    for word in shared:
        hits_a = kwic(tokens_a, word, args.context)
        hits_b = kwic(tokens_b, word, args.context)
        # Position of first occurrence in file A (for sorting)
        pos_a = hits_a[0]['position'] if hits_a else 0
        shared_details.append({
            'word': word,
            'kwic_a': hits_a,
            'kwic_b': hits_b,
            'position_a': pos_a,
        })

    # Sort
    if args.sort == 'alpha':
        shared_details.sort(key=lambda x: x['word'])
    elif args.sort == 'position':
        shared_details.sort(key=lambda x: x['position_a'])
    else:
        shared_details.sort(key=lambda x: x['word'])

    return {
        'file_a': file_a,
        'file_b': file_b,
        'tokens_a': len(tokens_a),
        'tokens_b': len(tokens_b),
        'types_a': len(set(t for t, _ in tokens_a)),
        'types_b': len(set(t for t, _ in tokens_b)),
        'hapaxes_a': hapaxes_a,
        'hapaxes_b': hapaxes_b,
        'shared': shared,
        'shared_details': shared_details,
        'jaccard_distance': jac_dist,
    }


def print_results(results, args):
    """Print analysis results to stdout."""
    name_a = os.path.basename(results['file_a'])
    name_b = os.path.basename(results['file_b'])

    print("=" * 78)
    print(f"  A: {results['file_a']}")
    print(f"  B: {results['file_b']}")
    print("=" * 78)
    print()

    print(f"  {'':30s} {'A':>10s} {'B':>10s}")
    print(f"  {'Tokens':30s} {results['tokens_a']:>10,d} {results['tokens_b']:>10,d}")
    print(f"  {'Types':30s} {results['types_a']:>10,d} {results['types_b']:>10,d}")
    print(f"  {'Hapax legomena':30s} {len(results['hapaxes_a']):>10,d} {len(results['hapaxes_b']):>10,d}")
    print()
    print(f"  Shared hapaxes: {len(results['shared'])}")
    print(f"  Jaccard distance: {results['jaccard_distance']:.6f}")
    print()

    if not results['shared']:
        print("  No shared hapax legomena found.")
        print()
        return

    print("-" * 78)
    print(f"  SHARED HAPAX LEGOMENA ({len(results['shared'])} words)")
    print("-" * 78)
    print()

    for detail in results['shared_details']:
        word = detail['word']
        print(f"  >> {word.upper()}")
        print()
        print(f"     in A ({name_a}):")
        for hit in detail['kwic_a']:
            print(format_kwic_line(hit, args.context))
        print()
        print(f"     in B ({name_b}):")
        for hit in detail['kwic_b']:
            print(format_kwic_line(hit, args.context))
        print()

    if args.show_all:
        only_a = results['hapaxes_a'] - results['shared']
        only_b = results['hapaxes_b'] - results['shared']

        print("-" * 78)
        print(f"  HAPAXES ONLY IN A ({len(only_a)} words)")
        print("-" * 78)
        for word in sorted(only_a):
            print(f"    {word}")

        print()
        print("-" * 78)
        print(f"  HAPAXES ONLY IN B ({len(only_b)} words)")
        print("-" * 78)
        for word in sorted(only_b):
            print(f"    {word}")
        print()


def write_csv(all_results, output_path):
    """Write shared hapaxes across all pairs to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'file_a', 'file_b', 'shared_hapax',
            'context_a', 'context_b',
            'jaccard_distance',
            'total_shared', 'hapaxes_a', 'hapaxes_b',
        ])
        for results in all_results:
            for detail in results['shared_details']:
                ctx_a = ' ... '.join(
                    f"{h['left']} [{h['keyword']}] {h['right']}"
                    for h in detail['kwic_a']
                )
                ctx_b = ' ... '.join(
                    f"{h['left']} [{h['keyword']}] {h['right']}"
                    for h in detail['kwic_b']
                )
                writer.writerow([
                    results['file_a'],
                    results['file_b'],
                    detail['word'],
                    ctx_a,
                    ctx_b,
                    f"{results['jaccard_distance']:.6f}",
                    len(results['shared']),
                    len(results['hapaxes_a']),
                    len(results['hapaxes_b']),
                ])
    print(f"\nCSV written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract shared hapax legomena between texts for close reading.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('files', nargs='*', help='Two text files to compare')
    parser.add_argument('--batch', help='TSV file with pairs of file paths (one pair per line)')
    parser.add_argument('--context', type=int, default=8, help='Words of context each side (default: 8)')
    parser.add_argument('--no-strip', action='store_true', help='Skip TEI/XML stripping')
    parser.add_argument('--stopwords', action='store_true', help='Remove stopwords before hapax computation')
    parser.add_argument('--show-all', action='store_true', help='List non-shared hapaxes too')
    parser.add_argument('--csv', help='Write results to CSV file')
    parser.add_argument('--sort', choices=['alpha', 'position'], default='position',
                        help='Sort order for shared hapaxes (default: position in first file)')

    args = parser.parse_args()

    # Determine pairs to process
    pairs = []
    if args.batch:
        with open(args.batch, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    parts = line.split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
    elif len(args.files) == 2:
        pairs.append((args.files[0], args.files[1]))
    else:
        parser.error('Provide exactly two files, or use --batch with a pairs file.')

    # Process pairs
    all_results = []
    for file_a, file_b in pairs:
        if not os.path.exists(file_a):
            print(f"Error: {file_a} not found", file=sys.stderr)
            continue
        if not os.path.exists(file_b):
            print(f"Error: {file_b} not found", file=sys.stderr)
            continue

        results = analyze_pair(file_a, file_b, args)
        all_results.append(results)
        print_results(results, args)

    # CSV output
    if args.csv and all_results:
        write_csv(all_results, args.csv)


if __name__ == '__main__':
    main()