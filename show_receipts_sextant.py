#!/usr/bin/env python3
"""
Show the Receipts: Eliot→Lawrence Anchor Case Analysis

Adapted for SEXTANT directory structure:
    projects/eltec-100/splits/{year}-{id}—{author}/{year}-{id}—{author}-chapter_{n}

Usage:
    python show_receipts_sextant.py --sextant-root /path/to/SEXTANT

Configure the anchor case in the CONFIG section below.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# =============================================================================
# CONFIG - EDIT THESE FOR YOUR ANCHOR CASE
# =============================================================================

# Source text (earlier publication - the influencer)
SOURCE_CONFIG = {
    'author': 'Eliot',
    'novel': 'Middlemarch',
    'year': '1872',
    'eltec_id': 'ENG18721',            # Middlemarch (87 chapters)
    'chapter': 79,
}

# Target text (later publication - the influenced)
TARGET_CONFIG = {
    'author': 'Lawrence',
    'novel': 'Women in Love',
    'year': '1920',
    'eltec_id': 'ENG19200',            # Women in Love (31 chapters)
    'chapter': 29,
}

# Paths relative to SEXTANT root
SPLITS_SUBDIR = 'projects/eltec-100/splits'
ALIGNMENTS_SUBDIR = 'projects/eltec-100/alignments'

# =============================================================================
# TEXT LOADING
# =============================================================================

def find_chapter_file(splits_dir: Path, year: str, eltec_id: str, author: str, chapter: int) -> Path:
    """
    Find a chapter file matching the naming pattern.
    
    Pattern: {year}-{eltec_id}—{author}-chapter_{n}
    
    Note: The dash between ID and author might be em-dash (—) or regular dash (-)
    """
    # Try different dash variants
    folder_patterns = [
        f"{year}-{eltec_id}—{author}",      # em-dash
        f"{year}-{eltec_id}-{author}",       # regular dash
        f"{year}-{eltec_id}_{author}",       # underscore
    ]
    
    chapter_patterns = [
        f"chapter_{chapter}",
        f"chapter-{chapter}",
        f"ch_{chapter}",
        f"ch{chapter}",
    ]
    
    # First, try to find the folder
    found_folder = None
    for pattern in folder_patterns:
        candidates = list(splits_dir.glob(f"*{pattern}*"))
        if candidates:
            found_folder = candidates[0]
            break
    
    # Also try searching by author name alone
    if not found_folder:
        candidates = list(splits_dir.glob(f"*{author}*"))
        # Filter by year if multiple matches
        year_matches = [c for c in candidates if year in c.name]
        if year_matches:
            found_folder = year_matches[0]
        elif candidates:
            print(f"Warning: Found {author} folders but none matching year {year}:")
            for c in candidates[:5]:
                print(f"  {c.name}")
            found_folder = candidates[0]
    
    if not found_folder:
        raise FileNotFoundError(
            f"Could not find folder for {author} ({year}) in {splits_dir}\n"
            f"Tried patterns: {folder_patterns}"
        )
    
    print(f"Found folder: {found_folder.name}")
    
    # Now find the chapter file
    found_chapter = None
    for pattern in chapter_patterns:
        candidates = list(found_folder.glob(f"*{pattern}*"))
        if candidates:
            found_chapter = candidates[0]
            break
    
    # Try with leading zeros
    if not found_chapter:
        for pattern in [f"chapter_{chapter:02d}", f"chapter_{chapter:03d}"]:
            candidates = list(found_folder.glob(f"*{pattern}*"))
            if candidates:
                found_chapter = candidates[0]
                break
    
    if not found_chapter:
        # List available chapters
        all_chapters = sorted(found_folder.glob("*chapter*"))
        print(f"Available chapters in {found_folder.name}:")
        for ch in all_chapters[:10]:
            print(f"  {ch.name}")
        if len(all_chapters) > 10:
            print(f"  ... and {len(all_chapters) - 10} more")
        raise FileNotFoundError(
            f"Could not find chapter {chapter} in {found_folder}"
        )
    
    return found_chapter


def load_chapter_text(chapter_path: Path) -> str:
    """Load text from a chapter file and strip XML/TEI markup."""
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            text = chapter_path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {chapter_path}")
    
    # Strip XML/TEI markup to match what the model sees
    text = strip_xml_markup(text)
    
    return text


def strip_xml_markup(text: str) -> str:
    """
    Remove XML/TEI markup from text, keeping only the literary content.
    This should match whatever stripping your Sextant pipeline does.
    """
    # Remove XML declaration and processing instructions
    text = re.sub(r'<\?xml[^>]*\?>', '', text)
    text = re.sub(r'<\?xml-model[^>]*\?>', '', text)
    
    # Remove entire TEI header (everything between <teiHeader> and </teiHeader>)
    text = re.sub(r'<teiHeader[^>]*>.*?</teiHeader>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove all XML tags but keep their content
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs that might be in the text
    text = re.sub(r'https?://\S+', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize(text: str, remove_stopwords: bool = False) -> list:
    """Simple tokenization."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    
    if remove_stopwords:
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'as', 'if', 'then', 'because', 'while', 'although', 'though', 'after',
            'before', 'when', 'where', 'there', 'here', 'from', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'once', 'about', 'out', 'up', 'down', 'off', 'over',
            'any', 'now', 'ever', 'also', 'back', 'well', 'much', 'even', 'still',
            'already', 'yet', 'never', 'always', 'often', 'sometimes', 'usually',
            'said', 'says', 'say', 'one', 'two', 'first', 'new', 'like', 'get',
            'make', 'made', 'go', 'going', 'went', 'come', 'came', 'take', 'took',
            'see', 'saw', 'know', 'knew', 'think', 'thought', 'want', 'give', 'gave',
            'tell', 'told', 'ask', 'asked', 'seem', 'seemed', 'let', 'put', 'keep',
            'kept', 'begin', 'began', 'look', 'looked', 'turn', 'turned', 'leave',
            'left', 'call', 'called', 'need', 'feel', 'felt', 'try', 'tried',
            'upon', 'little', 'long', 'great', 'good', 'old', 'right', 'man',
            'men', 'woman', 'women', 'way', 'day', 'time', 'year', 'hand', 'part',
            'place', 'case', 'week', 'nothing', 'something', 'anything', 'everything',
            'another', 'last', 'own', 'next', 'quite', 'being', 'mr', 'mrs', 'miss',
            'sir', 'yes', 'oh', 'yes', 'indeed', 'perhaps', 'however', 'without',
            'rather', 'enough', 'either', 'neither', 'whether', 'towards', 'against',
            'till', 'until', 'since', 'while', 'whose', 'whom', 'cannot', 'got',
            'thing', 'things', 'done', 'having', 'going', 'seeing', 'saying',
            'might', 'must', 'shall', 'should', 'could', 'would', 'herself',
            'himself', 'itself', 'themselves', 'myself', 'yourself', 'ourselves',
            'nobody', 'somebody', 'everybody', 'anyone', 'someone', 'everyone',
        }
        tokens = [t for t in tokens if t not in stop_words]
    
    return tokens


# =============================================================================
# HAPAX ANALYSIS
# =============================================================================

def find_hapaxes(tokens: list) -> set:
    """Find words appearing exactly once."""
    counts = Counter(tokens)
    return {word for word, count in counts.items() if count == 1}


def find_shared_hapaxes(tokens1: list, tokens2: list) -> set:
    """Find words that are hapax in BOTH texts."""
    return find_hapaxes(tokens1) & find_hapaxes(tokens2)


def get_word_context(text: str, word: str, window: int = 60) -> str:
    """Get the context around a word's first occurrence."""
    text_lower = text.lower()
    pattern = r'\b' + re.escape(word.lower()) + r'\b'
    match = re.search(pattern, text_lower)
    
    if not match:
        return ""
    
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    
    context = text[start:end].strip()
    
    # Clean up to word boundaries
    if start > 0:
        context = '...' + context.split(None, 1)[-1] if ' ' in context else '...' + context
    if end < len(text):
        context = context.rsplit(None, 1)[0] + '...' if ' ' in context else context + '...'
    
    return context


# =============================================================================
# N-GRAM ANALYSIS
# =============================================================================

def find_shared_ngrams(tokens1: list, tokens2: list, n: int = 3) -> list:
    """Find shared n-grams between texts."""
    def get_ngrams(tokens):
        return {' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}
    
    shared = get_ngrams(tokens1) & get_ngrams(tokens2)
    
    # Filter out very common patterns
    common = {'in the', 'of the', 'to the', 'and the', 'it was', 'he was',
              'she was', 'there was', 'that was', 'i was', 'it is', 'he had',
              'she had', 'they had', 'would be', 'could be', 'had been',
              'did not', 'do not', 'was not', 'could not', 'would not',
              'i have', 'i am', 'i do', 'as if', 'at the', 'on the',
              'for the', 'with the', 'from the', 'by the', 'that the',
              'which was', 'who was', 'what was', 'all the', 'but the',
              'into the', 'out of', 'one of', 'some of', 'more than',
              'such a', 'if you', 'if i', 'but i', 'and i', 'that i',
              'i could', 'i would', 'i should', 'i might', 'i must'}
    
    filtered = [ng for ng in sorted(shared) if not any(c in ng for c in common)]
    return filtered


# =============================================================================
# ALIGNMENT LOADING
# =============================================================================

def load_alignments(alignments_dir: Path, source_id: str, target_id: str) -> list:
    """Load TextPAIR alignments between two texts."""
    alignments = []
    
    if not alignments_dir.exists():
        return alignments
    
    # Try different file patterns
    for json_file in alignments_dir.glob('*.json'):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for align in data:
                    src = str(align.get('source', ''))
                    tgt = str(align.get('target', ''))
                    if source_id in src and target_id in tgt:
                        alignments.append(align)
        except (json.JSONDecodeError, Exception):
            continue
    
    return alignments


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(
    source_config: dict,
    target_config: dict,
    source_text: str,
    target_text: str,
    alignments: list = None
) -> str:
    """Generate the full receipts report."""
    
    lines = []
    def add(line=""):
        lines.append(line)
    
    add("=" * 80)
    add("SEXTANT ANCHOR CASE: SHOWING THE RECEIPTS")
    add("=" * 80)
    add()
    add(f"Source: {source_config['author']}, {source_config['novel']}, "
        f"Chapter {source_config['chapter']} ({source_config['year']})")
    add(f"Target: {target_config['author']}, {target_config['novel']}, "
        f"Chapter {target_config['chapter']} ({target_config['year']})")
    add()
    
    # Tokenize
    src_tokens = tokenize(source_text)
    tgt_tokens = tokenize(target_text)
    src_tokens_filtered = tokenize(source_text, remove_stopwords=True)
    tgt_tokens_filtered = tokenize(target_text, remove_stopwords=True)
    
    # Basic stats
    add("-" * 80)
    add("1. BASIC STATISTICS")
    add("-" * 80)
    add(f"Source word count: {len(src_tokens):,}")
    add(f"Target word count: {len(tgt_tokens):,}")
    add(f"Source unique words (excl. stopwords): {len(set(src_tokens_filtered)):,}")
    add(f"Target unique words (excl. stopwords): {len(set(tgt_tokens_filtered)):,}")
    add()
    
    # Hapax analysis
    add("-" * 80)
    add("2. SHARED HAPAX LEGOMENA")
    add("-" * 80)
    add()
    add("These are words appearing exactly ONCE in each chapter—rare vocabulary")
    add("choices that both authors made, suggesting potential vocabulary absorption.")
    add()
    
    src_hapaxes = find_hapaxes(src_tokens_filtered)
    tgt_hapaxes = find_hapaxes(tgt_tokens_filtered)
    shared_hapaxes = find_shared_hapaxes(src_tokens_filtered, tgt_tokens_filtered)
    
    add(f"Hapaxes in source chapter: {len(src_hapaxes)}")
    add(f"Hapaxes in target chapter: {len(tgt_hapaxes)}")
    add(f"SHARED HAPAXES: {len(shared_hapaxes)}")
    add()
    
    if shared_hapaxes:
        # Sort by word length (longer words often more meaningful)
        sorted_hapaxes = sorted(shared_hapaxes, key=lambda w: (-len(w), w))
        
        add("Shared hapax legomena (sorted by length, longer = more distinctive):")
        add()
        for word in sorted_hapaxes:
            add(f"  • {word}")
        add()
        
        # Show contexts for top 15 most interesting (longest) hapaxes
        add("Contexts for most distinctive shared hapaxes:")
        add()
        
        for word in sorted_hapaxes[:15]:
            src_ctx = get_word_context(source_text, word)
            tgt_ctx = get_word_context(target_text, word)
            
            add(f"  '{word}':")
            if src_ctx:
                add(f"    [{source_config['author']}]: \"{src_ctx}\"")
            if tgt_ctx:
                add(f"    [{target_config['author']}]: \"{tgt_ctx}\"")
            add()
    
    # N-gram analysis
    add("-" * 80)
    add("3. SHARED PHRASES (N-GRAM ANALYSIS)")
    add("-" * 80)
    add()
    add("Shared multi-word sequences suggest phrasal echo or structural similarity.")
    add()
    
    trigrams = find_shared_ngrams(src_tokens, tgt_tokens, n=3)
    bigrams = find_shared_ngrams(src_tokens, tgt_tokens, n=2)
    
    add(f"Shared trigrams (3-word sequences): {len(trigrams)}")
    add(f"Shared bigrams (2-word sequences, filtered): {len(bigrams)}")
    add()
    
    if trigrams:
        add("Notable shared trigrams:")
        for ng in trigrams[:25]:
            add(f"  • \"{ng}\"")
        if len(trigrams) > 25:
            add(f"  ... and {len(trigrams) - 25} more")
        add()
    
    # TextPAIR alignments
    add("-" * 80)
    add("4. SEQUENCE ALIGNMENTS (TextPAIR)")
    add("-" * 80)
    add()
    
    if alignments:
        add(f"Found {len(alignments)} sequence alignment(s) between these chapters:")
        add()
        for i, align in enumerate(alignments[:10], 1):
            add(f"  Alignment {i}:")
            if 'source_passage' in align:
                passage = align['source_passage'][:300]
                add(f"    Source: \"{passage}{'...' if len(align['source_passage']) > 300 else ''}\"")
            if 'target_passage' in align:
                passage = align['target_passage'][:300]
                add(f"    Target: \"{passage}{'...' if len(align['target_passage']) > 300 else ''}\"")
            if 'score' in align:
                add(f"    Score: {align['score']}")
            add()
    else:
        add("No TextPAIR alignments found for this specific chapter pair.")
        add("(This is expected—sequence alignments are rare in fiction.)")
        add()
    
    # Thematic analysis
    add("-" * 80)
    add("5. THEMATIC VOCABULARY")
    add("-" * 80)
    add()
    add("Shared vocabulary in semantically meaningful categories:")
    add()
    
    # Domain-specific word lists
    psychological = {
        'consciousness', 'soul', 'spirit', 'mind', 'feeling', 'emotion',
        'thought', 'desire', 'passion', 'fear', 'hope', 'despair', 'joy',
        'sorrow', 'grief', 'pain', 'suffering', 'will', 'intention', 'purpose',
        'meaning', 'truth', 'reality', 'illusion', 'self', 'identity', 'being',
        'existence', 'life', 'death', 'fate', 'freedom', 'struggle', 'conflict',
        'revelation', 'discovery', 'awakening', 'transformation', 'crisis',
        'anguish', 'agony', 'ecstasy', 'rapture', 'yearning', 'longing'
    }
    
    relationship = {
        'marriage', 'husband', 'wife', 'wedding', 'married', 'love', 'lover',
        'passion', 'desire', 'devotion', 'fidelity', 'betrayal', 'union',
        'bond', 'intimacy', 'tenderness', 'affection', 'attachment', 'embrace'
    }
    
    moral = {
        'duty', 'obligation', 'conscience', 'guilt', 'shame', 'honor', 'virtue',
        'sin', 'wrong', 'right', 'good', 'evil', 'moral', 'ethical', 'judgment',
        'responsibility', 'sacrifice', 'renunciation', 'redemption', 'forgiveness'
    }
    
    src_vocab = set(src_tokens)
    tgt_vocab = set(tgt_tokens)
    shared_vocab = src_vocab & tgt_vocab
    
    shared_psych = psychological & shared_vocab
    shared_rel = relationship & shared_vocab
    shared_moral = moral & shared_vocab
    
    add(f"Psychological/emotional: {', '.join(sorted(shared_psych)) if shared_psych else '(none)'}")
    add(f"Relationship/marriage: {', '.join(sorted(shared_rel)) if shared_rel else '(none)'}")
    add(f"Moral/ethical: {', '.join(sorted(shared_moral)) if shared_moral else '(none)'}")
    add()
    
    # Summary
    add("=" * 80)
    add("SUMMARY: THE RECEIPTS")
    add("=" * 80)
    add()
    add(f"When Sextant flags {source_config['novel']} ch. {source_config['chapter']} → "
        f"{target_config['novel']} ch. {target_config['chapter']}")
    add(f"as a high-scoring pair (99.97th percentile), here is what it sees:")
    add()
    add(f"  • {len(shared_hapaxes)} shared hapax legomena (rare vocabulary)")
    add(f"  • {len(trigrams)} shared trigrams (phrasal patterns)")
    add(f"  • {len(shared_psych)} shared psychological terms")
    add(f"  • {len(shared_rel)} shared relationship terms")
    add(f"  • {len(alignments) if alignments else 0} sequence alignments")
    add()
    add("These are the interpretable features that a scholar can examine to")
    add("determine whether the detected similarity reflects genuine literary influence.")
    add()
    add("=" * 80)
    
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Show Sextant anchor case receipts")
    parser.add_argument('--sextant-root', type=Path, required=True,
                        help='Path to SEXTANT root directory')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output file path (default: print to stdout)')
    parser.add_argument('--list-folders', action='store_true',
                        help='List available author folders and exit')
    
    args = parser.parse_args()
    
    splits_dir = args.sextant_root / SPLITS_SUBDIR
    alignments_dir = args.sextant_root / ALIGNMENTS_SUBDIR
    
    if not splits_dir.exists():
        print(f"Error: Splits directory not found: {splits_dir}")
        print(f"Check that SEXTANT root is correct: {args.sextant_root}")
        return 1
    
    if args.list_folders:
        print("Available author folders:")
        for folder in sorted(splits_dir.iterdir()):
            if folder.is_dir():
                n_chapters = len(list(folder.glob("*chapter*")))
                print(f"  {folder.name} ({n_chapters} chapters)")
        return 0
    
    # Find and load source chapter
    print(f"Looking for source: {SOURCE_CONFIG['author']} ch. {SOURCE_CONFIG['chapter']}...")
    try:
        src_path = find_chapter_file(
            splits_dir,
            SOURCE_CONFIG['year'],
            SOURCE_CONFIG['eltec_id'],
            SOURCE_CONFIG['author'],
            SOURCE_CONFIG['chapter']
        )
        source_text = load_chapter_text(src_path)
        print(f"  Loaded: {src_path.name} ({len(source_text):,} chars)")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Find and load target chapter
    print(f"Looking for target: {TARGET_CONFIG['author']} ch. {TARGET_CONFIG['chapter']}...")
    try:
        tgt_path = find_chapter_file(
            splits_dir,
            TARGET_CONFIG['year'],
            TARGET_CONFIG['eltec_id'],
            TARGET_CONFIG['author'],
            TARGET_CONFIG['chapter']
        )
        target_text = load_chapter_text(tgt_path)
        print(f"  Loaded: {tgt_path.name} ({len(target_text):,} chars)")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Load alignments
    print(f"Looking for alignments in {alignments_dir}...")
    alignments = load_alignments(
        alignments_dir,
        SOURCE_CONFIG['eltec_id'],
        TARGET_CONFIG['eltec_id']
    )
    print(f"  Found {len(alignments)} relevant alignments")
    
    # Generate report
    print()
    report = generate_report(
        SOURCE_CONFIG,
        TARGET_CONFIG,
        source_text,
        target_text,
        alignments
    )
    
    if args.output:
        args.output.write_text(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)
    
    return 0


if __name__ == '__main__':
    exit(main())