#!/usr/bin/env python3
"""
Comprehensive sanity check for the splits/ directory.
Checks: empty files, non-XML, malformed XML, duplicates, wrong author attribution,
empty directories, wrong nesting level, naming convention violations, large files,
encoding issues.
"""

import os
import re
import sys
import hashlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

SPLITS_DIR = Path(__file__).parent / "splits"
MIN_WORD_COUNT = 50
MAX_FILE_SIZE_KB = 500

# Naming convention: {YEAR}-{LANG}{YEARVARIANT}--{Author_First}[-description]
# e.g. 1840-ENG18401--Whewell_William-scientific_ideas-chapter_37
NAME_PATTERN = re.compile(
    r'^(\d{4})-([A-Z]{3}\d{4,5})--([A-Za-z]+_[A-Za-z]+(?:_[A-Za-z]+)*)(-.*)?$'
)

# macOS copy artifacts
COPY_ARTIFACT_PATTERN = re.compile(r'( 2| \(\d+\)| copy)')

issues = defaultdict(list)  # category -> list of (path, detail)
stats = {
    "total_files": 0,
    "total_dirs": 0,
    "xml_files": 0,
    "clean_files": 0,
    "problematic_files": 0,
    "clean_dirs": 0,
    "problematic_dirs": 0,
}


def strip_xml_tags(text):
    """Remove XML tags and return plain text."""
    return re.sub(r'<[^>]+>', '', text)


def count_words(text):
    """Count words in plain text."""
    return len(text.split())


def check_encoding(filepath):
    """Check if file is valid UTF-8 and look for mojibake patterns."""
    mojibake_patterns = [
        b'\xc3\x83\xc2',      # double-encoded UTF-8
        b'\xef\xbf\xbd',      # replacement character
        b'\xc2\xa0' * 3,      # excessive non-breaking spaces
    ]
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        # Try UTF-8 decode
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            return "not-utf8"
        # Check for mojibake
        for pattern in mojibake_patterns:
            if pattern in raw:
                return "mojibake"
        # Check for null bytes
        if b'\x00' in raw:
            return "null-bytes"
        return None
    except Exception as e:
        return f"read-error: {e}"


def extract_author_from_header(xml_text):
    """Extract author name from TEI header."""
    # Use regex since namespaced parsing is fragile
    match = re.search(r'<author>([^<]+)</author>', xml_text)
    if match:
        return match.group(1).strip()
    return None


def extract_author_from_filename(name):
    """Extract author surname from filename like 1840-ENG18401--Whewell_William-..."""
    match = re.match(r'\d{4}-[A-Z]{3}\d{4,5}--([A-Za-z]+)_', name)
    if match:
        return match.group(1)
    return None


def normalize_author(author_str):
    """Normalize author string for comparison."""
    if not author_str:
        return ""
    # Remove dates in parentheses
    author_str = re.sub(r'\s*\([^)]*\)\s*', '', author_str)
    # Get surname (before comma or first word)
    parts = author_str.split(',')
    surname = parts[0].strip()
    return surname.lower()


def check_naming_convention(name, is_dir=False):
    """Check if name matches expected pattern. Returns list of issues."""
    problems = []

    # Skip .DS_Store and other hidden files
    if name.startswith('.'):
        return []

    # Strip .xml extension for file checks
    check_name = name
    if not is_dir and name.endswith('.xml'):
        check_name = name[:-4]

    # Check for spaces
    if ' ' in name:
        problems.append("contains spaces")

    # Check for macOS copy artifacts
    if COPY_ARTIFACT_PATTERN.search(name):
        problems.append("macOS copy artifact detected")

    # Check pattern match
    if not NAME_PATTERN.match(check_name):
        problems.append(f"does not match naming convention")
    else:
        # Additional checks on matched parts
        m = NAME_PATTERN.match(check_name)
        year = m.group(1)
        lang_code = m.group(2)
        # Year in lang code should start with the filename year
        if not lang_code[3:].startswith(year[:3]):
            problems.append(f"year mismatch in lang code: {year} vs {lang_code}")

    return problems


def main():
    print(f"Scanning: {SPLITS_DIR}")
    print("=" * 80)

    if not SPLITS_DIR.exists():
        print(f"ERROR: Directory does not exist: {SPLITS_DIR}")
        sys.exit(1)

    # Collect all files and dirs
    all_files = []
    all_dirs = []
    file_hashes = defaultdict(list)  # hash -> list of paths

    for root, dirs, files in os.walk(SPLITS_DIR):
        root_path = Path(root)
        rel_root = root_path.relative_to(SPLITS_DIR)

        for d in dirs:
            all_dirs.append(root_path / d)
        for f in files:
            all_files.append(root_path / f)

    stats["total_files"] = len(all_files)
    stats["total_dirs"] = len(all_dirs)

    print(f"Total files found: {stats['total_files']}")
    print(f"Total directories found: {stats['total_dirs']}")
    print()

    problematic_file_set = set()
    problematic_dir_set = set()

    # =========================================================================
    # CHECK FILES
    # =========================================================================

    for filepath in sorted(all_files):
        rel = filepath.relative_to(SPLITS_DIR)
        name = filepath.name
        file_issues = []

        # Skip .DS_Store for most checks but flag it
        if name == '.DS_Store':
            issues["non-xml-files"].append((str(rel), "macOS .DS_Store file"))
            problematic_file_set.add(str(rel))
            continue

        # --- Check 2: Non-XML files ---
        if not name.endswith('.xml'):
            issues["non-xml-files"].append((str(rel), f"extension: {filepath.suffix or '(none)'}"))
            problematic_file_set.add(str(rel))
            # Still continue other checks

        # --- Check 10: Encoding ---
        enc_issue = check_encoding(filepath)
        if enc_issue:
            issues["encoding-issues"].append((str(rel), enc_issue))
            file_issues.append(True)

        # Read file content
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            issues["read-errors"].append((str(rel), str(e)))
            problematic_file_set.add(str(rel))
            continue

        # --- Check 9: Suspiciously large files ---
        file_size_kb = filepath.stat().st_size / 1024
        if file_size_kb > MAX_FILE_SIZE_KB:
            issues["large-files"].append((str(rel), f"{file_size_kb:.1f} KB"))
            file_issues.append(True)

        # --- Check 4: Duplicate content (hash-based) ---
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        file_hashes[content_hash].append(str(rel))

        # --- Check 1: Empty or near-empty files ---
        plain_text = strip_xml_tags(content)
        word_count = count_words(plain_text)
        if word_count < MIN_WORD_COUNT:
            issues["empty-or-near-empty"].append((str(rel), f"{word_count} words"))
            file_issues.append(True)

        # --- Check 3: Malformed XML ---
        if name.endswith('.xml'):
            stats["xml_files"] += 1
            try:
                ET.fromstring(content)
            except ET.ParseError as e:
                issues["malformed-xml"].append((str(rel), str(e)[:120]))
                file_issues.append(True)

        # --- Check 5: Wrong author attribution ---
        if name.endswith('.xml'):
            header_author = extract_author_from_header(content)
            filename_author = extract_author_from_filename(name)
            if header_author and filename_author:
                norm_header = normalize_author(header_author)
                norm_file = filename_author.lower()
                if norm_header != norm_file:
                    issues["wrong-author"].append(
                        (str(rel), f"filename='{filename_author}' header='{header_author}'")
                    )
                    file_issues.append(True)

        # --- Check 7: Files at the wrong nesting level ---
        # Bare files at top level (should be in a subdirectory)
        depth = len(rel.parts)
        if depth == 1 and name.endswith('.xml'):
            issues["wrong-nesting"].append((str(rel), "bare XML file at top level (not in subdirectory)"))
            file_issues.append(True)

        # --- Check 8: Naming convention violations ---
        naming_problems = check_naming_convention(name)
        if naming_problems:
            for prob in naming_problems:
                issues["naming-violations"].append((str(rel), prob))
            file_issues.append(True)

        if file_issues:
            problematic_file_set.add(str(rel))

    # --- Check 4 (cont): Report duplicates ---
    for h, paths in file_hashes.items():
        if len(paths) > 1:
            for p in paths:
                issues["duplicates"].append((p, f"duplicate of {len(paths)} files sharing hash {h[:12]}"))
                problematic_file_set.add(p)

    # =========================================================================
    # CHECK DIRECTORIES
    # =========================================================================

    for dirpath in sorted(all_dirs):
        rel = dirpath.relative_to(SPLITS_DIR)
        name = dirpath.name

        # Skip hidden directories
        if name.startswith('.'):
            continue

        # --- Check 6: Empty directories ---
        contents = list(dirpath.iterdir())
        xml_files_in = [c for c in contents if c.is_file() and c.suffix == '.xml']
        all_files_in = [c for c in contents if c.is_file() and c.name != '.DS_Store']
        subdirs_in = [c for c in contents if c.is_dir()]

        if not all_files_in and not subdirs_in:
            issues["empty-dirs"].append((str(rel), "completely empty directory"))
            problematic_dir_set.add(str(rel))
        elif not all_files_in and subdirs_in:
            issues["empty-dirs"].append((str(rel), "contains only subdirectories, no content files"))
            problematic_dir_set.add(str(rel))

        # --- Check 8: Directory naming convention ---
        naming_problems = check_naming_convention(name, is_dir=True)
        if naming_problems:
            for prob in naming_problems:
                issues["naming-violations"].append((str(rel), f"(directory) {prob}"))
            problematic_dir_set.add(str(rel))

    # =========================================================================
    # REPORT
    # =========================================================================

    stats["problematic_files"] = len(problematic_file_set)
    stats["clean_files"] = stats["total_files"] - stats["problematic_files"]
    stats["problematic_dirs"] = len(problematic_dir_set)
    stats["clean_dirs"] = stats["total_dirs"] - stats["problematic_dirs"]

    print("=" * 80)
    print("SANITY CHECK RESULTS")
    print("=" * 80)

    print(f"\n--- SUMMARY ---")
    print(f"Total files checked:        {stats['total_files']}")
    print(f"Total directories checked:  {stats['total_dirs']}")
    print(f"XML files:                  {stats['xml_files']}")
    print(f"Clean files:                {stats['clean_files']}")
    print(f"Problematic files:          {stats['problematic_files']}")
    print(f"Clean directories:          {stats['clean_dirs']}")
    print(f"Problematic directories:    {stats['problematic_dirs']}")

    total_issues = sum(len(v) for v in issues.values())
    print(f"\nTotal issues found:         {total_issues}")

    # Print each category
    categories = [
        ("empty-or-near-empty", "1. EMPTY OR NEAR-EMPTY FILES (<50 words)"),
        ("non-xml-files", "2. NON-XML FILES"),
        ("malformed-xml", "3. MALFORMED XML"),
        ("duplicates", "4. DUPLICATE CONTENT"),
        ("wrong-author", "5. WRONG AUTHOR ATTRIBUTION"),
        ("empty-dirs", "6. EMPTY/CONTENT-FREE DIRECTORIES"),
        ("wrong-nesting", "7. FILES AT WRONG NESTING LEVEL"),
        ("naming-violations", "8. NAMING CONVENTION VIOLATIONS"),
        ("large-files", "9. SUSPICIOUSLY LARGE FILES (>500KB)"),
        ("encoding-issues", "10. ENCODING ISSUES"),
        ("read-errors", "READ ERRORS"),
    ]

    for key, title in categories:
        items = issues.get(key, [])
        print(f"\n{'=' * 80}")
        print(f"{title}: {len(items)} issue(s)")
        print(f"{'=' * 80}")
        if items:
            for path, detail in sorted(items):
                print(f"  {path}")
                print(f"    -> {detail}")
        else:
            print("  (none)")

    print(f"\n{'=' * 80}")
    print("END OF REPORT")
    print(f"{'=' * 80}")

    return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
