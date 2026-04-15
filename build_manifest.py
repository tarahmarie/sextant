#!/usr/bin/env python3
"""Build a manifest CSV for corpus_review/ by combining the master audit CSV
with data derived from the actual files on disk."""

import csv
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent
CORPUS_REVIEW = REPO / "corpus_review"
# Path to the master audit CSV. Override with LOVELACE_AUDIT_CSV env var if
# stored elsewhere (e.g. outside the repo for privacy).
AUDIT_CSV = Path(os.environ.get("LOVELACE_AUDIT_CSV", REPO / "audit" / "master_audit.csv"))
OUTPUT = REPO / "corpus_review_manifest.csv"

# Load audit CSV into a dict keyed by filename (without .xml)
audit = {}
with open(AUDIT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fn = row["filename"].replace(".xml", "").strip()
        audit[fn] = row


def count_words_xml(path):
    """Strip XML tags and count words."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Remove XML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove XML declaration and processing instructions
        text = re.sub(r"<\?[^>]+\?>", " ", text)
        words = text.split()
        return len(words)
    except Exception:
        return 0


def extract_author_from_dirname(dirname):
    """Extract author from directory name like 1843-ENG18430--Lovelace_Ada-Note_A"""
    m = re.match(r"\d{4}-\w+--(.+?)(?:-|$)", dirname)
    if m:
        author_part = m.group(1)
        # Convert underscore to space
        return author_part.replace("_", " ")
    return ""


def extract_year_from_dirname(dirname):
    m = re.match(r"(\d{4})", dirname)
    return m.group(1) if m else ""


def extract_lang_from_dirname(dirname):
    m = re.match(r"\d{4}-([A-Z]{3})", dirname)
    return m.group(1) if m else ""


LISA_VALIDATED_MARCH_2026 = {
    "1826-ENG18260--Lovelace_Ada-letter_ladybyron_box41_folio_032",
    "1828-ENG18290--Lovelace_Ada-letter_ladybyron_box41_folio_063",
    "1828-ENG18291--Lovelace_Ada-letter_ladybyron_box41_folio_067",
    "1843-ENG18430--Lovelace_Ada-letter_ladybyron_box42_folio_081",
    "1843-ENG18430--Lovelace_Ada-letter_ladybyron_box42_folio_086",
    "1844-ENG18440--Lovelace_Ada-letter_ladybyron_box42_folio_142",
    "1844-ENG18440--Lovelace_Ada-letter_ladybyron_box42_folio_152",
    "1844-ENG18440--Lovelace_Ada-letter_ladybyron_box44_folio_210",
    "1851-ENG18510--Lovelace_Ada-letter_ladybyron_box44_folio_174",
    "1851-ENG18520--Lovelace_Ada-letter_ladybyron_box43_folio_208",
}


def classify_role(dirname, author, audit_row):
    """Determine the role of this item in the corpus."""
    # Override: 10 items validated by Lisa Gold March 27, 2026
    # (audit CSV predates the validation email)
    if dirname in LISA_VALIDATED_MARCH_2026:
        return "headline_lovelace_new"
    if audit_row:
        status = audit_row.get("transcription_status", "")
        pipeline_500 = audit_row.get("pipeline_ready_500w", "")
        is_ada = audit_row.get("is_ada", "")
        source_coll = audit_row.get("source_collection", "")
        if "babbage_add37192" in dirname:
            return "excluded_babbage_draft"
        if status == "validated" and is_ada == "yes":
            return "headline_lovelace"
        if status == "draft":
            return "unvalidated_draft"
    # Non-audit items: source/comparison authors
    author_lower = author.lower()
    if "lovelace" in author_lower:
        if "Note_" in dirname:
            return "headline_lovelace"
        if "Menabrea" in dirname:
            return "headline_menabrea_translation"
        return "lovelace_item"
    if "jameson" in author_lower:
        return "comparison_confound_test"
    return "source_author"


def determine_item_type(dirname):
    """What kind of text is this?"""
    if "Note_" in dirname:
        return "note"
    if "Menabrea" in dirname:
        return "menabrea_translation_section"
    if "letter" in dirname.lower():
        return "letter"
    if "chapter" in dirname or "section" in dirname:
        return "book_chapter"
    if "book_" in dirname:
        return "book_chapter"
    if "sonnet" in dirname:
        return "poem"
    return "text"


# Collect all entries in corpus_review (directories and bare XML files)
entries = sorted(os.listdir(CORPUS_REVIEW))

rows = []
for entry in entries:
    entrypath = CORPUS_REVIEW / entry
    dirname = entry

    if entrypath.is_dir():
        # Directory: find XML files inside
        xml_files = sorted([f for f in os.listdir(entrypath) if f.endswith(".xml")])
        if not xml_files:
            continue
        total_words = 0
        xml_count = len(xml_files)
        for xf in xml_files:
            total_words += count_words_xml(entrypath / xf)
    elif entry.endswith(".xml"):
        # Bare XML file at top level (e.g., flattened Lives sections)
        dirname = entry.replace(".xml", "")
        total_words = count_words_xml(entrypath)
        xml_count = 1
    else:
        continue

    year = extract_year_from_dirname(dirname)
    lang = extract_lang_from_dirname(dirname)
    author = extract_author_from_dirname(dirname)
    item_type = determine_item_type(dirname)

    # Try to match against audit CSV
    # The audit uses the base filename without chapter suffixes
    audit_row = audit.get(dirname, None)
    # Also try matching without section/chapter suffixes
    if not audit_row:
        base = re.sub(r"-section_\d+$|-chapter_\d+$|-book_\d+$", "", dirname)
        audit_row = audit.get(base, None)

    role = classify_role(dirname, author, audit_row)

    row = {
        "directory": dirname,
        "year": year,
        "language": lang,
        "author": author,
        "item_type": item_type,
        "role": role,
        "xml_file_count": xml_count,
        "word_count": total_words,
        "source_collection": audit_row.get("source_collection", "") if audit_row else "",
        "recipient": audit_row.get("recipient", "") if audit_row else "",
        "date_as_written": audit_row.get("date_as_written", "") if audit_row else "",
        "date_iso": audit_row.get("date_iso", "") if audit_row else "",
        "place": audit_row.get("place", "") if audit_row else "",
        "shelfmark": audit_row.get("shelfmark", "") if audit_row else "",
        "pipeline_ready_500w": audit_row.get("pipeline_ready_500w", "") if audit_row else "",
        "transcription_status": audit_row.get("transcription_status", "") if audit_row else "",
        "transcriber": audit_row.get("transcriber", "") if audit_row else "",
        "uncertainty_pct": audit_row.get("uncertainty_pct", "") if audit_row else "",
        "is_ada": audit_row.get("is_ada", "") if audit_row else "",
        "editorial_notes": audit_row.get("editorial_notes", "") if audit_row else "",
    }
    rows.append(row)

# Write output
fieldnames = [
    "directory", "year", "language", "author", "item_type", "role",
    "xml_file_count", "word_count",
    "source_collection", "recipient", "date_as_written", "date_iso",
    "place", "shelfmark", "pipeline_ready_500w", "transcription_status",
    "transcriber", "uncertainty_pct", "is_ada", "editorial_notes",
]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT}")

# Summary stats
roles = {}
for r in rows:
    role = r["role"]
    roles[role] = roles.get(role, 0) + 1
print("\nRole breakdown:")
for role, count in sorted(roles.items()):
    print(f"  {role}: {count}")
