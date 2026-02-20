"""Centralised TEI handling for sextant.

Every file in the pipeline may or may not contain a TEI header.
These three functions are the ONLY place TEI is parsed or stripped.
"""
import re

_TEI_TAG_RE = re.compile(r"<[^>]+>")
_AUTHOR_RE  = re.compile(r"<author>([^,]+)", re.IGNORECASE | re.DOTALL)
_DATE_RE    = re.compile(r'when="([0-9]{4})')
_PARENS_RE  = re.compile(r"\s*\([\s\d-]*\)")


def strip_tei(text: str) -> str:
    """Remove TEI header and all XML tags, returning plain text."""
    text = text.split("</teiHeader>", 1)[-1] if "</teiHeader>" in text else text
    text = _TEI_TAG_RE.sub(" ", text)
    return text.strip()


def extract_author(xml_text: str) -> str:
    """Pull author surname from TEI <author> tag."""
    match = _AUTHOR_RE.search(xml_text)
    if match:
        author = match.group(1).strip()
        author = _PARENS_RE.sub("", author)
    else:
        author = "Unknown Author"
    return author.replace("-", "_")


def extract_date(xml_text: str) -> int:
    """Pull 4-digit year from TEI <date when="YYYY"> or <date>YYYY</date>."""
    when_match = _DATE_RE.search(xml_text)
    if when_match:
        return int(when_match.group(1))
    # Fallback: bare <date>YYYY</date>
    date_match = re.search(r"<date>(\d{4})</date>", xml_text)
    if date_match:
        return int(date_match.group(1))
    raise ValueError("No date found in TEI header")