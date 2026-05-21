#!/usr/bin/env python3
"""Extract TextPAIR alignment records for the 18 listed Mary Shelley -> Lovelace pairs.

Reads the existing TextPAIR alignment output (alignments.jsonl) and writes a
markdown file listing the verbatim source passage, target passage, and
surrounding context for each pair the post-extract_author()-fix v1.1-thesis
Sextant run (4 May 2026) flagged with non-trivial alignment signal (al < 1.000).

Read-only with respect to the pipeline. Produces one new file:
    mary_lovelace_alignment_passages.md  (gitignored research material)

Run from the sextant repo root:
    python extract_mary_lovelace_passages.py
or
    poetry run python extract_mary_lovelace_passages.py
"""

import json
import os

# Paths are relative to the sextant repo root (this script's location).
HERE = os.path.dirname(os.path.abspath(__file__))
ALIGNMENTS = os.path.join(HERE, "projects", "shelley-lovelace", "alignments", "alignments.jsonl")
OUT_MD = os.path.join(HERE, "mary_lovelace_alignment_passages.md")

# (source_filename_basename, target_filename_basename, al_score) tuples from
# the score_shelley_lovelace.py output, ordered by al ascending.
PAIRS = [
    ("1844-ENG18440--Shelley_Mary-rambles-v1_letter_04",     "1844-ENG18441--Lovelace_Ada-letter_166_95-98",        0.9950),
    ("1823-ENG18230--Shelley_Mary-valperga-v1_chapter_20",   "1840-ENG18400--Lovelace_Ada-letter_165_88-91",        0.9960),
    ("1831-ENG18310--Shelley_Mary-frankenstein-chapter_24",  "1844-ENG18441--Lovelace_Ada-letter_166_112-114",      0.9961),
    ("1844-ENG18440--Shelley_Mary-rambles-v1_letter_09",     "1844-ENG18441--Lovelace_Ada-letter_166_95-98",        0.9962),
    ("1831-ENG18310--Shelley_Mary-frankenstein-chapter_22",  "1840-ENG18400--Lovelace_Ada-letter_165_162-164",      0.9965),
    ("1835-ENG18350--Shelley_Mary-lodore-v2_chapter_06",     "1844-ENG18441--Lovelace_Ada-letter_166_159-166",      0.9968),
    ("1835-ENG18350--Shelley_Mary-lodore-v2_chapter_10",     "1843-ENG18430--Lovelace_Ada-Note_E",                  0.9978),
    ("1823-ENG18230--Shelley_Mary-valperga-v2_chapter_14",   "1835-ENG18351--Lovelace_Ada-letter_165_3-5",          0.9982),
    ("1826-ENG18260--Shelley_Mary-last_man-chapter_11",      "1844-ENG18441--Lovelace_Ada-letter_faraday_Faraday1637", 0.9982),
    ("1826-ENG18260--Shelley_Mary-last_man-chapter_19",      "1838-ENG18381--Lovelace_Ada-letter_165_105-108",      0.9984),
    ("1838-ENG18380--Shelley_Mary-lives_france-v1_section_03", "1843-ENG18430--Lovelace_Ada-Note_D",                0.9988),
    ("1835-ENG18352--Shelley_Mary-lives_italy-v2_section_26", "1843-ENG18430--Lovelace_Ada-Note_E",                 0.9991),
    ("1838-ENG18380--Shelley_Mary-lives_france-v1_section_17", "1843-ENG18430--Lovelace_Ada-Note_G",                0.9992),
    ("1835-ENG18352--Shelley_Mary-lives_italy-v2_section_10", "1844-ENG18441--Lovelace_Ada-letter_166_126-128",     0.9992),
    ("1835-ENG18352--Shelley_Mary-lives_italy-v2_section_24", "1842-ENG18420--Lovelace_Ada-letter_166_33-38",       0.9992),
    ("1838-ENG18380--Shelley_Mary-lives_france-v1_section_09", "1843-ENG18431--Menabrea_Lovelace-translation",       0.9992),
    ("1835-ENG18352--Shelley_Mary-lives_italy-v2_section_02", "1843-ENG18430--Lovelace_Ada-Note_A",                 0.9992),
    ("1838-ENG18380--Shelley_Mary-lives_france-v2_section_06", "1843-ENG18430--Lovelace_Ada-Note_A",                0.9995),
]


def basename(path):
    """Strip directory and .xml extension from a TextPAIR filename field."""
    if not path:
        return ""
    name = os.path.basename(path)
    if name.endswith(".xml"):
        name = name[:-4]
    return name


def main():
    # Index: keyed by sorted-basename-tuple -> list of (src, tgt, al) entries
    # that map to it. Sorted tuple lets us catch records regardless of which
    # side TextPAIR put the Mary Shelley file on.
    pair_keys = {}
    for src, tgt, al in PAIRS:
        key = tuple(sorted([src, tgt]))
        pair_keys.setdefault(key, []).append((src, tgt, al))

    found = {key: [] for key in pair_keys}

    n_total = 0
    n_match = 0
    with open(ALIGNMENTS, "r", encoding="utf-8") as f:
        for line in f:
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = basename(rec.get("source_filename"))
            t = basename(rec.get("target_filename"))
            key = tuple(sorted([s, t]))
            if key in pair_keys:
                found[key].append(rec)
                n_match += 1

    print(f"Scanned {n_total:,} records; matched {n_match} alignment record(s) "
          f"across {sum(1 for v in found.values() if v)} of {len(pair_keys)} unique pair-keys.")

    lines = []
    lines.append("# Mary Shelley → Ada Lovelace alignment passages")
    lines.append("")
    lines.append("Source: `projects/shelley-lovelace/alignments/alignments.jsonl` (TextPAIR output, v1.1-thesis run, 2026-05-03).")
    lines.append("")
    lines.append(f"Records scanned: {n_total:,}.  Pairs requested: {len(PAIRS)}.  "
                 f"Pairs with at least one alignment record: {sum(1 for v in found.values() if v)}.")
    lines.append("")
    lines.append("Pairs are ordered by the alignment-distance score reported by Sextant "
                 "(`al` column, ascending = strongest overlap first).")
    lines.append("Within a pair, multiple alignment records (if any) are listed in the order TextPAIR emitted them.")
    lines.append("")
    lines.append("---")
    lines.append("")

    missing = []
    for i, (src, tgt, al) in enumerate(PAIRS, start=1):
        key = tuple(sorted([src, tgt]))
        recs = found.get(key, [])
        src_short = src.split("--", 1)[1] if "--" in src else src
        tgt_short = tgt.split("--", 1)[1] if "--" in tgt else tgt

        lines.append(f"## Pair {i}: {src_short} → {tgt_short} (al={al:.4f})")
        lines.append("")
        lines.append(f"**Source:** `{src}`")
        lines.append(f"**Target:** `{tgt}`")

        if not recs:
            lines.append("**TextPAIR alignment count:** 0 — NO RECORD FOUND in alignments.jsonl.")
            lines.append("")
            lines.append("> Sextant reported `al < 1.000` for this pair, but TextPAIR has no aligned-passage record matching it.")
            lines.append("> This is a discrepancy between the Sextant alignment scoring and the TextPAIR raw output.")
            lines.append("")
            lines.append("---")
            lines.append("")
            missing.append((i, src_short, tgt_short, al))
            continue

        banalities = [r.get("banality") for r in recs]
        if all(b is True for b in banalities):
            banal_str = "yes (all alignments flagged banal)"
        elif all(b is False for b in banalities):
            banal_str = "no (none flagged banal)"
        else:
            banal_str = f"mixed ({sum(1 for b in banalities if b)}/{len(banalities)} flagged banal)"

        lines.append(f"**TextPAIR alignment count:** {len(recs)}")
        lines.append(f"**Banality flag:** {banal_str}")
        lines.append("")

        for j, rec in enumerate(recs, start=1):
            rec_src_base = basename(rec.get("source_filename"))
            rec_tgt_base = basename(rec.get("target_filename"))
            if rec_src_base == src:
                mary_side = ("source_passage", "source_context_before", "source_context_after")
                lov_side  = ("target_passage", "target_context_before", "target_context_after")
                direction = "TextPAIR direction: Mary → Lovelace"
            elif rec_tgt_base == src:
                mary_side = ("target_passage", "target_context_before", "target_context_after")
                lov_side  = ("source_passage", "source_context_before", "source_context_after")
                direction = "TextPAIR direction: Lovelace → Mary (record is reversed; passages re-labelled below)"
            else:
                mary_side = ("source_passage", "source_context_before", "source_context_after")
                lov_side  = ("target_passage", "target_context_before", "target_context_after")
                direction = "TextPAIR direction: unknown (defaulting to source=Mary)"

            m_pas = (rec.get(mary_side[0]) or "").strip()
            m_bef = (rec.get(mary_side[1]) or "").strip()
            m_aft = (rec.get(mary_side[2]) or "").strip()
            l_pas = (rec.get(lov_side[0]) or "").strip()
            l_bef = (rec.get(lov_side[1]) or "").strip()
            l_aft = (rec.get(lov_side[2]) or "").strip()

            banal = rec.get("banality")
            lines.append(f"### Alignment {j}")
            lines.append("")
            lines.append(f"- {direction}")
            lines.append(f"- Banality: `{banal}`")
            lines.append("")
            lines.append("**Mary Shelley passage:**")
            lines.append("")
            lines.append(f"> {m_pas}")
            lines.append("")
            if m_bef or m_aft:
                lines.append("Surrounding context (Mary Shelley):")
                lines.append("")
                lines.append(f"> ...{m_bef}  **[{m_pas}]**  {m_aft}...")
                lines.append("")
            lines.append("**Lovelace passage:**")
            lines.append("")
            lines.append(f"> {l_pas}")
            lines.append("")
            if l_bef or l_aft:
                lines.append("Surrounding context (Lovelace):")
                lines.append("")
                lines.append(f"> ...{l_bef}  **[{l_pas}]**  {l_aft}...")
                lines.append("")
            lines.append(f"**Shared passage (verbatim, as TextPAIR stored it):** `{m_pas}`")
            if m_pas != l_pas:
                lines.append("")
                lines.append(f"_Note: source and target passages differ exactly. Target form was: `{l_pas}`._")
            lines.append("")

        lines.append("---")
        lines.append("")

    if missing:
        lines.append("## Pairs with no TextPAIR record")
        lines.append("")
        lines.append("Sextant reported `al < 1.000` for these pairs, but no alignment record was found in `alignments.jsonl`:")
        lines.append("")
        for i, src_short, tgt_short, al in missing:
            lines.append(f"- Pair {i}: {src_short} → {tgt_short} (al={al:.4f})")
        lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {OUT_MD}")
    print(f"Pairs with records: {sum(1 for v in found.values() if v)} / {len(pair_keys)} "
          f"unique keys ({len(PAIRS)} pair entries)")
    print(f"Pairs missing: {len(missing)}")
    for i, src_short, tgt_short, al in missing:
        print(f"  - Pair {i}: {src_short} -> {tgt_short} (al={al:.4f})")


if __name__ == "__main__":
    main()
