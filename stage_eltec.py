#!/usr/bin/env python3
"""Stage an ELTeC level-1 corpus into sextant's splits/ convention.

Reads ELTeC level-1 TEI files (e.g. FRA00501_Balzac.xml) and writes one file
per chapter/letter in the layout the pipeline expects:

    projects/<project>/splits/<year>-<ID>—<Surname>/<dirname>-chapter_<n>

matching the eltec-100 convention exactly: year prefix (temporal ordering is
enforced by filename sort), em-dash (U+2014) between corpus ID and surname,
extensionless chapter files containing a minimal TEI-Simple header (author
and date are re-extracted from it by tei.py downstream).

Unit selection: the OUTERMOST divs of type "chapter" or "letter" become text
units (a chapter that contains letters is one unit, so nothing is double
counted); "liminal", "notes", and "titlepage" divs are skipped as
non-authorial front/back matter; "group" divs are containers whose chapters
are picked up individually. The surname in the directory name comes from the
source filename (ASCII, hyphens removed), which keeps novel keys ASCII-only
and sidesteps filesystem Unicode-normalisation hazards.

Usage:
    python3 stage_eltec.py <eltec_level1_dir> <project_name>
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

TEI_NS = 'http://www.tei-c.org/ns/1.0'
NS = {'t': TEI_NS}

UNIT_TYPES = {'chapter', 'letter'}
SKIP_TYPES = {'liminal', 'notes', 'titlepage'}

SPLIT_HEADER = (
    '<?xml-model href="https://raw.githubusercontent.com/TEIC/TEI-Simple/master/teisimple.rng"'
    ' type="application/xml" schematypens="http://relaxng.org/ns/structure/1.0"?>'
    '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><titleStmt>'
    '<title>{title}</title><author>{author}</author></titleStmt>'
    '<publicationStmt><publisher>COST Action "Distant Reading for European'
    ' Literary History" (CA16204)</publisher><date>{year}</date>'
    '</publicationStmt></teiHeader><text><body>'
    '<div type="{dtype}" n="{n}">\n'
)
SPLIT_FOOTER = '\n</div></body></text></TEI>'


def outermost_units(body):
    """Yield (div_type, element) for outermost chapter/letter divs, in order."""
    def walk(elem, inside_unit):
        for child in elem:
            if child.tag == f'{{{TEI_NS}}}div':
                dtype = child.get('type', '')
                if dtype in SKIP_TYPES:
                    continue
                if dtype in UNIT_TYPES and not inside_unit:
                    yield (dtype, child)
                    # do not descend: nested letters stay inside this unit
                else:
                    yield from walk(child, inside_unit)
            else:
                yield from walk(child, inside_unit)
    yield from walk(body, False)


def div_text(div):
    """Plain text of a div, whitespace-normalised per line."""
    text = ' '.join(div.itertext())
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def first_edition_year(root):
    bibl = root.find('.//t:bibl[@type="firstEdition"]', NS)
    if bibl is None:
        raise ValueError('no firstEdition bibl')
    date = bibl.find('t:date', NS)
    if date is None:
        raise ValueError('no firstEdition date')
    m = re.search(r'\d{4}', (date.text or '') + (date.get('when') or ''))
    if not m:
        raise ValueError('no 4-digit year in firstEdition date')
    return m.group()


def stage_file(xml_path, splits_dir):
    root = ET.parse(xml_path).getroot()

    author_el = root.find('.//t:titleStmt/t:author', NS)
    title_el = root.find('.//t:titleStmt/t:title', NS)
    author = (author_el.text or '').strip()
    title = re.sub(r'\s*:\s*édition ELTeC\s*$', '', (title_el.text or '').strip())
    year = first_edition_year(root)

    stem = os.path.basename(xml_path).replace('.xml', '')      # FRA00501_Balzac
    corpus_id, _, surname = stem.partition('_')
    surname = surname.replace('-', '')                          # Viel-Castel -> VielCastel
    dirname = f'{year}-{corpus_id}—{surname}'

    out_dir = os.path.join(splits_dir, dirname)
    os.makedirs(out_dir, exist_ok=True)

    body = root.find('.//t:text/t:body', NS)
    counters = {}
    units = []
    for dtype, div in outermost_units(body):
        counters[dtype] = counters.get(dtype, 0) + 1
        n = counters[dtype]
        text = div_text(div)
        out_path = os.path.join(out_dir, f'{dirname}-{dtype}_{n}')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(SPLIT_HEADER.format(title=escape(title), author=escape(author),
                                        year=year, dtype=dtype, n=n))
            f.write(escape(text))
            f.write(SPLIT_FOOTER)
        units.append((dtype, n, len(text.split())))
    return dirname, author, units


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    level1_dir, project = sys.argv[1], sys.argv[2]
    splits_dir = f'./projects/{project}/splits'
    os.makedirs(splits_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(level1_dir, '*.xml')))
    if not files:
        sys.exit(f'No XML files in {level1_dir}')

    total_units = 0
    warnings = []
    for xml_path in files:
        dirname, author, units = stage_file(xml_path, splits_dir)
        total_units += len(units)
        small = [f'{t}_{n} ({w}w)' for t, n, w in units if w < 500]
        if len(units) == 0:
            warnings.append(f'{dirname}: NO UNITS FOUND')
        elif len(units) == 1:
            warnings.append(f'{dirname}: single unit of {units[0][2]:,} words')
        if small:
            warnings.append(f'{dirname}: {len(small)} unit(s) under 500 words: '
                            + ', '.join(small[:5]))
        print(f'{dirname}: {len(units)} units '
              f'({sum(w for _, _, w in units):,} words) [{author}]')

    print(f'\nStaged {total_units} units from {len(files)} novels '
          f'into {splits_dir}')
    if warnings:
        print(f'\n{len(warnings)} warnings:')
        for w in warnings:
            print(f'  - {w}')


if __name__ == '__main__':
    main()
