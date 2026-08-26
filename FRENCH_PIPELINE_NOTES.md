# The French (ELTeC-fra) Run: Comparability Notes

This note records what the French pipeline run is and is not comparable to,
relative to the English thesis runs (ELTeC-100, Shelley-Lovelace), and the
staging decisions behind it. Companion to the language-profile refactor; the
per-language processing decisions themselves are documented in
[`language_profiles.py`](language_profiles.py).

## What the French run IS

A replication of the **method** on a new corpus: the same three signals
(hapax overlap, SVM stylometry, TextPAIR sequence alignment), the same
pair construction (temporal filter, 500-word minimum, same-author labels
derived from the corpus itself), the same logistic regression with freshly
fitted French coefficients, and the same validation logic applied to
documented French influence pairs.

Within-corpus quantities are fully meaningful and reportable:

- ROC AUC on the same-author task (French model, French pairs)
- percentile ranks of documented French influence cases
- coefficient signs and relative magnitudes for the French model
- the lemmatisation-on/off sensitivity delta (the `eltec-fra` vs
  `eltec-fra-lemma` projects)

## What it is NOT

**Not a coefficient-level comparison with the English model.** The English
and French runs differ by design in their preprocessing regimes, so absolute
coefficient values, hapax counts, and alignment counts are not commensurable
across languages:

- The English SVM path removed NLTK *English* stopwords and applied a
  shallow WordNet lemmatisation (no-POS noun default, post-stopword — in
  effect a plural strip). The French path removes NLTK *French* stopwords
  and does not lemmatise (reasoning in `language_profiles.py`). Both are the
  right treatment for their language, but they are different treatments.
- The English tokenisation quirks (elision gluing, ligature splitting) are
  deliberately preserved in the `en` profile for thesis reproducibility;
  the `fr` profile fixes them for French. Type counts (and therefore hapax
  inventories) are produced by different tokenisers.
- TextPAIR used the Porter stemmer for English and uses the French Snowball
  stemmer for French; these differ in aggressiveness. `modernize = yes` was
  a no-op for English but actively normalises older French orthography
  (kept enabled to mirror the English configuration line-for-line — the
  deviation is in effect, not in configuration).
- French morphology inflates type counts (verb conjugation, agreement), so
  raw hapax rates run higher in French for reasons that have nothing to do
  with style. Compare percentiles and ranks within a corpus, never raw
  counts across corpora.

**Not directly comparable in corpus composition either.** ELTeC-fra: 100
novels, 74 authors, 1840-1920, ~2,850 staged text units. ELTeC-100 (eng):
100 novels, 76 authors, 3,514 chapters. The French corpus includes
epistolary novels (169 letter units), children's literature, and
roman-feuilleton, and four novels are encoded as a single large chapter div
(both Balzac novels, Adam's *Païenne*, Barrès's *Colette Baudoche*), which
leaves those works as one text unit each rather than chapter-sized units.

## Honest-reporting checklist for any French numbers

1. Report French AUC/percentiles as a *replication on a second corpus*, not
   as a head-to-head with the English figures.
2. When quoting the validation table, note that Gaboriau → Leroux is
   documented at the author level but the corpus's Leroux novel is *La
   Reine du Sabbat*, not the famous Lecoq-referencing *Chambre jaune*; and
   that Dumas → Maquet is a collaboration control (known shared hand), not
   an influence case.
3. Note the four single-unit novels when discussing per-chapter statistics;
   two of them are Balzac, who appears in three validation pairs.
4. The English "receipts" reporting scripts (`show_receipts_sextant*.py`,
   `compare_influence_cases.py`) still carry a private English-only
   tokeniser and have not been migrated to profiles; do not quote their
   token/stopword statistics for French pairs until they are.

## Staging record

- Corpus: COST-ELTeC/ELTeC-fra, level-1 encoding, cloned at repo sibling
  `../ELTeC-fra`.
- `stage_eltec.py` converts level-1 TEI to the splits convention:
  outermost `chapter`/`letter` divs become units; `liminal`, `notes`, and
  `titlepage` divs are excluded; first-edition year from the TEI header
  forms the directory-name prefix that enforces temporal ordering; surnames
  in directory names come from the ELTeC filenames (ASCII, hyphens
  removed), keeping novel keys free of Unicode-normalisation hazards.
- TextPAIR input is a flat copy of the splits with `.xml` extensions in
  `text-pair/in-and-out/eltec_fra_held/`, aligned with
  `my_config_fra.ini` on the fork's `fra-alignment` branch (base:
  `mac-bare-metal`, the thesis branch).
