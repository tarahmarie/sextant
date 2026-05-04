# Thesis Reproduction: Sextant v1.0-thesis

This document records the exact pipeline run that produced the results cited in Tarah Wheeler's DPhil thesis in 2026 at the University of Oxford. It is intended for the viva committee, post-submission readers, and anyone attempting to reproduce the thesis-specific findings.

For general usage and installation instructions, see [README.md](README.md).

## Summary

The v1.0-thesis Sextant run was executed on 29 April 2026 against two corpora:

1. **ELTeC-100** (validation, model training): 5,956,426 chapter pairs, 76 authors, ROC AUC 0.78, all eight documented Victorian influence cases above the 99th percentile.
2. **Shelley-Lovelace** (thesis target): 577,145 cross-author pairs, 54 authors, frozen ELTeC weights applied via `score_shelley_lovelace.py`.

Total compute cost: 16.2 minutes wall-clock, 0.434 g CO₂eq combined emissions.

The model was trained once on ELTeC and applied without modification to the Shelley-Lovelace corpus. This separation between training and application corpora is methodologically important: the influence-detection model is the same model anyone could verify against the eight known Victorian cases, then the same model produces the Lovelace-targeted percentile rankings.

---

## Three-Repository Reproducibility Chain

Three companion repositories at their `v1.0-thesis` tagged versions:

| Repository | Branch | Tag | Role |
|---|---|---|---|
| `tarahmarie/gutenberg-text-splitter` | `batch-tweaks` | `v1.0-thesis` | Gutenberg ingestion and ELTeC-pattern splitting |
| `tarahmarie/text-pair` | `mac-bare-metal` | `v1.0-thesis` | Sequence alignment generation (fork of `ARTFL-Project/text-pair`) |
| `tarahmarie/sextant` | `main` | `v1.0-thesis` | Corpus assembly, modelling, and analysis |

A complete reproduction requires checking out all three at their `v1.0-thesis` tags and following the phases below.

---

## Hardware and Software Environment

The v1.0-thesis run was executed on:

- **Hardware:** Apple M2 Max, 12 threads, 96 GB RAM
- **OS:** macOS 26.4.1 (arm64)
- **Python:** 3.11.7
- **Poetry:** 2.3.4
- **CodeCarbon:** 3.2.6 (`OfflineEmissionsTracker`, CPU-load × pinned TDP mode)

Carbon-tracking environment variables:
```bash
SEXTANT_COUNTRY_ISO=GBR        # UK grid intensity profile
SEXTANT_CPU_TDP_WATTS=30       # Conservative laptop TDP
```

---

## Phase 1: TextPAIR Alignment Generation

Run twice, once per corpus, from the `text-pair/` repository.

### 1.1 Stage corpus splits

```bash
# ELTeC
rm -rf in-and-out/eltec_held/*
mkdir -p in-and-out/eltec_held
find ../sextant/projects/eltec-100/splits/ -name "*.xml" -exec cp {} in-and-out/eltec_held/ \;
ls in-and-out/eltec_held/ | wc -l   # 3,514 expected

# Shelley-Lovelace
rm -rf in-and-out/shelley_lovelace_held/*
mkdir -p in-and-out/shelley_lovelace_held
find ../sextant/projects/shelley-lovelace/splits/ -name "*.xml" -exec cp {} in-and-out/shelley_lovelace_held/ \;
ls in-and-out/shelley_lovelace_held/ | wc -l   # 1,332 expected
```

### 1.2 Set absolute source path

```bash
# ELTeC
sed -i '' 's|^source_file_path =.*|source_file_path = /Users/tarah/Library/Mobile Documents/com~apple~CloudDocs/ONGOING-Oxford/OxfordTree/oxford_uni/text-pair/in-and-out/eltec_held|' my_config.ini

# Shelley-Lovelace
sed -i '' 's|^source_file_path =.*|source_file_path = /Users/tarah/Library/Mobile Documents/com~apple~CloudDocs/ONGOING-Oxford/OxfordTree/oxford_uni/text-pair/in-and-out/shelley_lovelace_held|' my_config.ini
```

Adjust the absolute path to match your local checkout.

### 1.3 Run TextPAIR

```bash
# ELTeC
textpair --config=my_config.ini --skip_web_app \
         --output_path=/tmp/textpair-eltec-out --workers=8 eltec_thesis

# Shelley-Lovelace
cd text-pair
textpair --config=my_config.ini --skip_web_app --output_path=/tmp/textpair-shelley-lovelace-out --workers=8 shelley_lovelace_thesis
```

### 1.4 Decompress and stage alignments

```bash
# ELTeC
mv ../sextant/projects/eltec-100/alignments/alignments.jsonl \
   ../sextant/projects/eltec-100/alignments/alignments.jsonl.bak.$(date +%Y%m%d-%H%M%S) 2>/dev/null
mkdir -p ../sextant/projects/eltec-100/alignments
lz4 -d /tmp/textpair-eltec-out/results/alignments.jsonl.lz4 \
       ../sextant/projects/eltec-100/alignments/alignments.jsonl

# Shelley-Lovelace
mv ../sextant/projects/shelley-lovelace/alignments/alignments.jsonl \
   ../sextant/projects/shelley-lovelace/alignments/alignments.jsonl.bak.$(date +%Y%m%d-%H%M%S) 2>/dev/null
mkdir -p ../sextant/projects/shelley-lovelace/alignments
lz4 -d /tmp/textpair-shelley-lovelace-out/results/alignments.jsonl.lz4 \
       ../sextant/projects/shelley-lovelace/alignments/alignments.jsonl
```

### TextPAIR Output Statistics

| Corpus | Pairwise alignments | Banalities flagged | Final passage groups | Decompressed size | Wall-clock |
|---|---|---|---|---|---|
| ELTeC | 103,959 | 60,783 (58%) | 46,613 | 308 MB | ~8 min |
| Shelley-Lovelace | 29,588 | 14,607 (49%) | 15,759 | 91 MB | ~2 min |

---

## Phase 2: Sextant Pipeline (ELTeC, Model Training)

```bash
cd /path/to/sextant
./begin.sh
# Select project = eltec-100, alignments = alignments.jsonl, answer "n" to "run again?"

SEXTANT_COUNTRY_ISO=GBR SEXTANT_CPU_TDP_WATTS=30 poetry run python carbon_run.py
```

The 500-word minimum filter is applied at the load_authors_and_texts stage. For ELTeC, 62 chapters fell below threshold (1.8% exclusion rate).

Wall-clock time: 14.0 minutes.

### ELTeC Run: Headline Numbers

#### Corpus

- 3,514 chapters loaded; 3,452 retained (98.2%)
- 76 unique authors
- 5,956,426 same-year-or-prior text pairs computed
- 103,959 TextPAIR alignments loaded; 103,581 retained (99.6%)

#### Train / Test Split

- Total observations: 5,956,426 pairs
- Same-author pairs: 170,473 (2.86%)
- Cross-author pairs: 5,785,953 (97.14%)
- Training set: 4,765,140 pairs (80%, stratified)
- Test set (held out): 1,191,286 pairs (20%, stratified)

#### Model Performance

| Metric | 10-fold CV (training) | Held-out test |
|---|---|---|
| ROC AUC | 0.784 (±0.002) | 0.783 |
| Accuracy | 0.972 (±0.000) | 0.972 |
| Avg Precision | — | 0.212 |
| Precision | — | 0.829 |
| Recall | — | 0.034 |
| F1 | — | 0.066 |

The low recall is by design: the model is tuned for high precision in the influence-detection use case, where false positives waste humanist time on false leads. The 99% precision on documented influence cases is the metric that matters.

#### Coefficient Estimates (used as frozen weights for SL analysis)

| Variable | Coefficient | Std. Err. | p-value |
|---|---|---|---|
| Constant | -4.207439184487401 (sklearn intercept) | 0.0043 | < 0.001 |
| Hapax Legomena | -1.2319672017708805 | 0.0032 | < 0.001 |
| Sequence Alignment | -0.15314075364182664 | 0.0015 | < 0.001 |
| SVM Stylometry | +0.18399544222763672 | 0.0026 | < 0.001 |

McFadden Pseudo R²: 0.1544.

Z-score normalisation parameters (fitted on ELTeC 80% training split):

| Variable | Mean | Std |
|---|---|---|
| Hapax Jaccard distance | 0.945407929605967 | 0.011339318040662662 |
| Alignment Jaccard distance | 0.9999721042109713 | 0.00025511513289408944 |
| SVM score | 0.32427006589173824 | 0.2573745232415633 |

#### SHAP Decomposition (mean |SHAP value|, normalised)

- Hapax Legomena: 84.5%
- SVM Stylometry: 14.0%
- Sequence Alignment: 1.5%

All three variables retained at C=1.0 and at the stricter C=0.1 under L1 regularisation; each contributes independent signal.

#### Validation Against 8 Documented Influence Cases

| Influence Pair | N pairs | Best Rank | Percentile |
|---|---|---|---|
| Eliot → Lawrence | 4,991 | 218 | 100.00% |
| Thackeray → Disraeli | 6,633 | 1,401 | 99.98% |
| Dickens → Collins | 5,775 | 562 | 99.99% |
| Thackeray → Trollope | 7,571 | 476 | 99.99% |
| Dickens → Hardy | 7,425 | 1,265 | 99.98% |
| Eliot → Hardy | 7,245 | 34 | 100.00% |
| Gaskell → Dickens | 3,914 | 1,588 | 99.97% |
| Brontë → Gaskell | 760 | 264 | 100.00% |

8 of 8 documented influence cases rank above the 99th percentile. Average percentile: 99.99%.

#### Compute Cost

| Metric | Value |
|---|---|
| Total runtime | 842.3 s (14.0 min) |
| Total emissions | 0.373 g CO₂eq |
| Total energy | 1.571 Wh (CPU 0.708 Wh, RAM 0.863 Wh) |
| Avg CPU utilisation | 29.1% |
| Emission rate | 1.596 g CO₂eq/hour |

---

## Phase 3: Sextant Pipeline (Shelley-Lovelace, Database Population)

```bash
cd /path/to/sextant
./begin.sh
# Select project = shelley-lovelace, alignments = alignments.jsonl, answer "n" to "run again?"

SEXTANT_COUNTRY_ISO=GBR SEXTANT_CPU_TDP_WATTS=30 poetry run python carbon_run.py
```

This populates the Shelley-Lovelace SQLite database with hapax intersections, alignment data, and SVM scores. The `logistic_regression.py` step at the end of `carbon_run.py` does train a model on the Shelley-Lovelace data, but those results are not used in the thesis: the next phase applies the frozen ELTeC weights instead. The SL-trained model would have different coefficients (driven by SL's much higher same-author pair fraction, 14.6% vs ELTeC's 2.86%, and the resulting different feature contribution mix); using the ELTeC weights ensures the methodology applied to the thesis target is the same methodology validated against the eight known cases.

For Shelley-Lovelace, 166 chapters fell below the 500-word threshold (12.5% exclusion rate). The higher rate compared to ELTeC reflects corpus composition: short letters (most Lovelace correspondence), subdivided philosophical works (Locke, Whewell), short essay sections (Mary Shelley *Lives*), and short poetic stanzas (Coleridge *Mariner* parts). Items below threshold are documented in `corpus_inventory.xlsx` and remain in the inventory's headline-corpus count of 190 Lovelace items, but are excluded from model fitting.

The thesis-relevant items below threshold include: Note C of the Menabrea translation (471 words), the "poetical science" fragment from folio 210 (91 words), three Mary Shelley *Lives* sections, and five Faraday letters from the scattered Lovelace correspondence. The methodology chapter notes these exclusions; they are invoked in the close-reading chapters as historical and contextual evidence rather than as direct computational findings.

Wall-clock time: 2.2 minutes.

### Shelley-Lovelace Pipeline Statistics

- 1,332 files loaded; 1,166 retained after the 500-word filter (87.5%)
- 54 unique authors
- 679,195 same-year-or-prior text pairs computed
- 29,588 TextPAIR alignments loaded; 29,300 retained (99.0%)
- 666,799 pairs after SVM-score validation
- 577,145 cross-author pairs (the rank denominator for influence claims)
- 98,962 same-author pairs (excluded from cross-author rank denominator)

---

## Phase 4: Apply Frozen ELTeC Weights to Shelley-Lovelace

```bash
cd /path/to/sextant
poetry run python score_shelley_lovelace.py
```

`score_shelley_lovelace.py` loads the Shelley-Lovelace combined-Jaccard table and SVM scores, applies the ELTeC-trained intercept and coefficients (hardcoded in the script header), normalises against the ELTeC training-set means and standard deviations, and ranks all pairs by influence probability.

The script reports percentile rankings against **cross-author pairs only** (matching the ELTeC validation convention used in `logistic_regression.py`). Same-author pairs are computed for context but excluded from the rank denominator used for cross-author influence claims.

Output is written to terminal and to `./projects/shelley-lovelace/results/score_shelley_lovelace_output.txt`.

### Shelley-Lovelace: Headline Author → Lovelace Percentile Summary

Best rank for each source author against Lovelace targets, ranked against 577,145 cross-author pairs:

| Author | Best rank | Percentile | Top probability | Med hapax dist. | Med SVM |
|---|---|---|---|---|---|
| Lovelace (Menabrea translation channel)* | 72 | 99.99th | 0.5851 | 0.969 | 0.607 |
| **Babbage** | **246** | **99.96th** | 0.4072 | 0.961 | 0.229 |
| **Whewell** | **597** | **99.90th** | 0.3132 | 0.958 | 0.235 |
| **Locke** | **851** | **99.85th** | 0.2809 | 0.956 | 0.290 |
| **Somerville** | **1,203** | **99.79th** | 0.2513 | 0.966 | 0.153 |
| **Shelley** (Mary or Percy) | **1,225** | **99.79th** | 0.2498 | 0.966 | 0.105 |
| Goldsmith | 1,673 | 99.71th | 0.2243 | 0.956 | 0.245 |
| Wollstonecraft | 2,157 | 99.63th | 0.2046 | 0.964 | 0.133 |
| Godwin | 2,560 | 99.56th | 0.1910 | 0.965 | 0.147 |
| Lyell | 3,085 | 99.47th | 0.1755 | 0.975 | 0.082 |
| Byron | 7,586 | 98.69th | 0.1172 | 0.978 | 0.175 |
| Coleridge | 8,745 | 98.48th | 0.1094 | 0.959 | 0.225 |

*The "Lovelace → Lovelace" entry at rank 72 is the Menabrea-Lovelace translation pair: Lovelace's English translation of Menabrea is correctly identified by the model as stylistically close to Lovelace's own writing, even though Menabrea was the original author of the ideas. The composite-author marker `Menabrea_Lovelace` makes this a cross-author pair by construction.

Note that Chambers (rank 20,692, 96.41st percentile) and Menabrea_Lovelace (rank 67,388, 88.32nd percentile) appear lower than expected; per-pair inspection shows their SVM scores cluster at 1.000, an artefact of small per-author training samples in the SVM stage rather than a substantive influence finding. Hapax-driven and probability-ranked findings remain the methodologically reliable signals.

### Top Cross-Author Influence Candidate (Independent Methodology Validation)

The single highest-ranked cross-author pair in the entire 577,145-pair Shelley-Lovelace corpus is:

> **Babbage, *Ninth Bridgewater Treatise*, ch. 5 (1837) → Chambers, *Vestiges of the Natural History of Creation*, ch. 14 (1844)**
> rank 1, p = 1.0000, al = 0.880, hap = 0.921, svm = 0.818

All three signals fire strongly. Chambers is suspected to have read Babbage in preparing *Vestiges*; the methodology surfaces this without prior labelling. This pair functions as additional methodology validation independent of the thesis question.

### Compute Cost

| Metric | Value |
|---|---|
| Total runtime | 131.5 s (2.2 min) |
| Total emissions | 0.061 g CO₂eq |
| Total energy | 0.256 Wh |
| Avg CPU utilisation | 46.5% |
| Emission rate | 1.668 g CO₂eq/hour |

### Combined v1.0-thesis Pipeline Cost

| Run | Wall-clock | Emissions |
|---|---|---|
| ELTeC | 14.0 min | 0.373 g CO₂eq |
| Shelley-Lovelace | 2.2 min | 0.061 g CO₂eq |
| **Total** | **16.2 min** | **0.434 g CO₂eq** |

---

## Phase 5: Result Artefact Snapshot

After both pipelines completed, result directories were archived for forensic reproducibility:

```bash
cd /path/to/sextant
DATESTAMP=2026-04-29
mkdir -p corpus_review/_pipeline_run_results_${DATESTAMP}/{eltec-100,shelley-lovelace}
cp -r projects/eltec-100/results/* corpus_review/_pipeline_run_results_${DATESTAMP}/eltec-100/
cp -r projects/shelley-lovelace/results/* corpus_review/_pipeline_run_results_${DATESTAMP}/shelley-lovelace/
```

The snapshot directory is gitignored on disk but is included in the post-submission forensic-reproducibility zip kept by the author.

---

## Notes for the Methodology Chapter

The following items belong in the thesis methodology chapter and are recorded here so they are preserved alongside the run record.

### Cross-author rank denominator

Percentile rankings throughout the thesis are reported against cross-author pairs only (577,145 of 666,799 total). This convention is inherited from the ELTeC validation in `logistic_regression.py` and applied consistently in `score_shelley_lovelace.py`. Same-author pairs are excluded from the denominator because the question being asked is "how does a documented or hypothesised cross-author pair rank against the distribution of unrelated cross-author pairs?" Including same-author pairs would deflate the percentiles by adding a guaranteed-high-probability subset to the comparison.

### 500-word minimum filter

Items below 500 words are excluded from the active corpus during model fitting because hapax legomena and TF-IDF stylometric signals require sufficient text length to stabilise. This threshold is consistent with standard digital-humanities practice (cf. Burrows 2007). The threshold disproportionately excludes short-genre items (letters, subdivided philosophical works, short essay sections), and the methodology chapter must acknowledge which thesis-relevant items fell below threshold.

### Poetical science fragment provenance

The "poetical science" passage (folio 210, 91 words) is one of five fragments from five different letters catalogued as a single 13-page item (007_box44_folio_204) in the Bodleian Library's Dep. Lovelace Byron 44. No opening, no closing, no signature; attribution rests on hand and catalogued grouping. The fragment is below the 500-word threshold and is not in the model's view; its presence in the thesis argument is as historical and contextual evidence supported by close reading, not as a direct computational finding. The methodology chapter is explicit about this distinction.

### AI-assisted transcription validation

Box 42-44 letters were initially transcribed by AI-assisted OCR and validated by a contracted transcriptionist (Lisa Gold). The validation step caught substantive errors that were not flagged by question-mark uncertainty markers in the AI draft, including: "Maker's Will" for "inspiring voice"; "Science may be the Rehearsal" for "I may be the Deborah, the Elijah of Science"; "commercial" for "mesmerical"; "depression" for "illness"; "the processing" for "slow poisoning". This is methodology material: machine learning reduces transcription workload but does not eliminate the need for a skilled (and expensive!) human historic transcriptionist to validate.

### Inventory schema

The corpus inventory (`corpus_inventory.xlsx`, gitignored) is the canonical source of truth for what is in the corpus, what was excluded, and on what basis. Five sheets: `inventory` (2,060 rows, item-level inclusion/exclusion), `appendix_table` (33 rows, aggregated stats by author/source), `exclusion_codes` (21 rows, code definitions with citations), `inclusion_codes` (5 rows, headline tier definitions), `gap_analysis` (28 rows, scholar-suggested works cross-checked against corpus presence). The xlsx is gitignored because it references unpublished Bodleian transcription material under Paper Lion Ltd publication restrictions.

### Composite-influence interpretation

The data supports a rich interpretation: Lovelace's voice in 1843-1844 is computationally close to Babbage (rank 246), Whewell (597), Locke (851), Somerville (1,203), and Shelley (1,225) simultaneously. Each contributes a different stylistic register. The thesis methodology chapter argues for the composite influence reading rather than singling out any one source.

---

## Document Status

This document is committed to the public `tarahmarie/sextant` repository at the `v1.0-thesis` tag. It records the specific run that produced the thesis-cited results and is intended to remain stable; subsequent re-runs of the pipeline (for follow-up papers, methodological refinements, or post-viva work) should not modify this file but rather add new files alongside it.