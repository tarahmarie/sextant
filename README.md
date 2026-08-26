# Sextant

**Interpretable Literary Influence Detection via Classic Machine Learning**

Sextant is a computational method for detecting potential literary influence relationships between texts. It combines three classic machine learning signals (hapax legomena analysis, SVM stylometry, and sequence alignment) into an interpretable model that surfaces candidate influence relationships for humanistic interpretation.

This tool was developed as part of a DPhil dissertation at the Oxford Internet Institute. The methodology prioritises interpretability over black-box neural approaches, enabling scholars to examine the textual evidence behind each computational prediction.

For documentation of the specific pipeline run that produced the thesis results, see [THESIS_REPRODUCTION.md](THESIS_REPRODUCTION.md).

## Design Principles

Sextant was built with the following principles in mind:

- **Sustainability**: Classic ML methods produce a fraction of the carbon footprint of training a transformer. A full ELTeC validation run (5.9M chapter pairs from 100 novels by 76 authors) emits well under a gram of CO₂eq on a modern laptop. Every methodological choice favours computational efficiency.
- **Openness**: All outputs use open standard formats (SQLite, CSV, GraphML). No proprietary formats or vendor lock-in.
- **Transparency**: Every calculation can be audited. The `trace_single_pair.py` script shows exactly how each prediction is made.
- **Reproducibility**: Model parameters (coefficients, scaler values, intercepts) are saved to enable exact replication. Frozen weights from one corpus can be applied to another via `score_shelley_lovelace.py` (or a similar wrapper for your own corpus).
- **Accessibility**: Designed for humanities scholars who can operate a command line, not just computer scientists.

## Table of Contents

1. [How Sextant Works](#how-sextant-works)
2. [Installation](#installation)
3. [Quick Start (ELTeC as Sample Corpus)](#quick-start-eltec-as-sample-corpus)
4. [Project Structure](#project-structure)
5. [Core Pipeline Scripts](#core-pipeline-scripts)
6. [Analysis & Validation Scripts](#analysis--validation-scripts)
7. [Visualisation Scripts](#visualisation-scripts)
8. [Output Files](#output-files)
9. [Auditing & Reproducibility](#auditing--reproducibility)

---

## How Sextant Works

Sextant answers the question: *"Which pairs of texts show unexpected stylistic similarity that might indicate literary influence?"*

The method works by:
1. Training a same-author vs. cross-author classifier on three interpretable signals.
2. Identifying cross-author pairs that the model "mistakes" as same-author.
3. Ranking these pairs by probability; high-probability cross-author pairs are influence candidates.

### The Three Signals

| Signal | What It Measures |
|--------|------------------|
| **Hapax Legomena** | Shared rare vocabulary (words appearing exactly once) |
| **SVM Stylometry** | Writing style similarity via TF-IDF features |
| **Sequence Alignment** | Shared phrases and textual echoes |

The relative contribution of each signal is reported by `logistic_regression.py` after each run; the SHAP value decomposition is in `projects/{name}/results/influence_coefficients_shap_cv.csv`.

All three signals are tested for independent contribution via L1 regularisation; the model logs which signals are retained at C=1.0 and at the stricter C=0.1.

### Why Classic ML?

| Consideration | Sextant (Classic ML) | Neural Approaches |
|---------------|---------------------|-------------------|
| **Carbon footprint** | Fractions of a gram of CO₂eq per run | Hundreds of kg for a single training run |
| **Interpretability** | Full transparency, every coefficient explainable | Black box |
| **Hardware requirements** | Runs on laptop CPU | Often requires GPU |
| **Reproducibility** | Deterministic, parameters saved | Stochastic, version-sensitive |
| **Scholarly value** | Surfaces evidence for close reading | Replaces human interpretation |

---

## Installation

### Prerequisites

- **Python 3.11+** (per `pyproject.toml`'s `^3.11` constraint)
- **Poetry 2.x** for dependency management
- **SQLite3** (open standard, no server required, comes with Python)
- **TextPAIR** (set up separately; see below)
- **lz4** for decompressing TextPAIR output (`brew install lz4` on macOS)
- ~8GB RAM for full corpus analysis

### Setup

```bash
# Clone the repository
git clone https://github.com/tarahmarie/sextant.git
cd sextant

# Install dependencies via Poetry (the supported path)
poetry install
```

The repository ships with `poetry.lock`; `poetry install` reproduces the exact dependency graph used to produce the thesis results.

### Required NLTK Data

The pipeline downloads these automatically on first use, but you can pre-install:

```bash
poetry run python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### TextPAIR Setup

Sextant depends on alignment files produced by TextPAIR (forked from ARTFL-Project for macOS bare-metal compatibility). See the companion repository:

- https://github.com/tarahmarie/text-pair (branch `mac-bare-metal`)

Follow the macOS bare-metal installation instructions in that repository's README. TextPAIR runs as a separate command-line tool; Sextant consumes its output (`alignments.jsonl`).

---

## Quick Start (ELTeC as Sample Corpus)

This walkthrough assumes macOS: TextPAIR here is the mac-bare-metal fork, and the commands below include `sed -i ''`, `ulimit`, and `brew install lz4`. Linux users should adjust the shell-specific commands accordingly.

This walkthrough uses the ELTeC-100 corpus (a benchmark collection of 100 nineteenth-century English-language novels by 76 authors) as a worked example. The full sequence runs end-to-end in roughly 22 minutes on a modern laptop (about 8 minutes for TextPAIR alignment generation plus about 14 minutes for the Sextant pipeline) and produces all model artefacts.

### Before you start: getting the corpus

- **ELTeC splits ship with this repository, by design.** Cloning gives you all 3,514 chapter files (100 novels by 76 authors) under `projects/eltec-100/splits/`. No separate download and no gutenberg-text-splitter run is needed for this walkthrough.
- **ELTeC alignments do not ship** (the `alignments.jsonl` file is roughly 308 MB and is gitignored). You generate it yourself in Phase 1 below; the ELTeC run produces 103,959 pairwise alignments.
- **The Shelley-Lovelace corpus is not public and does not ship with the repo.** Its splits contain restricted archival material and are gitignored, along with its alignments (the corpus-complete run produces 30,510) and databases. Reproducing the thesis Shelley-Lovelace results requires contacting the author to obtain the dataset; once the data is in hand, [THESIS_REPRODUCTION.md](THESIS_REPRODUCTION.md) documents the pipeline that processed it.

For this walkthrough, `{project}` is `eltec-100`.

### Phase 1: TextPAIR Alignment Generation

**Before running TextPAIR, in this same shell, run `ulimit -n` and confirm it returns at least 4096. A fresh terminal resets this, and a low limit silently truncates alignment output with no error.**

Run from the `text-pair/` repository directory.

#### 1.1 Stage source files in TextPAIR's input directory

The corpus splits live in Sextant under `projects/{project}/splits/`, organised in nested subdirectories per the ELTeC naming convention: `{YEAR}-{LANG}{VARIANT}--{Author_First}/` with one or more XML files inside each directory. TextPAIR's loader expects flat files, not nested directories, and will fail with `IsADirectoryError` if pointed at the nested structure directly. Flatten on copy:

```bash
# Clean any prior input
rm -rf in-and-out/{project}_held/*
mkdir -p in-and-out/{project}_held

# Flatten the splits into TextPAIR's expected location.
# `find ... -exec cp` walks all subdirectories and copies every .xml file
# to the flat staging directory.
find ../sextant/projects/{project}/splits/ -name "*.xml" -exec cp {} in-and-out/{project}_held/ \;

# Sanity check: how many files landed?
ls in-and-out/{project}_held/ | wc -l
```

If the file count is lower than the count of XML files in the source tree, there may be filename collisions across subdirectories. Sextant's naming convention scopes chapter numbering inside each author/year directory, so two `chapter_1.xml` files in different directories would collide when flattened. If this happens, rename the colliding files with a per-directory prefix on copy.

#### 1.2 Set the source path in `my_config.ini`

`my_config.ini` is intentionally left with empty path values in the public repository (see TextPAIR README for rationale). Set the source path to your staging directory before each run:

```bash
sed -i '' 's|^source_file_path =.*|source_file_path = /absolute/path/to/text-pair/in-and-out/{project}_held|' my_config.ini
```

Note: `sed -i ''` is macOS/BSD syntax. On Linux, use `sed -i` (no empty-string argument).

The path must be absolute. TextPAIR will fail with `FileNotFoundError: [Errno 2] No such file or directory: ''` if `source_file_path` is empty.

#### 1.3 Run TextPAIR

```bash
textpair --config=my_config.ini \
         --skip_web_app \
         --output_path=/tmp/textpair-{project}-out \
         --workers=8 \
         {project}_thesis
```

Parameters:
- `--skip_web_app` because Sextant consumes the JSONL directly; the web app is not required.
- `--output_path` to a tmpfs path keeps the working set off iCloud Drive.
- `--workers=8` empirically a good fit for an 8-core laptop.
- `{project}_thesis` is the database name used internally by TextPAIR.

For ELTeC, expect roughly 8 minutes wall-clock with 8 workers on an M-series Mac.

The two harmless warnings about `/var/lib/philologic5/web_app/*` relate to the web-app deployment that `--skip_web_app` already disabled.

#### 1.4 Decompress and stage the alignments file for Sextant

TextPAIR's current default is to write `alignments.jsonl.lz4` rather than the uncompressed `alignments.jsonl` that Sextant's `load_alignments.py` expects. Decompress on the way out:

```bash
mv ../sextant/projects/{project}/alignments/alignments.jsonl \
   ../sextant/projects/{project}/alignments/alignments.jsonl.bak.$(date +%Y%m%d-%H%M%S) \
   2>/dev/null

mkdir -p ../sextant/projects/{project}/alignments

lz4 -d /tmp/textpair-{project}-out/results/alignments.jsonl.lz4 \
       ../sextant/projects/{project}/alignments/alignments.jsonl
```

For scale, and as a sanity check on your own output: the ELTeC run produces 103,959 pairwise alignments (about 308 MB decompressed); the corpus-complete Shelley-Lovelace run produces 30,510. If your line count is dramatically lower than expected for the corpus you ran, suspect the `ulimit` hazard flagged at the top of this phase.

### Phase 2: Sextant Pipeline

Run from the `sextant/` repository directory.

#### 2.1 Configure project name and alignments file

```bash
./begin.sh
```

Answer the prompts to select project name and alignments file. When asked "Did you want to run everything again? (y/n)", answer **n** to write `.current_project` and `.alignments_file_name` without firing the pipeline. The carbon-tracked wrapper will run those ten stages itself.

#### 2.2 Run the carbon-tracked pipeline

```bash
SEXTANT_COUNTRY_ISO=GBR SEXTANT_CPU_TDP_WATTS=30 poetry run python carbon_run.py
```

`carbon_run.py` runs the same ten stages as `begin.sh` (init_db → load_authors_and_texts → load_alignments → nltk_data_download → load_hapaxes → load_hapax_intersects → load_relationships → load_jaccard → do_svm → logistic_regression), wrapped in a single CodeCarbon `OfflineEmissionsTracker` context. That is nine pipeline scripts plus one discrete NLTK data-download stage; the NLTK data downloads automatically on first run and the stage is a fast no-op when the data is already present.

Configuration:
- `SEXTANT_COUNTRY_ISO=GBR` selects the UK grid intensity profile. Default is GBR; override for non-UK reproductions.
- `SEXTANT_CPU_TDP_WATTS=30` sets the assumed CPU TDP for the energy-cost estimate. Default 30W approximates a typical modern laptop. Set to match your hardware for a tighter absolute estimate.

The 500-word minimum filter is applied at the load_authors_and_texts stage. Chapters below 500 words are skipped from the analysis because hapax legomena and TF-IDF stylometric signals require sufficient text length to stabilise. The exclusion rate depends on corpus composition: ELTeC has long novels with chapter divisions and exclusions are typically under 2%; corpora composed of letters, short essays, or subdivided philosophical works see higher exclusion rates.

For ELTeC, expect roughly 14 minutes wall-clock for the full ten-stage Sextant pipeline.

Outputs written to `projects/{project}/results/`:

- `emissions.csv` (full CodeCarbon report)
- `emissions_summary.md` (paper-ready table)
- `influence_coefficients_shap_cv.csv` (model coefficients with SHAP contributions)
- `scaler_parameters.csv` (StandardScaler mean/std for each feature)
- `model_intercept.txt` (logistic regression intercept)
- `influence_model_summary_cv.txt` (complete model summary)
- `influence_validation_results.csv` (validation case results, if applicable)

### Applying a Trained Model to a Different Corpus

For thesis or domain-specific work, you may want to train the model on one corpus (e.g. a benchmark like ELTeC) and apply the frozen weights to a different target corpus. The `score_shelley_lovelace.py` script in this repository demonstrates the pattern: it loads the SL combined-Jaccard table and SVM scores, normalises against ELTeC training-set parameters, and ranks all pairs by influence probability.

To adapt for your own corpus, copy `score_shelley_lovelace.py`, point it at your project name, and update the `INTERCEPT`, `COEFS`, `MEANS`, and `STDS` constants from your trained model's output files.

The script reports percentile rankings against **cross-author pairs only**, which matches the convention used in the ELTeC validation. Same-author pairs are computed for context but excluded from the rank denominator used for influence claims.

---

## Project Structure

```
sextant/
├── begin.sh                          # Main entry point (interactive)
├── carbon_run.py                     # Carbon-tracked wrapper for begin.sh stages
├── score_shelley_lovelace.py         # Apply frozen ELTeC weights to SL corpus (template)
├── projects/                         # Project data directories
│   └── eltec-100/                    # ELTeC validation project
│       ├── alignments/               # TextPAIR alignment files (.jsonl)
│       ├── db/                       # SQLite databases (open format)
│       ├── splits/                   # Chapter text files in nested dirs
│       ├── results/                  # Model outputs (gitignored)
│       └── visualisations/           # Network graphs (GraphML, HTML)
├── anchor_case_output/               # Anchor case analysis outputs
└── validation_output/                # Validation analysis outputs (CSV)
```

**Format choices for openness:**
- **SQLite**: Open standard, single-file database, readable by any programming language
- **CSV**: Universal tabular format, opens in Excel, R, Python, or any text editor
- **GraphML**: Open XML standard for networks, supported by Cytoscape, Gephi, NetworkX, igraph
- **XML / TEI**: Chapter files are UTF-8 TEI, no proprietary encoding

---

## Core Pipeline Scripts

### 1. `init_db.py`
Initialises the SQLite database with required tables.

### 2. `load_authors_and_texts.py`
Scans the `splits/` directory, extracts author metadata from TEI headers, populates the `authors` and `all_texts` tables, and applies the 500-word minimum filter.

### 3. `load_alignments.py`
Parses TextPAIR alignment output (JSONL format) and loads sequence alignments into the database. Expects uncompressed `alignments.jsonl`. If TextPAIR produced `alignments.jsonl.lz4`, decompress first (see Phase 1.4).

### 4. `load_hapaxes.py`
Extracts hapax legomena (words appearing exactly once) from each text.

### 5. `load_hapax_intersects.py`
Computes the intersection of hapax legomena between all text pairs.

### 6. `load_relationships.py`
Generates all valid text pairs (filtered to same-year-or-prior pairings) and computes basic statistics.

### 7. `load_jaccard.py`
Computes Jaccard similarity/distance for hapax overlap and alignments, normalised by text length.

### 8. `do_svm.py`
Trains an SVM classifier on TF-IDF features to measure stylistic similarity between chapters and novels.

### 9. `logistic_regression.py`
The main analysis script. Trains a logistic regression model with 80/20 train/test split, 10-fold cross-validation on the training set, SHAP value decomposition, L1 regularisation check, and validation against documented influence cases. Saves model coefficients, scaler parameters, and intercept for reproducibility.

### 10. `score_shelley_lovelace.py`
Template script for applying frozen weights from one corpus to another. Loads the target corpus's combined-Jaccard table and SVM scores, normalises against the source corpus's training-set parameters, computes influence probabilities, and ranks against cross-author pairs only. Adapt the constants at the top of the file to retarget for your own corpus.

---

## Analysis & Validation Scripts

### `trace_single_pair.py`
Audit script that traces a single text pair through the entire pipeline, showing every calculation step. *This is the key transparency tool.*

```bash
poetry run python trace_single_pair.py --pair-id 4448692
poetry run python trace_single_pair.py --source Eliot --source-chapter 79 --target Lawrence --target-chapter 29
poetry run python trace_single_pair.py --interactive
```

Output shows step-by-step:
1. Raw database values
2. Hapax legomena calculation with actual shared words listed
3. SVM stylometry score and novel rankings
4. Sequence alignments (if any) with actual text snippets
5. Logistic regression calculation with z-scores and coefficients
6. Final probability and percentile ranking

### Other analysis scripts

`anchor_case.py`, `validate_influence_pairs.py`, `compare_influence_cases.py`, `show_receipts_sextant.py`, `show_receipts_sextant_lowball.py`, `comprehensive_analysis.py`, `discriminant_analysis.py`, `diagnostic_distributions.py`, `author_max_percentiles.py`, `lowest_influence_pairs.py`, `negative_pairs.py`, `sample_all_pairs.py`, `novel_discovery_analysis.py`, `stat_stats.py`, `export_for_analysis.py`. Prefix all `python` invocations with `poetry run`.

---

## Visualisation Scripts

### `export_graphml.py`
Exports the influence network to GraphML format for use in Cytoscape Desktop, Gephi, or any other network visualisation tool. Produces `.graphml` files at three thresholds (top 10%, top 5%, top 1% of edges).

### `visualize_influence_network.py`
Creates standalone interactive HTML network visualisations. Standalone files; no server, no Jupyter, no installation. Open in any browser.

Note: the `py4cytoscape` and `ipycytoscape` Python integrations were removed from this fork's dependencies in the current version. Cytoscape Desktop integration via the GraphML export remains supported and is the recommended path for desktop visualisation.

---

## Output Files

All output files use **open, non-proprietary formats**.

### Database Files (`projects/{name}/db/`)

| File | Format | Contents |
|------|--------|----------|
| `{project}.db` | SQLite | Main database with all tables |
| `svm.db` | SQLite | SVM stylometry scores |

### Results Files (`projects/{name}/results/`)

| File | Format | Contents |
|------|--------|----------|
| `influence_coefficients_shap_cv.csv` | CSV | Model coefficients with SHAP contributions |
| `scaler_parameters.csv` | CSV | Feature scaling parameters (mean, std) |
| `model_intercept.txt` | Plain text | Logistic regression intercept |
| `influence_model_summary_cv.txt` | Plain text | Complete model summary |
| `influence_validation_results.csv` | CSV | Validation results for documented cases |
| `paper_statistics.txt` / `.csv` | Plain text / CSV | Formatted statistics for citation |
| `emissions.csv` | CSV | CodeCarbon report (kWh, kg CO₂eq, hardware) |
| `emissions_summary.md` | Markdown | Paper-ready compute-cost table |

### Visualisation Files (`projects/{name}/visualisations/`)

| File | Format | Contents |
|------|--------|----------|
| `influence_network_*.graphml` | GraphML (XML) | Network for Cytoscape/Gephi |
| `influence_network_*.html` | HTML | Interactive browser visualisation |

---

## Language Profiles

All language-dependent text processing (tokenisation, stopwords,
lemmatisation, character handling) is defined in one place per language:
[`language_profiles.py`](language_profiles.py). Three code paths consume a
profile — `extract_hapaxes.py`, `hapaxes_1tM.py`, and `do_svm.py` — so a
language is defined exactly once, not scattered across conditionals.

- **Selection is per project, not per run.** `projects/{name}/language`
  contains the profile code (`en` or `fr`); optional override lines such as
  `lemmatiser=simplemma` make parameter choices visible in config. A missing
  file means `en`, so existing projects are untouched.
- **The `en` profile freezes the pre-refactor English behaviour exactly,
  quirks included**, because the English outputs are thesis results.
  `equivalence_harness.py` proves this: `capture` was run at the pre-refactor
  commit over every file in `projects/*/splits`, and `verify` asserts
  byte-identical token streams from the refactored code
  (`python3 equivalence_harness.py verify`).
- **The `fr` profile's decisions** (leading-elision splitting with a closed
  clitic set, explicit œ/æ ligature mapping, NLTK French stopwords, no
  lemmatisation by default) are documented with their reasoning in the
  `language_profiles.py` module docstring. French validation cases live in
  `projects/eltec-fra/validation_cases.json`.
- **Staging a French corpus:** `python3 stage_eltec.py <level1-dir>
  <project>` converts ELTeC level-1 TEI into the splits convention. The
  matching TextPAIR configuration is `my_config_fra.ini` in the text-pair
  fork (identical to the English thesis config except `language = french`
  and the source path).

---

## Auditing & Reproducibility

Sextant is designed for full transparency and reproducibility. *No black boxes.*

### Trace Any Pair

```bash
poetry run python trace_single_pair.py --pair-id 4448692
```

Shows every calculation step from raw data to final probability, including:
- The actual shared hapax legomena (listed by word)
- The exact z-score transformations
- The coefficient multiplications
- The sigmoid probability calculation

### Verify Calculations

The trace script:
1. Loads exact scaler parameters from `scaler_parameters.csv`
2. Loads exact model intercept from `model_intercept.txt`
3. Shows z-score transformations with full arithmetic
4. Shows coefficient multiplication step-by-step
5. Confirms stored values match computed values

### Alignment direction is assigned at scoring time

The raw `alignments` table stores `source`/`target` in TextPAIR's own file-processing order, which is not the influence direction. Sextant assigns direction at scoring time (source = earlier-published text, target = later-published text) in the `combined_jaccard` table. To count or inspect alignments for a given directed pair, join through `combined_jaccard` for source/target identity rather than reading the raw `alignments` columns; querying the raw table directly can return the wrong count or the reversed direction.

### Export for External Verification

All results can be exported for analysis in other tools:
- **R**: Read CSVs directly, or connect to SQLite with `RSQLite`
- **SPSS / Stata / Excel**: Import CSVs
- **Gephi / Cytoscape**: Import GraphML files

### Three-Repository Reproducibility Chain

A complete reproduction uses three companion repositories:

- `gutenberg-text-splitter`: Gutenberg ingestion and ELTeC pattern splitting. Not required for the ELTeC walkthrough, since the ELTeC splits already ship in this repo; needed only if you are splitting a corpus of your own.
- `text-pair` (mac-bare-metal branch): sequence alignment generation.
- `sextant`: corpus assembly, modelling, and analysis.

For the specific commands and parameters that produced the thesis results, see [THESIS_REPRODUCTION.md](THESIS_REPRODUCTION.md).

---

## License

MIT License. See `LICENSE` for details.

---

## Author

**Tarah Wheeler**
DPhil Candidate, Oxford Internet Institute
University of Oxford