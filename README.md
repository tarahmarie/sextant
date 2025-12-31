# Sextant

**Interpretable Literary Influence Detection via Classic Machine Learning**

Sextant is a computational method for detecting potential literary influence relationships between texts. It combines three classic machine learning signals—hapax legomena analysis, SVM stylometry, and sequence alignment—into an interpretable model that surfaces candidate influence relationships for humanistic interpretation.

This tool was developed as part of a DPhil dissertation at the Oxford Internet Institute. The methodology prioritises interpretability over black-box neural approaches, enabling scholars to examine the textual evidence behind each computational prediction.

## Design Principles

Sextant was built with the following principles in mind:

- **Sustainability**: Classic ML methods produce 18.5g CO₂ vs. 652kg for training BERT—a 35,000× difference. Every methodological choice favours computational efficiency.
- **Openness**: All outputs use open standard formats (SQLite, CSV, GraphML). No proprietary formats or vendor lock-in.
- **Transparency**: Every calculation can be audited. The `trace_single_pair.py` script shows exactly how each prediction is made.
- **Reproducibility**: Model parameters (coefficients, scaler values, intercepts) are saved to enable exact replication.
- **Accessibility**: Designed for humanities scholars who can operate a command line, not just computer scientists.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Core Pipeline Scripts](#core-pipeline-scripts)
6. [Analysis & Validation Scripts](#analysis--validation-scripts)
7. [Visualisation Scripts](#visualisation-scripts)
8. [Utility Scripts](#utility-scripts)
9. [Output Files](#output-files)
10. [Running the Full Pipeline](#running-the-full-pipeline)
11. [Auditing & Reproducibility](#auditing--reproducibility)

---

## Overview

Sextant answers the question: *"Which pairs of texts show unexpected stylistic similarity that might indicate literary influence?"*

The method works by:
1. Training a same-author vs. cross-author classifier on three interpretable signals
2. Identifying cross-author pairs that the model "mistakes" as same-author
3. Ranking these pairs by probability—high-probability cross-author pairs are influence candidates

### The Three Signals

| Signal | Contribution | What It Measures |
|--------|-------------|------------------|
| **Hapax Legomena** | ~85% | Shared rare vocabulary (words appearing exactly once) |
| **SVM Stylometry** | ~13% | Writing style similarity via TF-IDF features |
| **Sequence Alignment** | ~2% | Shared phrases and textual echoes |

### Key Results

- ROC AUC: 0.78 (10-fold cross-validation)
- All 8 documented Victorian influence cases rank above 99th percentile
- Environmental impact: 18.5g CO₂ (vs. 652kg for BERT training)

### Why Classic ML?

| Consideration | Sextant (Classic ML) | Neural Approaches |
|---------------|---------------------|-------------------|
| **Carbon footprint** | 18.5g CO₂ | 652kg CO₂ (BERT) |
| **Interpretability** | Full transparency—every coefficient explainable | Black box |
| **Hardware requirements** | Runs on laptop CPU | Often requires GPU |
| **Reproducibility** | Deterministic, parameters saved | Stochastic, version-sensitive |
| **Scholarly value** | Surfaces evidence for close reading | Replaces human interpretation |

---

## Installation

### Prerequisites

- Python 3.10+
- SQLite3 *(open standard, no server required)*
- ~8GB RAM for full corpus analysis

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/sextant.git
cd sextant

# Install dependencies
pip install -r requirements.txt

# Or using Poetry
poetry install
```

### Required NLTK Data

The pipeline will download these automatically, but you can pre-install:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## Quick Start

### Interactive Mode

```bash
./begin.sh
```

This launches an interactive menu to:
- Create a new project
- Select an existing project
- Run the full pipeline

### Running Analysis on Existing Data

If you already have a populated database:

```bash
# Run the main logistic regression analysis
python logistic_regression.py

# Audit a specific text pair
python trace_single_pair.py --pair-id 4448692

# Generate evidence for anchor case
python show_receipts_sextant.py

# Export network for Cytoscape
python export_graphml.py
```

---

## Project Structure

```
sextant/
├── begin.sh                          # Main entry point (interactive)
├── projects/                         # Project data directories
│   └── eltec-100/                    # Example project
│       ├── alignments/               # TextPAIR alignment files (.jsonl)
│       ├── db/                       # SQLite databases (open format)
│       │   ├── eltec-100.db          # Main database (hapax, alignments, pairs)
│       │   └── svm.db                # SVM stylometry scores
│       ├── splits/                   # Chapter text files (plain text)
│       ├── results/                  # Model outputs (CSV format)
│       └── visualisations/           # Network graphs (GraphML, HTML)
├── anchor_case_output/               # Anchor case analysis outputs
├── validation_output/                # Validation analysis outputs (CSV)
└── notebooks/                        # Jupyter notebooks for exploration
```

**Format choices for openness:**
- **SQLite**: Open standard, single-file database, readable by any programming language
- **CSV**: Universal tabular format, opens in Excel, R, Python, or any text editor
- **GraphML**: Open XML standard for networks, supported by Cytoscape, Gephi, NetworkX, igraph
- **Plain text**: Chapter files are UTF-8 text, no proprietary encoding

---

## Core Pipeline Scripts

These scripts form the main data processing pipeline. **Run them in order** via `begin.sh` or manually:

### 1. `init_db.py`
**Purpose:** Initialises the SQLite database with required tables.

```bash
python init_db.py
```

Creates empty tables: `authors`, `all_texts`, `text_pairs`, `alignments`, `hapaxes`, `hapax_overlaps`, `stats_all`, `combined_jaccard`, etc.

*SQLite chosen for portability—entire database is a single file that can be shared, backed up, or queried with any SQLite client.*

---

### 2. `load_authors_and_texts.py`
**Purpose:** Scans the `splits/` directory, extracts author metadata from TEI headers, and populates the `authors` and `all_texts` tables.

```bash
python load_authors_and_texts.py
```

**Input:** Text files in `projects/{name}/splits/{author-dir}/{chapter-file}`
**Output:** Populated `authors` and `all_texts` tables with text IDs, author IDs, word counts, and chapter numbers.

---

### 3. `load_alignments.py`
**Purpose:** Parses TextPAIR alignment output (JSONL format) and loads sequence alignments into the database.

```bash
python load_alignments.py alignments.jsonl
```

**Input:** JSONL file from TextPAIR containing aligned passages
**Output:** Populated `alignments` table with source/target passages and word counts.

*JSONL is human-readable and streamable—each line is valid JSON.*

---

### 4. `load_hapaxes.py`
**Purpose:** Extracts hapax legomena (words appearing exactly once) from each text.

```bash
python load_hapaxes.py
```

**Output:** Populated `hapaxes` table with each text's unique vocabulary.

---

### 5. `load_hapax_intersects.py`
**Purpose:** Computes the intersection of hapax legomena between all text pairs.

```bash
python load_hapax_intersects.py
```

**Output:** Populated `hapax_overlaps` table with shared rare words for each pair.

---

### 6. `load_relationships.py`
**Purpose:** Generates all valid text pairs and computes basic statistics.

```bash
python load_relationships.py
```

**Output:** Populated `text_pairs` and `stats_all` tables.

---

### 7. `load_jaccard.py`
**Purpose:** Computes Jaccard similarity/distance for hapax overlap and alignments, normalised by text length.

```bash
python load_jaccard.py
```

**Output:** Populated `combined_jaccard` table with:
- `hap_jac_sim` / `hap_jac_dis`: Hapax Jaccard similarity/distance
- `al_jac_sim` / `al_jac_dis`: Alignment Jaccard similarity/distance

---

### 8. `do_svm.py`
**Purpose:** Trains an SVM classifier on TF-IDF features to measure stylistic similarity between chapters and novels.

```bash
python do_svm.py
```

**Output:** Creates `svm.db` with `chapter_assessments` table containing probability scores for each chapter resembling each novel's style.

*SVM chosen over neural networks for interpretability and efficiency—trains in minutes on CPU.*

---

### 9. `logistic_regression.py`
**Purpose:** The main analysis script. Trains a logistic regression model with proper train/test split and SHAP value decomposition.

```bash
python logistic_regression.py
```

**Features:**
- 80/20 train/test split with stratification
- 10-fold cross-validation on training set
- SHAP values for theoretically-grounded feature contributions
- L1 regularisation check (confirms all variables contribute independent signal)
- Validation against 8 documented influence cases
- Saves model coefficients, scaler parameters, and intercept for reproducibility

**Output Files (in `projects/{name}/results/`):**
- `influence_coefficients_shap_cv.csv`: Model coefficients with SHAP contributions
- `scaler_parameters.csv`: StandardScaler mean/std for each feature
- `model_intercept.txt`: Logistic regression intercept
- `influence_model_summary_cv.txt`: Complete model summary
- `influence_validation_results.csv`: Validation case results

*All outputs in CSV/text format—no proprietary model files. Anyone can inspect the exact coefficients.*

---

## Analysis & Validation Scripts

These scripts perform deeper analysis after the main pipeline has run.

### `trace_single_pair.py`
**Purpose:** Audit script that traces a single text pair through the entire pipeline, showing every calculation step. *This is the key transparency tool.*

```bash
# Random cross-author pair
python trace_single_pair.py

# Specific pair by ID
python trace_single_pair.py --pair-id 4448692

# Specific pair by author/chapter
python trace_single_pair.py --source Eliot --source-chapter 79 --target Lawrence --target-chapter 29

# Interactive mode
python trace_single_pair.py --interactive
```

**Output:** Step-by-step trace showing:
1. Raw database values
2. Hapax legomena calculation with actual shared words listed
3. SVM stylometry score and novel rankings
4. Sequence alignments (if any) with actual text snippets
5. Logistic regression calculation with z-scores and coefficients
6. Final probability and percentile ranking

*Every number is shown with its derivation—nothing is hidden.*

---

### `anchor_case.py`
**Purpose:** Generates detailed analysis and visualisations for the Eliot→Lawrence anchor case.

```bash
python anchor_case.py
```

**Output (in `anchor_case_output/`):**
- `anchor_case_results.txt`: Summary statistics
- `variable_contribution_chart.png`: Bar chart of feature contributions
- `percentile_distribution.png`: Histogram showing anchor case ranking
- `all_eliot_lawrence_pairs.csv`: All Eliot-Lawrence chapter pairs ranked

---

### `validate_influence_pairs.py`
**Purpose:** Validates multiple documented literary influence relationships.

```bash
# List all novels in corpus
python validate_influence_pairs.py --list-novels

# Run validation on known pairs
python validate_influence_pairs.py --validate

# Query specific pair
python validate_influence_pairs.py --query "Dickens" "Collins"
```

---

### `compare_influence_cases.py`
**Purpose:** Compares high-scoring (anchor) cases with low-scoring (negative) cases to demonstrate what Sextant detects.

```bash
python compare_influence_cases.py --sextant-root .
```

---

### `show_receipts_sextant.py`
**Purpose:** Generates detailed textual evidence for the Eliot→Lawrence anchor case, showing the actual shared hapax legomena and aligned passages.

```bash
python show_receipts_sextant.py --sextant-root .
```

*"Show the receipts"—presents the raw textual evidence behind the computational prediction.*

---

### `show_receipts_sextant_lowball.py`
**Purpose:** Same as above but configured for a low-signal pair (Cross→Conrad) as a negative control comparison.

```bash
python show_receipts_sextant_lowball.py --sextant-root .
```

---

### `comprehensive_analysis.py`
**Purpose:** Computes extended metrics for author pair analysis including mean/median percentiles, effect sizes (Cohen's d), and bootstrap confidence intervals.

```bash
python comprehensive_analysis.py
```

---

### `discriminant_analysis.py`
**Purpose:** Analyses what distinguishes documented influence pairs from implausible pairings, examining distribution shapes and individual variable contributions.

```bash
python discriminant_analysis.py
```

---

### `diagnostic_distributions.py`
**Purpose:** Examines score distributions for key author pairs to understand percentile patterns.

```bash
python diagnostic_distributions.py
```

---

### `author_max_percentiles.py`
**Purpose:** Calculates the maximum percentile for all author pairs in the corpus.

```bash
python author_max_percentiles.py
```

**Output:** `validation_output/author_max_percentiles.csv`

---

### `lowest_influence_pairs.py`
**Purpose:** Finds book pairings with the lowest influence scores to identify negative controls.

```bash
python lowest_influence_pairs.py
```

---

### `negative_pairs.py`
**Purpose:** Multithreaded version for ranking all author pairs by influence score.

```bash
python negative_pairs.py
```

---

### `sample_all_pairs.py`
**Purpose:** Fast version that samples 1% of pairs to quickly identify lowest-scoring pairings.

```bash
python sample_all_pairs.py
```

---

### `novel_discovery_analysis.py`
**Purpose:** Extended anchor case analysis covering both Eliot→Lawrence (positive influence) and Thackeray→Disraeli (rivalry/contrast case).

```bash
python novel_discovery_analysis.py
```

---

### `stat_stats.py`
**Purpose:** Regenerates all statistical results for paper citations, outputting exact values for abstract and methods sections.

```bash
python stat_stats.py
```

**Output:**
- `projects/{name}/results/paper_statistics.txt`
- `projects/{name}/results/paper_statistics.csv`

---

### `export_for_analysis.py`
**Purpose:** Exports targeted subsets of the database for external analysis (R, SPSS, etc.).

```bash
python export_for_analysis.py [project_name]
```

*Designed for interoperability—export to any statistical software.*

---

## Visualisation Scripts

These scripts generate network visualisations of the influence relationships.

### `export_graphml.py`
**Purpose:** Exports the influence network to GraphML format for use in Cytoscape Desktop, Gephi, or any other network visualisation tool.

```bash
python export_graphml.py
```

**Output (in `projects/{name}/visualisations/`):**
- `influence_network_top10pct.graphml`: Comprehensive view (top 10% of edges)
- `influence_network_top5pct.graphml`: Balanced view (top 5%)
- `influence_network_top1pct.graphml`: Strongest connections only (top 1%)

**GraphML format chosen because:**
- Open XML standard (not proprietary)
- Supported by Cytoscape, Gephi, NetworkX, igraph, yEd, and virtually all network tools
- Human-readable (it's just XML)
- Preserves all node and edge attributes

**Node attributes included:**
- `label`: Author name
- `year`: Approximate first publication year
- `era`: Literary era category (for colour mapping)
- `out_degree` / `in_degree` / `total_connections`: For size mapping
- `is_validated_author`: Whether author appears in a documented influence case

**Edge attributes included:**
- `percentile`: Influence score (0-100)
- `is_validated`: `true` for the 8 documented influence cases (for colour mapping)
- `weight`: For layout algorithms
- `n_chapter_pairs`: Number of chapter comparisons

**To use in Cytoscape Desktop:**
1. File → Import → Network from File
2. Select a `.graphml` file
3. Apply layout: Layout → yFiles Organic (or Prefuse Force Directed)
4. Style nodes: Fill Color → Column: `era` → Discrete Mapping
5. Style edges: Stroke Color → Column: `is_validated` → Discrete Mapping (`true` = red)
6. Style edge width: Width → Column: `percentile` → Continuous Mapping

---

### `visualize_influence_network.py`
**Purpose:** Creates interactive HTML network visualisations using Pyvis. No desktop application required—opens in any web browser.

```bash
python visualize_influence_network.py
```

**Output (in `projects/{name}/visualisations/`):**
- `influence_network_top5pct.html`
- `influence_network_top2pct.html`
- `influence_network_top1pct.html`

**Features:**
- Drag nodes to rearrange layout
- Scroll to zoom
- Click nodes/edges for details
- Colour-coded by literary era
- Validated influence cases highlighted in red

*Standalone HTML files—no server, no installation, no account required. Just open in a browser.*

---

## Utility Scripts

### `util.py`
Shared utility functions used across the codebase:
- `get_project_name()`: Read current project from `.current_project`
- `get_algnments_file_name()`: Read alignment file from `.alignments_file_name`
- `getListOfFiles()`: Recursively list files in directory
- `extract_author_name()`: Parse author from TEI header
- `get_word_count_for_text()`: Count words in text

### `database_ops.py`
Database operations and table management:
- Table creation and indexing
- Insert functions for all data types
- Jaccard similarity calculations
- Database cleanup and optimisation

### `predict_ops.py`
Prediction-related database operations:
- Setup for author prediction tables
- Helper functions for pair lookups

### `hapaxes_1tM.py`
Core hapax extraction logic:
- `remove_tei_lines_from_text()`: Strip TEI markup
- Hapax identification functions

### `utils/get_choices.py`
Helper for interactive menu choices.

### `show_previous_averages.py`
Displays statistics from the previous pipeline run without recomputing.

---

## Output Files

All output files use **open, non-proprietary formats** that can be read by any software.

### Database Files (`projects/{name}/db/`)

| File | Format | Contents |
|------|--------|----------|
| `{project}.db` | SQLite | Main database with all tables |
| `svm.db` | SQLite | SVM stylometry scores |

*SQLite databases can be opened with DB Browser for SQLite (free), DBeaver (free), Python, R, or any SQLite client.*

### Results Files (`projects/{name}/results/`)

| File | Format | Contents |
|------|--------|----------|
| `influence_coefficients_shap_cv.csv` | CSV | Model coefficients with SHAP contributions |
| `scaler_parameters.csv` | CSV | Feature scaling parameters (mean, std) |
| `model_intercept.txt` | Plain text | Logistic regression intercept |
| `influence_model_summary_cv.txt` | Plain text | Complete model summary |
| `influence_validation_results.csv` | CSV | Validation results for documented cases |
| `paper_statistics.txt` | Plain text | Formatted statistics for citation |
| `top_influence_candidates.csv` | CSV | Highest-scoring cross-author pairs |

### Visualisation Files (`projects/{name}/visualisations/`)

| File | Format | Contents |
|------|--------|----------|
| `influence_network_*.graphml` | GraphML (XML) | Network for Cytoscape/Gephi |
| `influence_network_*.html` | HTML | Interactive browser visualisation |

### Analysis Output Files

| Directory | Contents |
|-----------|----------|
| `anchor_case_output/` | Anchor case analysis, charts (PNG), CSVs |
| `validation_output/` | Validation analysis results (CSV) |

---

## Running the Full Pipeline

### Option 1: Interactive (Recommended for First Run)

```bash
./begin.sh
```

Follow the prompts to:
1. Create a new project or select existing
2. Point to your alignment file
3. The pipeline runs automatically

### Option 2: Manual Step-by-Step

```bash
# Set project name
echo "eltec-100" > .current_project
echo "alignments.jsonl" > .alignments_file_name

# Run pipeline in order
python init_db.py
python load_authors_and_texts.py
python load_alignments.py alignments.jsonl
python load_hapaxes.py
python load_hapax_intersects.py
python load_relationships.py
python load_jaccard.py
python do_svm.py
python logistic_regression.py

# Generate visualisations
python export_graphml.py
python visualize_influence_network.py
```

### Option 3: Analysis Only (Database Already Populated)

```bash
python logistic_regression.py
python trace_single_pair.py
python anchor_case.py
python export_graphml.py
```

---

## Auditing & Reproducibility

Sextant is designed for full transparency and reproducibility. *No black boxes.*

### Trace Any Pair

```bash
python trace_single_pair.py --pair-id 4448692
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
5. Confirms stored values match computed values (with checkmarks)

### Reproduce Results

All model parameters are saved during training:
- Coefficients with standard errors and p-values
- Scaler means and standard deviations (exact values from training set)
- Model intercept
- SHAP contribution percentages

*Anyone with the code and data can reproduce the exact same results.*

### Export for External Verification

All results can be exported for analysis in other tools:
- **R**: Read CSVs directly, or connect to SQLite with `RSQLite`
- **SPSS**: Import CSVs
- **Stata**: Import CSVs
- **Excel**: Open CSVs directly
- **Gephi/Cytoscape**: Import GraphML files

---

## License

MIT License. See `LICENSE` for details.

*MIT chosen as one of the most permissive open source licences—maximises reusability.*

---

## Author

**Tarah Wheeler**  
DPhil Candidate, Oxford Internet Institute  
University of Oxford