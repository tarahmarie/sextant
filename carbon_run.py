#!/usr/bin/env python3
"""Run the sextant pipeline's compute-heavy stages inside a single
CodeCarbon OfflineEmissionsTracker context. Writes emissions.csv to
the current project's results/ directory.

Design goals:
  * No sudo / admin rights required (runs on university-locked machines).
  * No internet required at runtime (uses embedded country grid-intensity
    tables — OfflineEmissionsTracker).
  * Deterministic CPU-power estimation via TDP × CPU-load (MODE_CPU_LOAD)
    rather than hardware sensors. This trades instantaneous accuracy for
    reproducibility across heterogeneous hardware.

Configuration (via environment variables):
  SEXTANT_COUNTRY_ISO     3-letter country code for grid intensity
                          (default "GBR", Oxford / UK National Grid).
  SEXTANT_CPU_TDP_WATTS   Assumed CPU TDP in watts. Default 30 (typical
                          laptop). Set to match your hardware for a
                          tighter estimate; the default is chosen for
                          cross-study comparability.

Usage:
    python3 carbon_run.py

Requires .current_project and .alignments_file_name to be set (done
by begin.sh when you select a project + alignment file).
"""

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# macOS workaround: CodeCarbon's is_psutil_available() checks whether
# psutil.cpu_times().nice > 0.0001, but on Darwin/macOS the 'nice' field is
# always reported as 0.0 (macOS doesn't aggregate nice CPU time the way
# Linux does). That false negative causes CodeCarbon to fall through from
# CPU-load mode into powermetrics, which requires sudo.
#
# Patch the check to trust psutil directly — it IS available on every
# modern Python install that has codecarbon as a dep. Must run before
# OfflineEmissionsTracker is imported so the reference is fresh.
# ---------------------------------------------------------------------------
import sys as _sys

if _sys.platform == "darwin":
    from codecarbon.core import cpu as _cc_cpu

    def _psutil_really_available() -> bool:
        try:
            import psutil  # noqa: F401

            psutil.cpu_percent(interval=0.0, percpu=False)
            return True
        except Exception:
            return False

    _cc_cpu.is_psutil_available = _psutil_really_available

from codecarbon import OfflineEmissionsTracker

REPO = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COUNTRY_ISO = os.environ.get("SEXTANT_COUNTRY_ISO", "GBR")
CPU_TDP_WATTS = float(os.environ.get("SEXTANT_CPU_TDP_WATTS", "30"))

# ---------------------------------------------------------------------------
# Resolve current project + alignment file (set by begin.sh)
# ---------------------------------------------------------------------------
current_project_file = REPO / ".current_project"
alignments_name_file = REPO / ".alignments_file_name"

if not current_project_file.exists() or not alignments_name_file.exists():
    print(
        "✗ Missing .current_project or .alignments_file_name.\n"
        "  Run ./begin.sh first, select your project and alignment file,\n"
        "  then answer 'n' at the 'run everything again?' prompt to exit\n"
        "  without running. Then re-run: python3 carbon_run.py",
        file=sys.stderr,
    )
    sys.exit(1)

project = current_project_file.read_text().strip()
alignments_file = alignments_name_file.read_text().strip()

results_dir = REPO / "projects" / project / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# Wipe stale emissions.csv so each run starts with a clean one-row report
# (CodeCarbon defaults to append, which mixes failed + successful runs).
emissions_csv = results_dir / "emissions.csv"
if emissions_csv.exists():
    print(f"  Removing stale emissions report: {emissions_csv.name}")
    emissions_csv.unlink()

# Mirror begin.sh::load_from_db — wipe stale dbs before init_db.py so that
# re-runs don't hit UNIQUE constraint violations on the existing rows.
db_dir = REPO / "projects" / project / "db"
db_dir.mkdir(parents=True, exist_ok=True)
for stale in (db_dir / f"{project}.db", db_dir / f"{project}-predictions.db"):
    if stale.exists():
        print(f"  Removing stale db: {stale.name}")
        stale.unlink()

# ---------------------------------------------------------------------------
# Pipeline stages (same order as begin.sh::load_from_db)
# ---------------------------------------------------------------------------
STAGES = [
    ["python3", "init_db.py"],
    ["python3", "load_authors_and_texts.py"],
    ["python3", "load_alignments.py", alignments_file],
    [
        "python3",
        "-c",
        "import nltk; nltk.download('punkt_tab'); "
        "nltk.download('stopwords'); nltk.download('wordnet')",
    ],
    ["python3", "load_hapaxes.py"],
    ["python3", "load_hapax_intersects.py"],
    ["python3", "load_relationships.py"],
    ["python3", "load_jaccard.py"],
    ["python3", "do_svm.py"],
    ["python3", "logistic_regression.py"],
]

# ---------------------------------------------------------------------------
# Tracker: offline + TDP-load mode → no sudo, no network, reproducible.
# ---------------------------------------------------------------------------
tracker = OfflineEmissionsTracker(
    project_name=f"sextant-{project}",
    output_dir=str(results_dir),
    output_file="emissions.csv",
    country_iso_code=COUNTRY_ISO,
    log_level="warning",
    measure_power_secs=15,
    tracking_mode="process",
    force_mode_cpu_load=True,       # use CPU-load × TDP (no hw sensors)
    force_cpu_power=CPU_TDP_WATTS,  # pin TDP → works on unknown CPUs
)

print("🌱 CodeCarbon tracking started (offline / estimate mode)")
print(f"   Project:         sextant-{project}")
print(f"   Country (grid):  {COUNTRY_ISO}")
print(f"   Assumed CPU TDP: {CPU_TDP_WATTS:.1f} W")
print(f"   Emissions CSV:   {results_dir / 'emissions.csv'}")
print()

tracker.start()
try:
    for stage in STAGES:
        print(f"\n▶  {' '.join(stage)}")
        result = subprocess.run(stage, cwd=REPO)
        if result.returncode != 0:
            print(
                f"✗ Stage failed: {' '.join(stage)} (exit {result.returncode})",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
finally:
    emissions = tracker.stop()

print(f"\n🌱 Total estimated emissions: {emissions:.6f} kg CO₂eq")
print(f"   Full report: {results_dir / 'emissions.csv'}")
print(
    "\n   Note: this is a TDP-based estimate (no hardware power sensors).\n"
    "   Numbers are comparable across runs with the same SEXTANT_CPU_TDP_WATTS\n"
    "   and SEXTANT_COUNTRY_ISO settings."
)

# ---------------------------------------------------------------------------
# Generate a paper-ready markdown summary from the single-row emissions.csv
# ---------------------------------------------------------------------------
import csv  # noqa: E402

with open(emissions_csv, newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))

duration_s = float(row["duration"])
emissions_kg = float(row["emissions"])
energy_kwh = float(row["energy_consumed"])
cpu_energy_kwh = float(row["cpu_energy"])
ram_energy_kwh = float(row["ram_energy"])
cpu_power_w = float(row["cpu_power"])
ram_power_w = float(row["ram_power"])
cpu_util_pct = float(row["cpu_utilization_percent"])
ram_used_gb = float(row["ram_used_gb"])

# Phone-streaming CO2 comparison: ~36 g CO2eq/hr of HD streaming
# (≈1 minute equivalent per 0.6 g). Source: Carbon Trust 2021.
streaming_seconds_equiv = (emissions_kg * 1000 / 0.6) * 60

summary_lines = [
    f"# Carbon report — sextant/{project}",
    "",
    f"**Run timestamp:** {row['timestamp']}",
    f"**Run ID:** `{row['run_id']}`",
    "",
    "## Methodology",
    "",
    "- Tracker: CodeCarbon `OfflineEmissionsTracker` v"
    f"{row['codecarbon_version']}",
    "- Mode: CPU-load × pinned TDP (no hardware power sensors, no sudo).",
    f"- Assumed CPU TDP: **{CPU_TDP_WATTS:.1f} W** "
    f"(override via `SEXTANT_CPU_TDP_WATTS`)",
    f"- Grid intensity: **{row['country_name']} ({row['country_iso_code']})** "
    f"(override via `SEXTANT_COUNTRY_ISO`)",
    f"- Tracking scope: `{row['tracking_mode']}` "
    "(this process tree only, not the whole machine).",
    f"- Hardware: {row['cpu_model']} × {row['cpu_count']} threads, "
    f"{int(float(row['ram_total_size']))} GB RAM, "
    f"{row['os']}, Python {row['python_version']}.",
    "",
    "## Results",
    "",
    "| Metric | Value |",
    "|---|---|",
    f"| Total runtime | {duration_s:,.1f} s ({duration_s/60:.1f} min) |",
    f"| **Total emissions** | **{emissions_kg*1000:.3f} g CO₂eq** |",
    f"| Total energy consumed | {energy_kwh*1000:.3f} Wh |",
    f"| &nbsp;&nbsp;CPU energy | {cpu_energy_kwh*1000:.3f} Wh "
    f"(avg draw {cpu_power_w:.2f} W) |",
    f"| &nbsp;&nbsp;RAM energy | {ram_energy_kwh*1000:.3f} Wh "
    f"(avg draw {ram_power_w:.2f} W) |",
    f"| Avg CPU utilization | {cpu_util_pct:.1f}% |",
    f"| Avg RAM usage | {ram_used_gb:.2f} GB |",
    f"| Emission rate | {(emissions_kg*3600/duration_s)*1000:.3f} g CO₂eq/hour |",
    "",
    "## Context",
    "",
    f"- Equivalent to ~{streaming_seconds_equiv:.0f} seconds of HD video "
    "streaming (Carbon Trust 2021 estimate: ~36 g CO₂eq/hour).",
    "- Reproducibility: this number is an estimate and depends on the "
    "assumed CPU TDP, RAM power model, and grid intensity. Identical "
    "settings on different hardware will yield comparable numbers; "
    "swap `SEXTANT_CPU_TDP_WATTS` to match your machine for a tighter "
    "absolute estimate.",
    "",
]

summary_path = results_dir / "emissions_summary.md"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
print(f"\n📄 Markdown summary: {summary_path}")
print("\n" + "\n".join(summary_lines))
