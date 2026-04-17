# Carbon report — sextant/eltec-100

**Run timestamp:** 2026-04-17T12:03:16
**Run ID:** `5a570ae7-364a-4bd8-9f25-4d8de2b9ef6f`

## Methodology

- Tracker: CodeCarbon `OfflineEmissionsTracker` v3.2.6
- Mode: CPU-load × pinned TDP (no hardware power sensors, no sudo).
- Assumed CPU TDP: **30.0 W** (override via `SEXTANT_CPU_TDP_WATTS`)
- Grid intensity: **United Kingdom (GBR)** (override via `SEXTANT_COUNTRY_ISO`)
- Tracking scope: `process` (this process tree only, not the whole machine).
- Hardware: Apple M2 Max × 12 threads, 96 GB RAM, macOS-26.4.1-arm64-arm-64bit, Python 3.11.7.

## Results

| Metric | Value |
|---|---|
| Total runtime | 846.2 s (14.1 min) |
| **Total emissions** | **0.346 g CO₂eq** |
| Total energy consumed | 1.457 Wh |
| &nbsp;&nbsp;CPU energy | 0.703 Wh (avg draw 2.99 W) |
| &nbsp;&nbsp;RAM energy | 0.754 Wh (avg draw 3.21 W) |
| Avg CPU utilization | 37.0% |
| Avg RAM usage | 35.26 GB |
| Emission rate | 1.473 g CO₂eq/hour |

## Context

- Equivalent to ~35 seconds of HD video streaming (Carbon Trust 2021 estimate: ~36 g CO₂eq/hour).
- Reproducibility: this number is an estimate and depends on the assumed CPU TDP, RAM power model, and grid intensity. Identical settings on different hardware will yield comparable numbers; swap `SEXTANT_CPU_TDP_WATTS` to match your machine for a tighter absolute estimate.
