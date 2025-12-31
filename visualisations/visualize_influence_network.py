#!/usr/bin/env python3
"""
visualize_influence_network.py

Creates interactive network visualisations and CSV exports of literary 
influence relationships. Outputs standalone HTML files that can be opened 
in any browser, plus CSV files for Cytoscape import.

Usage:
    python visualize_influence_network.py

Output:
    projects/eltec-100/visualisations/influence_network_*.html
    projects/eltec-100/visualisations/influence_network_*_nodes.csv
    projects/eltec-100/visualisations/influence_network_*_edges.csv
"""

import pandas as pd
import numpy as np
from pyvis.network import Network
from collections import defaultdict
import os

# Configuration
PROJECT = "eltec-100"
OUTPUT_DIR = f"./projects/{PROJECT}/visualisations"

# Documented influence cases (for highlighting)
VALIDATED_CASES = [
    ("Eliot", "Lawrence"),
    ("Thackeray", "Disraeli"),
    ("Dickens", "Collins"),
    ("Thackeray", "Trollope"),
    ("Dickens", "Hardy"),
    ("Eliot", "Hardy"),
    ("Gaskell", "Dickens"),
    ("Brontë", "Gaskell"),
]

# Author eras (approximate earliest publication year in corpus)
AUTHOR_ERAS = {
    'Ainsworth': 1839, 'Braddon': 1862, 'Brontë': 1847, 'Bulwer': 1828,
    'Caird': 1883, 'Collins': 1860, 'Conrad': 1895, 'Corelli': 1886,
    'Craik': 1850, 'Cross': 1895, 'Dickens': 1836, 'Disraeli': 1826,
    'Eliot': 1859, 'Gaskell': 1848, 'Gissing': 1884, 'Grand': 1888,
    'Hardy': 1871, 'Harraden': 1893, 'Henty': 1868, 'Hope': 1894,
    'James': 1871, 'Jenkins': 1870, 'Kingsley': 1850, 'Lawrence': 1911,
    'Lee': 1880, 'Lewis': 1918, 'Lyall': 1882, 'Lytton': 1828,
    'Macdonald': 1858, 'Mallock': 1877, 'Meredith': 1856, 'Moore': 1883,
    'Morris': 1890, 'Oliphant': 1849, 'Ouida': 1863, 'Reade': 1852,
    'Reynolds': 1844, 'Schreiner': 1883, 'Shorthouse': 1881, 'Silberrad': 1899,
    'Sinclair': 1904, 'Skene': 1846, 'Stevenson': 1883, 'Stoker': 1890,
    'Thackeray': 1844, 'Trollope': 1847, 'Tupper': 1844, 'Ward': 1881,
    'Wells': 1895, 'Wood': 1861, 'Yonge': 1853, 'Yeats': 1891,
}


def load_author_pairs_from_csv():
    """Load author-level influence scores from CSV."""
    print("Loading author pair data from CSV...")
    
    csv_path = "./validation_output/author_max_percentiles.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} author pairs from {csv_path}")
        return df
    
    # Fallback to top candidates if author_max_percentiles doesn't exist
    csv_path = f"./projects/{PROJECT}/results/top_influence_candidates.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Aggregate to author level
        author_df = df.groupby(['source_author_name', 'target_author_name']).agg({
            'influence_score': 'max'
        }).reset_index()
        author_df.columns = ['source_author', 'target_author', 'max_percentile']
        print(f"  Loaded and aggregated from {csv_path}")
        return author_df
    
    raise FileNotFoundError("No author pair data found!")


def get_era_color(author):
    """Get node color based on author's era."""
    year = AUTHOR_ERAS.get(author, 1870)
    
    if year < 1850:
        return "#c0392b"  # Dark red - early Victorian
    elif year < 1865:
        return "#e74c3c"  # Red - mid-early Victorian
    elif year < 1880:
        return "#e67e22"  # Orange - mid Victorian
    elif year < 1895:
        return "#f39c12"  # Yellow/gold - late Victorian
    else:
        return "#3498db"  # Blue - Edwardian/Modern


def get_era_label(year):
    """Get era label for a year."""
    if year < 1850:
        return "early_victorian"
    elif year < 1865:
        return "mid_early_victorian"
    elif year < 1880:
        return "mid_victorian"
    elif year < 1895:
        return "late_victorian"
    else:
        return "edwardian"


def create_csv_export(top_edges, output_prefix):
    """
    Create CSV files for Cytoscape import alongside the HTML visualisation.
    
    Parameters:
        top_edges: DataFrame of filtered edges
        output_prefix: Prefix for output filenames (without extension)
    """
    all_authors = set(top_edges['source_author'].unique()) | set(top_edges['target_author'].unique())
    
    # Calculate node statistics
    out_connections = defaultdict(int)
    in_connections = defaultdict(int)
    max_out_percentile = defaultdict(float)
    max_in_percentile = defaultdict(float)
    
    for _, row in top_edges.iterrows():
        src, tgt, pct = row['source_author'], row['target_author'], row['max_percentile']
        out_connections[src] += 1
        in_connections[tgt] += 1
        max_out_percentile[src] = max(max_out_percentile[src], pct)
        max_in_percentile[tgt] = max(max_in_percentile[tgt], pct)
    
    # Create nodes dataframe
    nodes_data = []
    for author in sorted(all_authors):
        year = AUTHOR_ERAS.get(author, 1870)
        era = get_era_label(year)
        out_deg = out_connections[author]
        in_deg = in_connections[author]
        is_validated = any(author in case for case in VALIDATED_CASES)
        
        nodes_data.append({
            'id': author,
            'label': author,
            'year': year,
            'era': era,
            'out_degree': out_deg,
            'in_degree': in_deg,
            'total_connections': out_deg + in_deg,
            'max_influence_given': round(max_out_percentile[author], 4),
            'max_influence_received': round(max_in_percentile[author], 4),
            'is_validated_author': is_validated
        })
    
    nodes_df = pd.DataFrame(nodes_data)
    
    # Create edges dataframe
    edges_data = []
    for _, row in top_edges.iterrows():
        source = row['source_author']
        target = row['target_author']
        percentile = row['max_percentile']
        n_pairs = row.get('n_chapter_pairs', 0)
        is_validated = (source, target) in VALIDATED_CASES
        
        edges_data.append({
            'source': source,
            'target': target,
            'percentile': round(percentile, 4),
            'n_chapter_pairs': int(n_pairs) if pd.notna(n_pairs) else 0,
            'is_validated': is_validated,
            'weight': round(percentile / 100, 4),
            'interaction': 'influences'
        })
    
    edges_df = pd.DataFrame(edges_data)
    
    # Save CSV files
    nodes_file = f"{output_prefix}_nodes.csv"
    edges_file = f"{output_prefix}_edges.csv"
    
    nodes_df.to_csv(nodes_file, index=False)
    edges_df.to_csv(edges_file, index=False)
    
    print(f"  CSV: {nodes_file} ({len(nodes_df)} nodes)")
    print(f"  CSV: {edges_file} ({len(edges_df)} edges)")
    
    return nodes_df, edges_df


def create_network_visualization(author_scores, percentile_threshold=95, output_file="influence_network.html"):
    """
    Create an interactive network visualisation using Pyvis, plus CSV exports.
    """
    print(f"\nCreating network visualisation...")
    print(f"  Filtering to edges above {percentile_threshold}th percentile")
    
    # Filter to top edges
    threshold = np.percentile(author_scores['max_percentile'], percentile_threshold)
    top_edges = author_scores[author_scores['max_percentile'] >= threshold].copy()
    print(f"  {len(top_edges)} edges above threshold (percentile >= {threshold:.2f})")
    
    # Create CSV export alongside HTML
    output_prefix = output_file.replace('.html', '')
    create_csv_export(top_edges, output_prefix)
    
    # Create network
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#fafafa",
        font_color="#333333",
        directed=True,
        notebook=False,
        select_menu=True,
        filter_menu=True,
    )
    
    # Physics settings for nice layout
    net.barnes_hut(
        gravity=-5000,
        central_gravity=0.3,
        spring_length=250,
        spring_strength=0.04,
        damping=0.09,
    )
    
    # Collect all authors that appear in top edges
    all_authors = set(top_edges['source_author'].unique()) | set(top_edges['target_author'].unique())
    
    # Calculate node sizes based on connections
    out_connections = defaultdict(int)
    in_connections = defaultdict(int)
    for _, row in top_edges.iterrows():
        out_connections[row['source_author']] += 1
        in_connections[row['target_author']] += 1
    
    # Add nodes
    print(f"  Adding {len(all_authors)} author nodes")
    for author in all_authors:
        year = AUTHOR_ERAS.get(author, 1870)
        out_conn = out_connections[author]
        in_conn = in_connections[author]
        total_conn = out_conn + in_conn
        
        size = 20 + (total_conn * 4)
        color = get_era_color(author)
        
        is_validated_author = any(author in case for case in VALIDATED_CASES)
        border_width = 3 if is_validated_author else 1
        border_color = "#2c3e50" if is_validated_author else color
        
        net.add_node(
            author,
            label=author,
            title=f"<b>{author}</b> (c. {year})<br>Influences: {out_conn}<br>Influenced by: {in_conn}",
            size=size,
            color={
                'background': color,
                'border': border_color,
                'highlight': {'background': '#ecf0f1', 'border': '#2c3e50'}
            },
            borderWidth=border_width,
            font={'size': 16, 'face': 'Georgia, serif', 'color': '#2c3e50'}
        )
    
    # Add edges
    print(f"  Adding {len(top_edges)} influence edges")
    
    min_pct = top_edges['max_percentile'].min()
    max_pct = top_edges['max_percentile'].max()
    
    for _, row in top_edges.iterrows():
        source = row['source_author']
        target = row['target_author']
        percentile = row['max_percentile']
        n_pairs = row.get('n_chapter_pairs', 'N/A')
        
        is_validated = (source, target) in VALIDATED_CASES
        
        if max_pct > min_pct:
            norm_pct = (percentile - min_pct) / (max_pct - min_pct)
        else:
            norm_pct = 0.5
        width = 1 + norm_pct * 6
        
        if is_validated:
            color = "#c0392b"
            width = width * 1.5
        else:
            gray = int(180 - norm_pct * 100)
            color = f"#{gray:02x}{gray:02x}{gray:02x}"
        
        net.add_edge(
            source,
            target,
            value=width,
            title=f"<b>{source} → {target}</b><br>Percentile: {percentile:.2f}%<br>Chapter pairs: {n_pairs}",
            color={'color': color, 'highlight': '#c0392b'},
            arrows={'to': {'enabled': True, 'scaleFactor': 0.6}},
            smooth={'type': 'curvedCW', 'roundness': 0.2},
        )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save
    print(f"  HTML: {output_file}")
    net.save_graph(output_file)
    
    # Add legend
    with open(output_file, 'r') as f:
        html = f.read()
    
    legend_html = """
    <div style="position: absolute; top: 10px; left: 10px; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); font-family: Georgia, serif; font-size: 13px; z-index: 1000; max-width: 280px;">
        <h2 style="margin: 0 0 15px 0; font-size: 18px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">Victorian Literary Influence</h2>
        
        <p style="margin: 0 0 10px 0; color: #7f8c8d; font-size: 11px;">Network of detected stylistic similarities that may indicate literary influence.</p>
        
        <p style="margin: 15px 0 8px 0; font-weight: bold; color: #2c3e50;">Node Color = Author's Era</p>
        <div style="display: flex; flex-wrap: wrap; gap: 5px;">
            <span style="background: #c0392b; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px;">Pre-1850</span>
            <span style="background: #e74c3c; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px;">1850-65</span>
            <span style="background: #e67e22; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px;">1865-80</span>
            <span style="background: #f39c12; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px;">1880-95</span>
            <span style="background: #3498db; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px;">Post-1895</span>
        </div>
        
        <p style="margin: 15px 0 8px 0; font-weight: bold; color: #2c3e50;">Edge Style</p>
        <p style="margin: 4px 0; font-size: 12px;"><span style="color: #c0392b; font-weight: bold;">━━</span> Documented influence (validated)</p>
        <p style="margin: 4px 0; font-size: 12px;"><span style="color: #888;">━━</span> Predicted influence candidate</p>
        <p style="margin: 4px 0; font-size: 12px; color: #7f8c8d;">Thickness = influence score</p>
        
        <p style="margin: 15px 0 8px 0; font-weight: bold; color: #2c3e50;">Interactions</p>
        <p style="margin: 4px 0; font-size: 11px; color: #7f8c8d;">• Drag nodes to rearrange</p>
        <p style="margin: 4px 0; font-size: 11px; color: #7f8c8d;">• Scroll to zoom</p>
        <p style="margin: 4px 0; font-size: 11px; color: #7f8c8d;">• Click node/edge for details</p>
        <p style="margin: 4px 0; font-size: 11px; color: #7f8c8d;">• Double-click to focus</p>
        
        <p style="margin: 15px 0 0 0; font-size: 10px; color: #bdc3c7; border-top: 1px solid #ecf0f1; padding-top: 10px;">Generated by Sextant<br>Oxford Internet Institute</p>
    </div>
    """
    
    html = html.replace('<body>', f'<body>\n{legend_html}')
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"\n✓ Visualisation saved to: {output_file}")
    
    return net


def main():
    print("=" * 70)
    print("SEXTANT INFLUENCE NETWORK VISUALISATION")
    print("(HTML and CSV formats)")
    print("=" * 70)
    
    # Load data
    author_scores = load_author_pairs_from_csv()
    
    # Print stats
    print(f"\nDataset summary:")
    print(f"  Total author pairs: {len(author_scores):,}")
    print(f"  Unique source authors: {author_scores['source_author'].nunique()}")
    print(f"  Unique target authors: {author_scores['target_author'].nunique()}")
    
    # Check validated cases
    print(f"\nValidated influence cases in data:")
    for source, target in VALIDATED_CASES:
        match = author_scores[
            (author_scores['source_author'] == source) & 
            (author_scores['target_author'] == target)
        ]
        if len(match) > 0:
            pct = match['max_percentile'].values[0]
            print(f"  ✓ {source} → {target}: {pct:.2f}th percentile")
        else:
            print(f"  ✗ {source} → {target}: not found")
    
    # Print top pairs
    print("\nTop 15 author pairs by influence score:")
    print("-" * 60)
    top15 = author_scores.nlargest(15, 'max_percentile')
    for _, row in top15.iterrows():
        validated = "✓" if (row['source_author'], row['target_author']) in VALIDATED_CASES else " "
        print(f"  {validated} {row['source_author']:15} → {row['target_author']:15} {row['max_percentile']:.4f}%")
    
    # Create visualisations at different thresholds
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    visualisations = [
        (95, "influence_network_top5pct.html", "Top 5%"),
        (98, "influence_network_top2pct.html", "Top 2%"),
        (99, "influence_network_top1pct.html", "Top 1%"),
    ]
    
    for threshold, filename, desc in visualisations:
        print(f"\n{'='*70}")
        print(f"{desc} - threshold {threshold}th percentile")
        print("=" * 70)
        create_network_visualization(
            author_scores, 
            percentile_threshold=threshold,
            output_file=f"{OUTPUT_DIR}/{filename}"
        )
    
    print("\n" + "=" * 70)
    print("VISUALISATION COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files in {OUTPUT_DIR}/:")
    print("\nFor each threshold (top5pct, top2pct, top1pct):")
    print("  • influence_network_*.html (interactive browser visualisation)")
    print("  • influence_network_*_nodes.csv (for Cytoscape)")
    print("  • influence_network_*_edges.csv (for Cytoscape)")


if __name__ == "__main__":
    main()