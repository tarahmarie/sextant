#!/usr/bin/env python3
"""
export_graphml.py

Exports the Sextant influence network to GraphML and CSV formats for use in
Cytoscape Desktop, Gephi, or any other network visualization tool.

GraphML is an open XML-based standard supported by virtually all
network analysis software. CSV provides maximum compatibility.

Usage:
    python export_graphml.py

Output:
    projects/eltec-100/visualisations/influence_network_*.graphml
    projects/eltec-100/visualisations/influence_network_*_nodes.csv
    projects/eltec-100/visualisations/influence_network_*_edges.csv

To use in Cytoscape Desktop:
    1. Open Cytoscape
    2. File → Import → Network from File
    3. Select the .graphml or _edges.csv file
    4. Apply a layout (Layout → yFiles Organic or Prefuse Force Directed)
    5. Style edges by 'is_validated' (red) and 'percentile' (width)
    6. Style nodes by 'era' (color) and 'total_connections' (size)
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
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

# Author metadata
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


def get_era_label(year):
    """Categorize author by era for coloring."""
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


def load_author_pairs():
    """Load author-level influence scores from CSV."""
    print("Loading author pair data...")
    
    csv_path = "./validation_output/author_max_percentiles.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df):,} author pairs")
        return df
    
    raise FileNotFoundError(f"Data file not found: {csv_path}")


def create_csv_export(author_scores, percentile_threshold=90, output_prefix="influence_network"):
    """
    Create CSV files for Cytoscape import.
    
    Parameters:
        author_scores: DataFrame with source_author, target_author, max_percentile
        percentile_threshold: Only include edges above this percentile
        output_prefix: Prefix for output filenames
    
    Returns:
        Tuple of (nodes_df, edges_df)
    """
    # Filter edges
    threshold = np.percentile(author_scores['max_percentile'], percentile_threshold)
    edges_df = author_scores[author_scores['max_percentile'] >= threshold].copy()
    
    # Get all authors in filtered network
    all_authors = set(edges_df['source_author'].unique()) | set(edges_df['target_author'].unique())
    
    # Calculate node statistics
    out_connections = defaultdict(int)
    in_connections = defaultdict(int)
    max_out_percentile = defaultdict(float)
    max_in_percentile = defaultdict(float)
    
    for _, row in edges_df.iterrows():
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
    
    nodes_out_df = pd.DataFrame(nodes_data)
    
    # Create edges dataframe
    edges_data = []
    for _, row in edges_df.iterrows():
        source = row['source_author']
        target = row['target_author']
        percentile = row['max_percentile']
        n_pairs = row.get('n_chapter_pairs', 0)
        is_validated = (source, target) in VALIDATED_CASES
        
        edges_data.append({
            'source': source,
            'target': target,
            'percentile': round(percentile, 4),
            'n_chapter_pairs': int(n_pairs),
            'is_validated': is_validated,
            'weight': round(percentile / 100, 4),
            'interaction': 'influences'
        })
    
    edges_out_df = pd.DataFrame(edges_data)
    
    # Save CSV files
    nodes_file = f"{OUTPUT_DIR}/{output_prefix}_nodes.csv"
    edges_file = f"{OUTPUT_DIR}/{output_prefix}_edges.csv"
    
    nodes_out_df.to_csv(nodes_file, index=False)
    edges_out_df.to_csv(edges_file, index=False)
    
    print(f"  Saved {nodes_file} ({len(nodes_out_df)} nodes)")
    print(f"  Saved {edges_file} ({len(edges_out_df)} edges)")
    
    return nodes_out_df, edges_out_df


def create_graphml(author_scores, percentile_threshold=90, output_file="influence_network.graphml"):
    """
    Create a GraphML file for Cytoscape/Gephi.
    
    Parameters:
        author_scores: DataFrame with source_author, target_author, max_percentile
        percentile_threshold: Only include edges above this percentile
        output_file: Output filename
    """
    print(f"\nCreating GraphML export...")
    print(f"  Filtering to edges above {percentile_threshold}th percentile")
    
    # Filter edges
    threshold = np.percentile(author_scores['max_percentile'], percentile_threshold)
    edges_df = author_scores[author_scores['max_percentile'] >= threshold].copy()
    print(f"  {len(edges_df)} edges above threshold")
    
    # Get all authors in filtered network
    all_authors = set(edges_df['source_author'].unique()) | set(edges_df['target_author'].unique())
    print(f"  {len(all_authors)} authors in network")
    
    # Calculate node statistics
    out_connections = defaultdict(int)
    in_connections = defaultdict(int)
    max_out_percentile = defaultdict(float)
    max_in_percentile = defaultdict(float)
    
    for _, row in edges_df.iterrows():
        src, tgt, pct = row['source_author'], row['target_author'], row['max_percentile']
        out_connections[src] += 1
        in_connections[tgt] += 1
        max_out_percentile[src] = max(max_out_percentile[src], pct)
        max_in_percentile[tgt] = max(max_in_percentile[tgt], pct)
    
    # Create GraphML structure
    ns = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace('', ns)
    
    graphml = ET.Element('graphml', xmlns=ns)
    
    # Define node attributes
    node_attrs = [
        ('node_label', 'string', 'Author name'),
        ('year', 'int', 'Approximate first publication year'),
        ('era', 'string', 'Literary era category'),
        ('out_degree', 'int', 'Number of authors influenced'),
        ('in_degree', 'int', 'Number of influencing authors'),
        ('total_connections', 'int', 'Total connections'),
        ('max_influence_given', 'double', 'Highest influence percentile (as source)'),
        ('max_influence_received', 'double', 'Highest influence percentile (as target)'),
        ('is_validated_author', 'boolean', 'Appears in a validated influence case'),
    ]
    
    for attr_name, attr_type, desc in node_attrs:
        key = ET.SubElement(graphml, 'key')
        key.set('id', attr_name)
        key.set('for', 'node')
        key.set('attr.name', attr_name)
        key.set('attr.type', attr_type)
        desc_elem = ET.SubElement(key, 'desc')
        desc_elem.text = desc
    
    # Define edge attributes (using unique IDs to avoid conflicts)
    edge_attrs = [
        ('percentile', 'double', 'Influence percentile (0-100)'),
        ('n_chapter_pairs', 'int', 'Number of chapter pairs analyzed'),
        ('is_validated', 'boolean', 'Documented literary influence'),
        ('weight', 'double', 'Edge weight for layout algorithms'),
        ('edge_label', 'string', 'Edge label'),
    ]
    
    for attr_name, attr_type, desc in edge_attrs:
        key = ET.SubElement(graphml, 'key')
        key.set('id', attr_name)
        key.set('for', 'edge')
        key.set('attr.name', attr_name)
        key.set('attr.type', attr_type)
        desc_elem = ET.SubElement(key, 'desc')
        desc_elem.text = desc
    
    # Create graph element
    graph = ET.SubElement(graphml, 'graph')
    graph.set('id', 'influence_network')
    graph.set('edgedefault', 'directed')
    
    # Add nodes
    for author in sorted(all_authors):
        node = ET.SubElement(graph, 'node')
        node.set('id', author)
        
        year = AUTHOR_ERAS.get(author, 1870)
        era = get_era_label(year)
        out_deg = out_connections[author]
        in_deg = in_connections[author]
        is_validated = any(author in case for case in VALIDATED_CASES)
        
        # Add node data
        data_label = ET.SubElement(node, 'data', key='node_label')
        data_label.text = author
        
        data_year = ET.SubElement(node, 'data', key='year')
        data_year.text = str(year)
        
        data_era = ET.SubElement(node, 'data', key='era')
        data_era.text = era
        
        data_out = ET.SubElement(node, 'data', key='out_degree')
        data_out.text = str(out_deg)
        
        data_in = ET.SubElement(node, 'data', key='in_degree')
        data_in.text = str(in_deg)
        
        data_total = ET.SubElement(node, 'data', key='total_connections')
        data_total.text = str(out_deg + in_deg)
        
        data_max_out = ET.SubElement(node, 'data', key='max_influence_given')
        data_max_out.text = f"{max_out_percentile[author]:.4f}"
        
        data_max_in = ET.SubElement(node, 'data', key='max_influence_received')
        data_max_in.text = f"{max_in_percentile[author]:.4f}"
        
        data_validated = ET.SubElement(node, 'data', key='is_validated_author')
        data_validated.text = 'true' if is_validated else 'false'
    
    # Add edges
    edge_id = 0
    for _, row in edges_df.iterrows():
        source = row['source_author']
        target = row['target_author']
        percentile = row['max_percentile']
        n_pairs = row.get('n_chapter_pairs', 0)
        is_validated = (source, target) in VALIDATED_CASES
        
        edge = ET.SubElement(graph, 'edge')
        edge.set('id', f"e{edge_id}")
        edge.set('source', source)
        edge.set('target', target)
        
        data_pct = ET.SubElement(edge, 'data', key='percentile')
        data_pct.text = f"{percentile:.4f}"
        
        data_pairs = ET.SubElement(edge, 'data', key='n_chapter_pairs')
        data_pairs.text = str(int(n_pairs))
        
        data_val = ET.SubElement(edge, 'data', key='is_validated')
        data_val.text = 'true' if is_validated else 'false'
        
        data_weight = ET.SubElement(edge, 'data', key='weight')
        data_weight.text = f"{percentile / 100:.4f}"
        
        data_edge_label = ET.SubElement(edge, 'data', key='edge_label')
        data_edge_label.text = f"{percentile:.1f}%"
        
        edge_id += 1
    
    # Pretty print XML
    xml_string = ET.tostring(graphml, encoding='unicode')
    pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")
    
    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"  Saved to: {output_file}")
    
    return len(all_authors), len(edges_df)


def main():
    print("=" * 70)
    print("SEXTANT NETWORK EXPORT FOR CYTOSCAPE")
    print("(GraphML and CSV formats)")
    print("=" * 70)
    
    # Load data
    author_scores = load_author_pairs()
    
    # Check validated cases
    print("\nValidated influence cases:")
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
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Export at different thresholds
    exports = [
        (90, "top10pct", "Top 10% - comprehensive view"),
        (95, "top5pct", "Top 5% - balanced"),
        (99, "top1pct", "Top 1% - strongest only"),
    ]
    
    for threshold, suffix, description in exports:
        print(f"\n{'='*70}")
        print(f"{description}")
        print("=" * 70)
        
        # CSV export
        print("\nCSV Export:")
        create_csv_export(
            author_scores,
            percentile_threshold=threshold,
            output_prefix=f"influence_network_{suffix}"
        )
        
        # GraphML export
        print("\nGraphML Export:")
        n_nodes, n_edges = create_graphml(
            author_scores,
            percentile_threshold=threshold,
            output_file=f"{OUTPUT_DIR}/influence_network_{suffix}.graphml"
        )
        print(f"  Nodes: {n_nodes}, Edges: {n_edges}")
    
    print("\n" + "=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nFiles saved to {OUTPUT_DIR}/")
    print("\nFor each threshold (top10pct, top5pct, top1pct):")
    print("  • influence_network_*_nodes.csv")
    print("  • influence_network_*_edges.csv")
    print("  • influence_network_*.graphml")
    print("\nTo use CSV in Cytoscape Desktop:")
    print("  1. File → Import → Network from File → select *_edges.csv")
    print("  2. Set Source Column: source, Target Column: target")
    print("  3. File → Import → Table from File → select *_nodes.csv")
    print("  4. Import as 'Node Table Columns', Key column: id")


if __name__ == "__main__":
    main()