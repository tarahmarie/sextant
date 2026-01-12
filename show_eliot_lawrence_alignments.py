"""
Show the actual aligned text passages between Eliot and Lawrence.

The top Eliot→Lawrence pair is:
- Source: 1861-ENG18610—Eliot-chapter_12 (Silas Marner)
- Target: 1920-ENG19200—Lawrence-chapter_13 (Women in Love)

This script retrieves the actual passages that TextPAIR identified as alignments.
"""

import sqlite3
from util import get_project_name


def main():
    project_name = get_project_name()
    db_path = f"./projects/{project_name}/db/{project_name}.db"
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    print("=" * 80)
    print("ALIGNED PASSAGES: Eliot → Lawrence")
    print("=" * 80)
    
    # First, find the text IDs for the specific chapters
    cursor.execute("""
        SELECT text_id, source_filename, chapter_num 
        FROM all_texts 
        WHERE source_filename LIKE '%Eliot%' AND chapter_num = '12'
    """)
    eliot_texts = cursor.fetchall()
    
    cursor.execute("""
        SELECT text_id, source_filename, chapter_num 
        FROM all_texts 
        WHERE source_filename LIKE '%Lawrence%' AND chapter_num = '13'
    """)
    lawrence_texts = cursor.fetchall()
    
    print(f"\nEliot chapter 12 texts found: {len(eliot_texts)}")
    for t in eliot_texts:
        print(f"  {t['text_id']}: {t['source_filename']}")
    
    print(f"\nLawrence chapter 13 texts found: {len(lawrence_texts)}")
    for t in lawrence_texts:
        print(f"  {t['text_id']}: {t['source_filename']}")
    
    # Get the text IDs
    eliot_ids = [t['text_id'] for t in eliot_texts]
    lawrence_ids = [t['text_id'] for t in lawrence_texts]
    
    if not eliot_ids or not lawrence_ids:
        print("\nCould not find matching texts!")
        return
    
    # Query alignments between these specific texts
    print("\n" + "=" * 80)
    print("ALIGNMENTS BETWEEN ELIOT CH12 AND LAWRENCE CH13")
    print("=" * 80)
    
    # Build query for all combinations
    placeholders_e = ','.join('?' * len(eliot_ids))
    placeholders_l = ','.join('?' * len(lawrence_ids))
    
    query = f"""
        SELECT 
            a.source_filename,
            a.target_filename,
            a.source_passage,
            a.target_passage,
            a.length_source_passage,
            a.length_target_passage,
            t1.source_filename as source_name,
            t2.source_filename as target_name
        FROM alignments a
        JOIN all_texts t1 ON a.source_filename = t1.text_id
        JOIN all_texts t2 ON a.target_filename = t2.text_id
        WHERE a.source_filename IN ({placeholders_e})
          AND a.target_filename IN ({placeholders_l})
        ORDER BY a.length_source_passage DESC
    """
    
    cursor.execute(query, eliot_ids + lawrence_ids)
    alignments = cursor.fetchall()
    
    print(f"\nFound {len(alignments)} alignment(s)\n")
    
    if len(alignments) == 0:
        # Try the reverse direction
        print("Trying reverse direction (Lawrence → Eliot)...")
        query = f"""
            SELECT 
                a.source_filename,
                a.target_filename,
                a.source_passage,
                a.target_passage,
                a.length_source_passage,
                a.length_target_passage,
                t1.source_filename as source_name,
                t2.source_filename as target_name
            FROM alignments a
            JOIN all_texts t1 ON a.source_filename = t1.text_id
            JOIN all_texts t2 ON a.target_filename = t2.text_id
            WHERE a.source_filename IN ({placeholders_l})
              AND a.target_filename IN ({placeholders_e})
            ORDER BY a.length_source_passage DESC
        """
        cursor.execute(query, lawrence_ids + eliot_ids)
        alignments = cursor.fetchall()
        print(f"Found {len(alignments)} alignment(s) in reverse direction\n")
    
    for i, a in enumerate(alignments, 1):
        print(f"--- Alignment {i} ---")
        print(f"Source: {a['source_name']}")
        print(f"Target: {a['target_name']}")
        print(f"Source passage ({a['length_source_passage']} words):")
        print(f"  \"{a['source_passage']}\"")
        print(f"\nTarget passage ({a['length_target_passage']} words):")
        print(f"  \"{a['target_passage']}\"")
        print()
    
    # Now let's also look at ALL Eliot→Lawrence alignments to see the patterns
    print("\n" + "=" * 80)
    print("ALL ELIOT → LAWRENCE ALIGNMENTS (any chapter)")
    print("=" * 80)
    
    cursor.execute("""
        SELECT 
            a.source_passage,
            a.target_passage,
            a.length_source_passage,
            a.length_target_passage,
            t1.source_filename as source_name,
            t2.source_filename as target_name,
            auth1.author_name as source_author,
            auth2.author_name as target_author
        FROM alignments a
        JOIN all_texts t1 ON a.source_filename = t1.text_id
        JOIN all_texts t2 ON a.target_filename = t2.text_id
        JOIN authors auth1 ON a.source_author = auth1.id
        JOIN authors auth2 ON a.target_author = auth2.id
        WHERE auth1.author_name LIKE '%Eliot%'
          AND auth2.author_name LIKE '%Lawrence%'
        ORDER BY a.length_source_passage DESC
        LIMIT 20
    """)
    
    all_alignments = cursor.fetchall()
    print(f"\nFound {len(all_alignments)} Eliot→Lawrence alignments (showing top 20 by length)\n")
    
    for i, a in enumerate(all_alignments, 1):
        print(f"--- Alignment {i} ---")
        print(f"Source: {a['source_name']}")
        print(f"Target: {a['target_name']}")
        print(f"Source ({a['length_source_passage']} words): \"{a['source_passage'][:200]}{'...' if len(a['source_passage']) > 200 else ''}\"")
        print(f"Target ({a['length_target_passage']} words): \"{a['target_passage'][:200]}{'...' if len(a['target_passage']) > 200 else ''}\"")
        print()
    
    # Count total alignments per author pair for context
    print("\n" + "=" * 80)
    print("ALIGNMENT COUNTS BY ANCHOR CASE")
    print("=" * 80)
    
    anchor_cases = [
        ('Eliot', 'Lawrence'),
        ('Thackeray', 'Disraeli'),
        ('Dickens', 'Collins'),
        ('Thackeray', 'Trollope'),
        ('Dickens', 'Hardy'),
        ('Eliot', 'Hardy'),
        ('Gaskell', 'Dickens'),
        ('Bront', 'Gaskell'),
    ]
    
    print(f"\n{'Source → Target':<25} {'# Alignments':>15}")
    print("-" * 45)
    
    for source, target in anchor_cases:
        cursor.execute("""
            SELECT COUNT(*) as cnt
            FROM alignments a
            JOIN authors auth1 ON a.source_author = auth1.id
            JOIN authors auth2 ON a.target_author = auth2.id
            WHERE auth1.author_name LIKE ?
              AND auth2.author_name LIKE ?
        """, (f'%{source}%', f'%{target}%'))
        
        count = cursor.fetchone()['cnt']
        print(f"{source} → {target:<15} {count:>15,}")
    
    conn.close()


if __name__ == "__main__":
    main()
