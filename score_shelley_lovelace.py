#!/usr/bin/env python3
from model import load_and_score

project = 'shelley-lovelace'
df, N = load_and_score(project)

print(f"\n{'='*80}")
print(f"=== TOP 25 PAIRS BY INFLUENCE PROBABILITY ===")
print(f"{'='*80}\n")
for i, row in df.head(25).iterrows():
    print(f"{row['rank']:5d}. p={row['prob']:.4f}  {row['source_name']} -> {row['target_name']}")

# Shelley -> Lovelace
sl = df[df['source_name'].str.contains('Shelley') & df['target_name'].str.contains('Lovelace')]
print(f"\n{'='*80}")
print(f"=== SHELLEY -> LOVELACE PAIRS ({len(sl)} pairs) ===")
print(f"{'='*80}\n")
for _, row in sl.iterrows():
    print(f"  rank {row['rank']:5d}/{N}  p={row['prob']:.4f}  hap={row['hap_jac_dis']:.6f}  al={row['al_jac_dis']:.6f}  svm={row['svm_score']:.4f}  {row['source_name']} -> {row['target_name']}")

if len(sl) > 0:
    best = sl['rank'].min()
    print(f"\n  Best rank: {best} of {N} (top {best/N*100:.1f}%)")
    print(f"  Percentile: {(1 - best/N)*100:.1f}th")

# Also show any -> Lovelace
al = df[df['target_name'].str.contains('Lovelace')]
print(f"\n{'='*80}")
print(f"=== ALL AUTHORS -> LOVELACE (top 20) ===")
print(f"{'='*80}\n")
for _, row in al.head(20).iterrows():
    print(f"  rank {row['rank']:5d}/{N}  p={row['prob']:.4f}  {row['source_name']} -> {row['target_name']}")