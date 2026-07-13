mv projects/eltec-100/alignments/alignments.jsonl \
   projects/eltec-100/alignments/alignments.jsonl.bak.$(date +%Y%m%d-%H%M%S) \
   2>/dev/null

mkdir -p projects/eltec-100/alignments
mkdir -p projects/eltec-100/results

lz4 -d /tmp/textpair-eltec-100-out/results/alignments.jsonl.lz4 \
       ../sextant/projects/eltec-100/alignments/alignments.jsonl