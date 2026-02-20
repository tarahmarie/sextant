import json, sys

from tqdm import tqdm

from database_ops import (insert_alignments_to_db, insert_last_run_stats_to_db,
                          read_all_author_names_and_ids_from_db,
                          read_all_text_names_and_ids_from_db,
                          read_all_text_names_by_id_from_db,
                          read_all_text_pair_names_and_ids_from_db,
                          read_author_names_by_id_from_db)
from util import (fix_the_author_name_from_aligns, get_project_name,
                  getCountOfFiles)

if len(sys.argv) > 1:
    alignments_file=sys.argv[1]
    print(f"Using alignments file: {alignments_file}")
else:
    alignments_file="alignments.jsonl"

# Goes and gets the pairs from the fresh db after having loaded authors 
# and texts into it during load_authors_and_texts.py. Matches the 
# authors and pairs up to the pairs found in the alignments json 
# file which has been previously generated in docker and moved to this 
# projects file location. Adds the values in sequence alignment to 
# the project db for later calculation.

author_and_id_dict = read_all_author_names_and_ids_from_db()
inverted_authors = read_author_names_by_id_from_db()
text_and_id_dict = read_all_text_names_and_ids_from_db()
inverted_text_and_id_dict = read_all_text_names_by_id_from_db()
project_name = get_project_name()
total_file_count = getCountOfFiles(f'./projects/{project_name}/splits')
text_pairs, inverted_pairs = read_all_text_pair_names_and_ids_from_db()
transactions = []

# Track skipped alignments
skipped_alignments = 0
skipped_texts = set()

#Alignments
i = 1
with open(f'./projects/{project_name}/alignments/{alignments_file}', 'r') as the_json:
    raw_json_list = list(the_json)
    length_json_list = len(raw_json_list)
    total_entries_in_alignments_file = length_json_list
    while i <= length_json_list:
        pbar = tqdm(desc='Loading Alignments', total=length_json_list, colour="magenta", bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]')
        for json_str in raw_json_list:
            result = json.loads(json_str)
            
            # Handle double .xml extension from TextPAIR output
            source_filename = result['source_filename'].split('TEXT/')[1].replace('.xml', '').replace('.xml', '')
            target_filename = result['target_filename'].split('TEXT/')[1].replace('.xml', '').replace('.xml', '')
            
            # Skip alignments for texts that were filtered out (e.g., below word count minimum)
            if source_filename not in text_and_id_dict:
                skipped_alignments += 1
                skipped_texts.add(source_filename)
                i += 1
                pbar.update(1)
                continue
            if target_filename not in text_and_id_dict:
                skipped_alignments += 1
                skipped_texts.add(target_filename)
                i += 1
                pbar.update(1)
                continue
            
            temp_source_author = fix_the_author_name_from_aligns(result['source_author'])
            source_author = author_and_id_dict.get(temp_source_author, '')
            source_text_name = text_and_id_dict[source_filename]
            temp_target_author = fix_the_author_name_from_aligns(result['target_author'])
            target_author = author_and_id_dict.get(temp_target_author, '')
            target_text_name = text_and_id_dict[target_filename]
            try:
                pair_id = inverted_pairs[(source_text_name, target_text_name)]
            except KeyError:
                pair_id = inverted_pairs[(target_text_name, source_text_name)]
            #The split below is part of the text-pair output path.
            transactions.append((source_text_name, target_text_name, result['source_passage'], result['target_passage'], source_author, target_author, len(result['source_passage'].split(' ')), len(result['target_passage'].split(' ')), pair_id))
            i+=1
            pbar.update(1)
        pbar.close()

# Report skipped alignments
if skipped_alignments > 0:
    print(f"\n⚠️  Skipped {skipped_alignments} alignments referencing {len(skipped_texts)} filtered text(s):")
    for text in sorted(skipped_texts):
        print(f"   {text}")
    print(f"\nLoaded {len(transactions)} alignments ({len(transactions) / total_entries_in_alignments_file * 100:.1f}% of alignments file)\n")
else:
    print(f"\n✓ Loaded all {len(transactions)} alignments.\n")

insert_alignments_to_db(transactions)
insert_last_run_stats_to_db(len(transactions), total_file_count)