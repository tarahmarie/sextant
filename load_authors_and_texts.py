# This script is finding, matching, cleaning, and counting all the authors
# and texts inside splits to get ready to do some math on them all. Adds
# them to a fresh db in the paired format for later calculation.

import itertools
from dataclasses import dataclass, field

from tqdm import tqdm

from database_ops import (insert_authors_to_db, insert_dirs_to_db,
                          insert_text_pairs_to_db, insert_texts_to_db,
                          read_all_text_names_and_ids_from_db)
from hapaxes_1tM import remove_tei_lines_from_text
from util import (extract_author_name, fix_alignment_file_names,
                  get_date_from_tei_header, get_project_name,
                  get_word_count_for_text, getCountOfFiles, getListOfFiles)

# Minimum word count for texts to be included in the corpus.
# Based on Burrows (2007) and Koppel et al. (2007) stylometric thresholds.
# Eder (2015) recommends 2,500-5,000 words for robust attribution, but
# 500 words is acceptable for constrained attribution tasks.
MIN_WORD_COUNT = 500

project_name = get_project_name()
list_of_files = getListOfFiles(f'./projects/{project_name}/splits')
file_count = getCountOfFiles(f'./projects/{project_name}/splits')
number_of_combinations = sum(1 for e in itertools.combinations(list_of_files, 2))

@dataclass
class Text:
    id: int = field(default=0)
    content: str = field(default="")
    date: int = field(default=0000)
    chapter_num: int = field(default=0)
    length: int = field(default=0000)

print("\n")
#All the Texts
dirs = {}
unique_dir_id = 0
seen_dirs = []

authors = {}
seen_authors = []
unique_author_id = 0

texts = {}
seen_texts = []
unique_text_id = 0

dates = {}
seen_dates = []
unique_date_id = 0

# Track skipped files for reporting
skipped_files = []

i = 1
while i <= file_count:
    temp_text = Text()
    pbar = tqdm(desc='Loading All Texts', total=file_count, colour="yellow", bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]')
    for file in list_of_files:
        the_dir = file.split('/')[4]
        name_of_text = file.split('/')[5]

        # We need just the middle bit of our name_of_text to match the SVM db
        # NOTE: Someone should standardize this, yeah?
        short_name_part_one = name_of_text.split('-')[1]
        short_name_for_svm = short_name_part_one.split('-')[0]

        #Python doesn't seem to want to have a read() and a readline() call to the same file handle.
        #So, we'll open it twice.
        with open(file, 'r') as temp_file:
            content = temp_file.read()
            author = extract_author_name(content)
            temp_text.date = get_date_from_tei_header(content)

            if author not in seen_authors:
                unique_author_id += 1
                authors[author] = unique_author_id
                seen_authors.append(author)
            if the_dir not in seen_dirs:
                unique_dir_id += 1
                dirs[the_dir] = unique_dir_id
                seen_dirs.append(the_dir)

        with open(file, 'r') as f:
            text = f.read()
            text = remove_tei_lines_from_text(text)
            temp_text.content = text
            temp_text.length = get_word_count_for_text(text)
            
            # MINIMUM WORD COUNT FILTER
            # Skip texts below the stylometric threshold
            if temp_text.length < MIN_WORD_COUNT:
                stripped_name = fix_alignment_file_names(name_of_text.split('.')[0].strip())
                skipped_files.append((temp_text.length, stripped_name))
                i += 1
                pbar.update(1)
                continue
            
            #Because the alignments file has funny ideas about filenames where Lovelace is concerned
            #I have to replace the final '-' with an '_' to match the filesystem
            #If I don't, I can't use the all_texts data with the alignments data.
            
            stripped_name_of_text = fix_alignment_file_names(name_of_text.split('.')[0].strip())
            temp_text.chapter_num = stripped_name_of_text.split('chapter_')[1]

            if text not in seen_texts:
                unique_text_id += 1
                texts[text] = unique_text_id
                temp_text.id = unique_text_id
                seen_texts.append(text)

            insert_texts_to_db(authors[author], temp_text.id, stripped_name_of_text, temp_text.content, temp_text.chapter_num, temp_text.length, dirs[the_dir], temp_text.date, short_name_for_svm) 
        i+=1
        pbar.update(1)
    pbar.close()

# Report skipped files
if skipped_files:
    print(f"\n⚠️  Skipped {len(skipped_files)} files below {MIN_WORD_COUNT} word minimum:")
    # Sort by word count to show shortest first
    for word_count, filename in sorted(skipped_files, key=lambda x: x[0]):
        print(f"  {word_count:5d} words: {filename}")
    print()

retained_count = unique_text_id
total_count = file_count
print(f"Retained {retained_count} texts ({retained_count/total_count*100:.1f}% of corpus)\n")

#Now, populate the crucial authors table
for name, id in authors.items():
    insert_authors_to_db(id, name)

#Now, populate the dirs table
for path, id in dirs.items():
    insert_dirs_to_db(id, path)

# Generate text pairs for influence analysis.
#
# IMPORTANT: Temporal ordering is enforced here by design.
# Filenames are prefixed with publication year (e.g., "1846-ENG18460—Reynolds-chapter_1").
# sorted() orders files chronologically, and itertools.combinations() preserves this order,
# so text_a (source) always precedes or equals text_b (target) in time.
# This ensures influence can only flow forward: source_year <= target_year for all pairs.
text_and_id_dict = read_all_text_names_and_ids_from_db()

# Update number_of_combinations based on retained texts
retained_files = [f for f in list_of_files 
                  if fix_alignment_file_names(f.split('/')[5].split('.')[0].strip()) in text_and_id_dict]
number_of_combinations = sum(1 for e in itertools.combinations(retained_files, 2))
print(f"Computing {number_of_combinations:,} text pairs (saved {sum(1 for e in itertools.combinations(list_of_files, 2)) - number_of_combinations:,} comparisons)\n")

transactions = []
i = 1
pbar = tqdm(desc='Computing file pairs', total=number_of_combinations, colour="#e0ffff", bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]')
for a, b in itertools.combinations(sorted(retained_files), 2):
    a = fix_alignment_file_names(a.split('/')[5])
    b = fix_alignment_file_names(b.split('/')[5])
    transactions.append((i, text_and_id_dict[a], text_and_id_dict[b]))
    i+=1
    pbar.update(1)
insert_text_pairs_to_db(transactions)
pbar.close()
