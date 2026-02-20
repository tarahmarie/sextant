# This script compares the hapaxes between two files and stores the overlap 
# in the project db for statistical analysis later.

import itertools

from tqdm import tqdm

from database_ops import (insert_hapax_overlaps_to_db,
                          read_all_hapaxes_from_db,
                          read_all_text_pair_names_and_ids_from_db)
from util import get_project_name, getCountOfFiles, getListOfFiles

hapaxes_dict = {}
project_name = get_project_name()
list_of_files = getListOfFiles(f'./projects/{project_name}/splits')
file_count = getCountOfFiles(f'./projects/{project_name}/splits')
text_pairs, inverted_pairs = read_all_text_pair_names_and_ids_from_db()
number_of_combinations = sum(1 for e in itertools.combinations(list_of_files, 2))
transactions = []

def make_hapax_overlaps_dict(one_id, two_id, pair_id):
    the_intersect_set = hapaxes_dict[one_id] & hapaxes_dict[two_id]
    transactions.append((pair_id, repr(the_intersect_set), len(the_intersect_set)))

#Fetch the hapaxes and store them in a working dict.
hapaxes_dict = read_all_hapaxes_from_db()

pbar = tqdm(desc='Computing hapax overlaps', total=number_of_combinations, colour="#ffaf87", bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]')
for id, item in text_pairs.items():
    make_hapax_overlaps_dict(item[0], item[1], id)
    pbar.update(1)
pbar.close()

#Now, insert the transactions:
insert_hapax_overlaps_to_db(transactions)