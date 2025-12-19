
# This script is basically the row or record generator for the purposes
# of the rest of the application. Goes through and for each relationship 
# (the relationships are what we're interested in, not individual texts)
# calculates the alignment and hapax overlaps. It calls the
# Jaccard calculations for similarity in database_ops for the rows, as well.

import itertools
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from database_ops import (insert_averages_to_db,
                          insert_results_to_db, insert_stats_to_db, make_reusable_dicts,
                          read_all_alignments_from_db,
                          read_all_author_names_and_ids_from_db,
                          read_all_dir_names_by_id_from_db,
                          read_all_hapax_intersects_lengths_from_db,
                          read_all_text_pair_names_and_ids_from_db,
                          read_author_from_db, read_author_names_by_id_from_db)
from util import (get_dir_lengths_for_processing, get_project_name,
                  getCountOfFiles, getListOfFiles)


# Global variables for worker processes
_worker_data = None


def init_worker(worker_data):
    """Initialize each worker process with the shared data dictionaries."""
    global _worker_data
    _worker_data = worker_data


def get_shared_aligns_count(source_id, target_id, pair_id):
    """Check for alignments between two texts."""
    found_alignments = 0
    inverted_text_pairs = _worker_data['inverted_text_pairs']
    dict_of_files_and_passages = _worker_data['dict_of_files_and_passages']
    
    if inverted_text_pairs.get((source_id, target_id)):
        try:
            align_record = dict_of_files_and_passages[pair_id]
            if align_record[1] >= 3 or align_record[3] >= 3:
                found_alignments += 1
        except KeyError:
            pass

    return found_alignments


def process_pair(args):
    """Worker function to process a single text pair."""
    pair_id, first_id, second_id = args
    
    # Unpack worker data
    text_and_id_dict = _worker_data['text_and_id_dict']
    inverted_text_and_id_dict = _worker_data['inverted_text_and_id_dict']
    chapter_lengths = _worker_data['chapter_lengths']
    length_of_corpus_text = _worker_data['length_of_corpus_text']
    dirs_dict = _worker_data['dirs_dict']
    texts_and_dirs = _worker_data['texts_and_dirs']
    authors = _worker_data['authors']
    inverted_authors = _worker_data['inverted_authors']
    the_hapax_intersects_lengths = _worker_data['the_hapax_intersects_lengths']
    author_lookup = _worker_data['author_lookup']
    
    first_name = text_and_id_dict[first_id]
    second_name = text_and_id_dict[second_id]
    
    first_year = dirs_dict[texts_and_dirs[first_name]].split('-')[0]
    second_year = dirs_dict[texts_and_dirs[second_name]].split('-')[0]
    pair_length = (chapter_lengths[first_name] + chapter_lengths[second_name])

    # Get Author Names
    first_author = authors[author_lookup[first_name]]
    second_author = authors[author_lookup[second_name]]

    # Words counted for this pair
    words_for_pair = chapter_lengths[first_name] + chapter_lengths[second_name]

    # Get alignments count
    num_alignments = get_shared_aligns_count(first_id, second_id, pair_id)
    num_alignments_over_pair_length = round((num_alignments / pair_length), 8)
    num_alignments_over_corpus_length = round((num_alignments / length_of_corpus_text), 8)

    # Get hapax intersects count
    counts_for_hapax_pair = [0]
    try:
        if the_intersects := the_hapax_intersects_lengths.get(pair_id):
            counts_for_hapax_pair.append(the_intersects)
    except KeyError:
        counts_for_hapax_pair.append(0)
    
    hapaxes_count_for_chap_pair = max(counts_for_hapax_pair, default=0)
    hapax_overlaps_over_pair_length = round((hapaxes_count_for_chap_pair / pair_length), 8)
    hapax_overlaps_over_corpus_length = round((hapaxes_count_for_chap_pair / length_of_corpus_text), 8)

    # Build transaction tuples
    first_text_id = inverted_text_and_id_dict[first_name]
    second_text_id = inverted_text_and_id_dict[second_name]
    first_author_id = inverted_authors[first_author]
    second_author_id = inverted_authors[second_author]
    first_len = chapter_lengths[first_name]
    second_len = chapter_lengths[second_name]

    stats_row = (
        first_author_id, first_year, first_text_id,
        second_author_id, second_year, second_text_id,
        hapaxes_count_for_chap_pair, hapax_overlaps_over_pair_length, hapax_overlaps_over_corpus_length,
        num_alignments, num_alignments_over_pair_length, num_alignments_over_corpus_length,
        pair_length, length_of_corpus_text, pair_id, first_len, second_len
    )

    hapax_row = (
        first_author_id, first_year, first_text_id,
        second_author_id, second_year, second_text_id,
        hapaxes_count_for_chap_pair, hapax_overlaps_over_pair_length, hapax_overlaps_over_corpus_length,
        pair_length, length_of_corpus_text, pair_id, first_len, second_len
    )

    align_row = (
        first_author_id, first_year, first_text_id,
        second_author_id, second_year, second_text_id,
        num_alignments, num_alignments_over_pair_length, num_alignments_over_corpus_length,
        pair_length, length_of_corpus_text, pair_id, first_len, second_len
    )

    # Return all results including data for insert_results_to_db
    return {
        'words_counted': words_for_pair,
        'alignments': num_alignments,
        'hapaxes': hapaxes_count_for_chap_pair,
        'stats_row': stats_row,
        'hapax_row': hapax_row,
        'align_row': align_row,
        'results_row': (first_text_id, second_text_id, hapaxes_count_for_chap_pair, num_alignments)
    }


def compute_the_averages(total_file_count, words_counted_in_comparisons, total_alignments, 
                         total_related_hapaxes):
    """Compute and display average statistics."""
    total_comparisons = total_file_count * (total_file_count - 1)
    total_words = words_counted_in_comparisons

    print("\n")
    if total_alignments == 0:
        print(f"Total Alignments Over Comparisons ({total_alignments:,} / {total_comparisons:,}): ", 0)
    elif total_alignments > 0:
        total = total_alignments / total_comparisons
        print(f"Total Alignments Over Comparisons ({total_alignments:,} / {total_comparisons:,}): {total:,}")
    
    if total_related_hapaxes == 0:
        print(f"Total Related Hapaxes Over Comparisons ({total_related_hapaxes:,} / {total_comparisons:,}): 0")
    elif total_related_hapaxes > 0:
        total = total_related_hapaxes / total_comparisons
        print(f"Total Related Hapaxes Over Comparisons ({total_related_hapaxes:,} / {total_comparisons:,}): {total:,}")
        
        total = total_related_hapaxes / total_words
        print(f"Total Related Hapaxes Over Total Words in Comparisons ({total_related_hapaxes:,} / {total_words:,}): {total:,}")
        
        # Update the last_run table for later
        insert_averages_to_db(
            total_comparisons, total_alignments, total_related_hapaxes, total_words,
            (total_alignments / total_comparisons),
            (total_related_hapaxes / total_comparisons), (total_related_hapaxes / total_words)
        )


def main():
    # Helper Vars
    project_name = get_project_name()
    list_of_files = getListOfFiles(f'./projects/{project_name}/splits')
    total_file_count = getCountOfFiles(f'./projects/{project_name}/splits')
    number_of_combinations = sum(1 for e in itertools.combinations(list_of_files, 2))

    # Load the Dictionaries for processing
    chapter_counts_dict = get_dir_lengths_for_processing()
    dict_of_files_and_passages = read_all_alignments_from_db()
    all_texts = make_reusable_dicts()
    chapter_lengths = {x['source_filename']: x['length'] for x in all_texts}
    length_of_corpus_text = sum(chapter_lengths.values())
    the_hapax_intersects_lengths = read_all_hapax_intersects_lengths_from_db()
    authors = read_author_names_by_id_from_db()
    inverted_authors = read_all_author_names_and_ids_from_db()
    text_pairs, inverted_text_pairs = read_all_text_pair_names_and_ids_from_db()
    text_and_id_dict = {x['text_id']: x['source_filename'] for x in all_texts}
    inverted_text_and_id_dict = dict(zip(text_and_id_dict.values(), text_and_id_dict.keys()))
    dirs_dict = read_all_dir_names_by_id_from_db()
    texts_and_dirs = {x['source_filename']: x['dir'] for x in all_texts}

    # Pre-compute author lookups for each text (avoiding db calls in workers)
    author_lookup = {}
    for text_name in text_and_id_dict.values():
        author_lookup[text_name] = read_author_from_db(text_name)

    # Pack all shared data for workers
    worker_data = {
        'text_and_id_dict': text_and_id_dict,
        'inverted_text_and_id_dict': inverted_text_and_id_dict,
        'chapter_lengths': chapter_lengths,
        'length_of_corpus_text': length_of_corpus_text,
        'dirs_dict': dirs_dict,
        'texts_and_dirs': texts_and_dirs,
        'authors': authors,
        'inverted_authors': inverted_authors,
        'inverted_text_pairs': inverted_text_pairs,
        'the_hapax_intersects_lengths': the_hapax_intersects_lengths,
        'dict_of_files_and_passages': dict_of_files_and_passages,
        'author_lookup': author_lookup,
    }

    # Prepare work items
    work_items = [(pair_id, item[0], item[1]) for pair_id, item in text_pairs.items()]

    # Determine optimal chunksize
    num_workers = cpu_count()
    chunksize = max(1, len(work_items) // (num_workers * 4))

    # Accumulators
    words_counted_in_comparisons = 0
    total_alignments = 0
    total_related_hapaxes = 0

    stats_transactions = []
    hapax_transactions = []
    align_transactions = []
    results_to_insert = []

    # Process in parallel
    with Pool(processes=num_workers, initializer=init_worker, initargs=(worker_data,)) as pool:
        pbar = tqdm(
            desc='Computing relationships',
            total=number_of_combinations,
            colour="#33ff33",
            bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]'
        )

        for result in pool.imap_unordered(process_pair, work_items, chunksize=chunksize):
            # Accumulate counters
            words_counted_in_comparisons += result['words_counted']
            total_alignments += result['alignments']
            total_related_hapaxes += result['hapaxes']

            # Collect transactions
            stats_transactions.append(result['stats_row'])
            hapax_transactions.append(result['hapax_row'])
            align_transactions.append(result['align_row'])
            results_to_insert.append(result['results_row'])

            pbar.update(1)

        pbar.close()

    # Insert results to db (batched for efficiency)
    print("Inserting results to database...")
    for first_id, second_id, hapax_count, align_count in tqdm(results_to_insert, desc="Inserting results"):
        insert_results_to_db(first_id, second_id, hapax_count, align_count)

    # Process transactions
    insert_stats_to_db(stats_transactions, hapax_transactions, align_transactions)

    # Now, some numbers...
    compute_the_averages(
        total_file_count, words_counted_in_comparisons, total_alignments,
        total_related_hapaxes
    )


if __name__ == '__main__':
    main()