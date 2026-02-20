# For each text file, get the list of hapaxes and store in a db. 
# This does the individual calculation on each file.

from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from database_ops import (insert_hapaxes_to_db,
                          read_all_text_names_and_ids_from_db,
                          read_text_from_db)
from hapaxes_1tM import compute_hapaxes
from tei import strip_tei


# Global variable for worker processes
_text_and_id_dict = None


def init_worker(text_and_id_dict):
    """Initialize each worker process with the shared text ID dictionary."""
    global _text_and_id_dict
    _text_and_id_dict = text_and_id_dict


def process_file(name_of_text):
    """Worker function to compute hapaxes for a single file."""
    temp_text = read_text_from_db(name_of_text)
    # DB text is already clean, but strip_tei is a safe no-op on clean text
    the_clean_data = strip_tei(temp_text)
    
    hapaxes_from_file = compute_hapaxes(the_clean_data)
    hapax_count = len(hapaxes_from_file)

    return (_text_and_id_dict[name_of_text], repr(set(hapaxes_from_file)), hapax_count)


def main():
    text_and_id_dict = read_all_text_names_and_ids_from_db()
    
    # Use texts from database, not filesystem — respects filtering
    list_of_texts = list(text_and_id_dict.keys())

    # Determine optimal chunksize
    num_workers = cpu_count()
    chunksize = max(1, len(list_of_texts) // (num_workers * 4))

    # Process in parallel
    transactions = []
    with Pool(processes=num_workers, initializer=init_worker, initargs=(text_and_id_dict,)) as pool:
        pbar = tqdm(
            desc='Computing hapaxes',
            total=len(list_of_texts),
            colour="#00875f",
            bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} | Elapsed: [{elapsed}]'
        )

        for result in pool.imap_unordered(process_file, list_of_texts, chunksize=chunksize):
            transactions.append(result)
            pbar.update(1)

        pbar.close()

    insert_hapaxes_to_db(transactions)


if __name__ == '__main__':
    main()