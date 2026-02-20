from functools import lru_cache
import os
import re
from dataclasses import dataclass

SKIP_EXTENSIONS = {'.npy', '.compressed', '.DS_Store'}


def get_algnments_file_name():
    with open("./.alignments_file_name", "r") as alignments_file_name:
        return alignments_file_name.readline().strip()

def get_project_name():
    with open('./.current_project', 'r') as current_project_file:
        return current_project_file.readline().strip()

@lru_cache
def getListOfFiles(dirName):
    filelist = []
    for root, dirs, files in os.walk(dirName):
        for file in files:
            if any(file.endswith(ext) for ext in SKIP_EXTENSIONS):
                continue
            filelist.append(os.path.join(root, file))
    return filelist

def getCountOfFiles(dirName):
    return len(getListOfFiles(dirName))

def get_dir_lengths_for_processing():
    project_name = get_project_name()
    counts_dict = {}
    for root, dirs, files in os.walk(f'./projects/{project_name}/splits/'):
        for dir in dirs:
            counts_dict[dir] = len(os.listdir(f'./projects/{project_name}/splits/{dir}'))
    return counts_dict

def get_word_count_for_text(text):
    """Count words in already-clean text."""
    return len(text.split())

def fix_the_author_name_from_aligns(name):
    """Clean author names from TextPAIR alignment results."""
    if '(' in name:
        name = name.split('(')[0].strip()
    name = name.strip().replace('\n', '')
    name = name.split(',')[0]
    name = name.replace('-', '_')
    return name

def fix_alignment_file_names(name):
    name = name.replace('.txt', '')
    if 'lovelace-transcription' in name:
        target = '-'
        payload = '_'
        return payload.join(name.rsplit(target, 1))
    else:
        return name

def create_author_pair_for_lookups(author_a, author_b):
    ordered_already = str(author_a) + " " + str(author_b)
    disorded_already = str(author_b) + " " + str(author_a)
    if int(author_a) <= int(author_b):
        return ordered_already
    elif int(author_a) > int(author_b):
        return disorded_already