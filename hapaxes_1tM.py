# imports
import string
from collections import Counter

from nltk.tokenize import word_tokenize

# Characters that string.punctuation misses
BAD_TOKENS = frozenset({
    '', '\u2014', '\u2013', '\u2018', '\u2019',
    '\u201c', '\u201d', '\u2026', '\u00ab', '\u00bb'
})


def compute_hapaxes(rawtext):
    words = word_tokenize(rawtext)

    # Remove punctuation from the words
    table = str.maketrans('', '', string.punctuation)
    words = [word.translate(table) for word in words]
    # Count the frequency of each word using a dictionary-based counter
    freq = Counter(words)

    # Find the hapaxes (words that occur only once), filtering bad tokens at construction
    hapaxes = [word.lower() for word in freq if freq[word] == 1 and word not in BAD_TOKENS]

    return hapaxes