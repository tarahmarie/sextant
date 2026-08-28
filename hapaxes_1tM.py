# imports
from collections import Counter

from language_profiles import get_profile

# Characters that string.punctuation misses
BAD_TOKENS = frozenset({
    '', '\u2014', '\u2013', '\u2018', '\u2019',
    '\u201c', '\u201d', '\u2026', '\u00ab', '\u00bb'
})


def compute_hapaxes(rawtext, profile=None):
    # Tokenisation is language-dependent and lives in language_profiles.py;
    # the profile comes from the current project unless passed explicitly.
    if profile is None:
        profile = get_profile()
    words = profile.hapax_tokenize(rawtext)

    # Count the frequency of each word using a dictionary-based counter
    freq = Counter(words)

    # Find the hapaxes (words that occur only once), filtering bad tokens at construction
    hapaxes = [word.lower() for word in freq if freq[word] == 1 and word not in BAD_TOKENS]

    return hapaxes