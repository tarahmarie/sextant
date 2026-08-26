"""Language profiles: the single home for language-dependent text processing.

Every tokenisation, stopword, lemmatisation, and character-class decision in
the pipeline is defined here, once per language. Three code paths consume a
profile:

  P1  extract_hapaxes.py   close-reading hapax tokeniser   -> kwic_tokenize()
  P2  hapaxes_1tM.py       pipeline hapax path             -> hapax_tokenize()
  P3  do_svm.py            SVM preprocessing               -> svm_clean()

plus do_svm's novel-key normalisation -> normalise_novel_key().

Language is selected PER PROJECT, not per run: a corpus has a language, and a
per-run switch could write French tokens into an English project's database.
The selection file is `projects/<name>/language`. Its first line is the
profile code ("en" or "fr"); later lines may override named parameters, e.g.

    fr
    lemmatiser=simplemma

A missing file means "en", so every existing project reproduces its historical
behaviour without being touched.

--------------------------------------------------------------------------
The "en" profile: frozen thesis behaviour
--------------------------------------------------------------------------
The English profile reproduces the pipeline's pre-refactor behaviour EXACTLY,
quirks included, because the English outputs are thesis results under
examination. Preserved quirks, deliberately not fixed:

  - P1 regex `[a-zA-ZÀ-ÿ]+(?:'[a-zA-Z]+)?`: trailing-apostrophe
    clause only (English contractions); the character range admits the
    multiplication and division signs and stops before oe/ae ligatures.
  - P2 strips string.punctuation from INSIDE tokens, gluing "l'homme" into
    "lhomme" and "peut-etre" into "peutetre".
  - P2 counts frequencies case-sensitively and lowercases afterwards, so a
    word capitalised once and lowercase many times can still yield a hapax.
  - P3 deletes non-[a-zA-Z0-9 À-ÿ] characters with "" (gluing),
    removes NLTK English stopwords, then applies the WordNet lemmatiser with
    no POS argument (noun default) AFTER stopword removal - a shallow plural
    strip, not real lemmatisation.
  - Novel keys are NFKD-normalised without dropping combining characters
    (a no-op for the all-ASCII English keys).

--------------------------------------------------------------------------
The "fr" profile: decisions and reasoning
--------------------------------------------------------------------------
Elision. French elision is leading (l'escabeau, qu'elle), so apostrophes are
split, not stripped. A token is split at its first apostrophe only when the
prefix is one of the closed clitic set {l, d, j, n, m, t, s, c, qu, jusqu,
lorsqu, puisqu, quoiqu}; BOTH halves are kept as tokens. Keeping the clitic
keeps the tokeniser a pure tokeniser: clitics are so frequent they never
surface as hapaxes, and the SVM path removes them via the stopword list
(NLTK's French list already contains the bare forms c, d, j, l, m, n, s, t,
qu). Lexicalised forms (aujourd'hui, presqu'ile family, quelqu'un family,
prud'homme family) are kept whole. Non-clitic apostrophes (dialect ch'aime,
names like O'Brien) are kept whole too - dialect rendering is itself a
stylistic signal.

Ligatures. oe (U+0153) and ae (U+00E6) are mapped to their two-letter forms.
This CANNOT be done with Unicode normalisation: U+0153 has no decomposition
mapping in any normal form. The mapping is required because the corpus mixes
"coeur" and the ligatured spelling across novels, which would otherwise mint
encoding-driven word types.

Accents. Preserved everywhere (ou/où, la/là, sur/sûr are distinct words).
No ascii folding appears anywhere in this profile.

Hyphens. Interior hyphens are preserved (peut-être, lui-même) and edge
punctuation is trimmed, instead of the English delete-from-inside behaviour.

Stopwords. nltk.corpus.stopwords.words("french") (157 words), the exact
structural analogue of the English pass's NLTK list, plus the clitic forms
our splitter emits that the list lacks (jusqu, lorsqu, puisqu, quoiqu).
Larger curated lists would do more work than the English pass did and
reintroduce asymmetry.

Lemmatisation. OFF by default. The English pass is not a morphological
reduction: WordNet with no POS argument, run after stopword removal, is a
shallow plural strip on content words. A real French lemmatiser would
collapse verb conjugation and adjective agreement and remove far more
variance, so calling both "lemmatisation" would create exactly the asymmetry
the design avoids; not lemmatising is the smaller deviation and preserves the
inflectional variation the SVM reads as style. The choice is exposed as the
named parameter `lemmatiser` ("none" | "simplemma") so it is visible in
config and can be run both ways; simplemma (dictionary lookup, no model) is
the only wireable option, chosen over neural pipelines for the method's
carbon-footprint claim.

Novel keys. NFC, because NFKD-decomposed keys silently fail to join against
the NFC/filesystem-form keys used elsewhere once titles contain accents.
"""

import os
import re
import string

from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Shared French machinery
# ---------------------------------------------------------------------------

# Curly/modifier apostrophes normalised to ASCII before any apostrophe logic.
_APOSTROPHES = str.maketrans({'’': "'", 'ʼ': "'"})

# U+0153/U+0152/U+00E6/U+00C6 have no Unicode decomposition; explicit map.
_LIGATURES = str.maketrans({'œ': 'oe', 'Œ': 'Oe',
                            'æ': 'ae', 'Æ': 'Ae'})

# Closed set of elided clitic prefixes.
_FR_CLITICS = frozenset({
    'l', 'd', 'j', 'n', 'm', 't', 's', 'c', 'qu',
    'jusqu', 'lorsqu', 'puisqu', 'quoiqu',
})

# Lexicalised apostrophe words, kept whole (never split).
_FR_ELISION_KEEP = frozenset({
    "aujourd'hui",
    "presqu'ile", "presqu'iles", "presqu'île", "presqu'îles",
    "quelqu'un", "quelqu'une", "quelqu'uns", "quelqu'unes",
    "prud'homme", "prud'hommes", "prud'homie", "prud'homal",
})

# French letters for word regexes: ASCII + Latin-1 letters (excluding the
# multiplication/division signs U+00D7 and U+00F7) + oe/ae ligatures.
_FR_LETTERS = 'a-zA-ZÀ-ÖØ-öø-ÿŒœÆæ'
_FR_WORD_RE = re.compile(f"[{_FR_LETTERS}]+(?:['’][{_FR_LETTERS}]+)*")

# Edge punctuation trimmed from P2 tokens (string.punctuation misses these).
_FR_EDGE_PUNCT = string.punctuation + '«»–—‘’“”…'


def _split_elisions(token):
    """Split leading clitic elisions: "qu'elle" -> ["qu", "elle"].

    Splits only when the prefix is a known clitic; lexicalised forms and
    unknown apostrophe words (dialect, foreign names) are kept whole.
    """
    if "'" not in token:
        return [token]
    if token.lower() in _FR_ELISION_KEEP:
        return [token]
    head, _, rest = token.partition("'")
    if rest and head.lower() in _FR_CLITICS:
        return [head] + _split_elisions(rest)
    return [token]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class EnglishProfile:
    """Frozen pre-refactor English behaviour. Do not "fix" anything here:
    equivalence_harness.py asserts byte-identity with the thesis pipeline."""

    code = 'en'
    textpair_language = 'english'

    def __init__(self, lemmatiser='wordnet-noun-default'):
        if lemmatiser not in ('wordnet-noun-default', 'none'):
            raise ValueError(f'unknown lemmatiser for en: {lemmatiser}')
        self.lemmatiser = lemmatiser
        self.stopwords = set(nltk_stopwords.words('english'))
        self._punct_table = str.maketrans('', '', string.punctuation)
        self._kwic_re = re.compile(r"[a-zA-ZÀ-ÿ]+(?:'[a-zA-Z]+)?")
        self._svm_strip_re = re.compile(r"[^a-zA-Z0-9\sÀ-ÿ]")
        self._wordnet = None

    def kwic_tokenize(self, text):
        """P1: (token, char_offset) pairs, lowercased."""
        return [(m.group().lower(), m.start()) for m in self._kwic_re.finditer(text)]

    def hapax_tokenize(self, rawtext):
        """P2: English punkt tokenisation, punctuation stripped from inside
        each token (historical gluing behaviour), case preserved."""
        words = word_tokenize(rawtext)
        return [word.translate(self._punct_table) for word in words]

    def svm_clean(self, text):
        """P3: expects TEI-stripped text; lowercase, strip characters,
        tokenise, remove stopwords, then shallow WordNet lemmatise."""
        text = text.lower()
        text = self._svm_strip_re.sub('', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stopwords]
        if self.lemmatiser == 'wordnet-noun-default':
            if self._wordnet is None:
                from nltk.stem import WordNetLemmatizer
                self._wordnet = WordNetLemmatizer()
            tokens = [self._wordnet.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def normalise_novel_key(self, key):
        import unicodedata
        return unicodedata.normalize('NFKD', key)


class FrenchProfile:
    """French profile. See the module docstring for the reasoning behind
    elision, ligature, stopword, and lemmatisation decisions."""

    code = 'fr'
    textpair_language = 'french'

    def __init__(self, lemmatiser='none'):
        if lemmatiser not in ('none', 'simplemma'):
            raise ValueError(f'unknown lemmatiser for fr: {lemmatiser}')
        self.lemmatiser = lemmatiser
        self.stopwords = set(nltk_stopwords.words('french')) | set(_FR_CLITICS)
        self._svm_strip_re = re.compile(f"[^0-9\\s'{_FR_LETTERS}]")
        self._simplemma = None

    def _normalise(self, text):
        return text.translate(_APOSTROPHES).translate(_LIGATURES)

    def kwic_tokenize(self, text):
        """P1: (token, char_offset) pairs, lowercased, elisions split.
        Subtokens of a split word share the word's start offset (offsets are
        only used for display ordering)."""
        tokens = []
        for m in _FR_WORD_RE.finditer(text):
            word = self._normalise(m.group())
            for part in _split_elisions(word):
                tokens.append((part.lower(), m.start()))
        return tokens

    def hapax_tokenize(self, rawtext):
        """P2: French punkt tokenisation, elisions split, edge punctuation
        trimmed (interior hyphens/apostrophes preserved), case preserved."""
        out = []
        for word in word_tokenize(rawtext, language='french'):
            word = self._normalise(word)
            for part in _split_elisions(word):
                part = part.strip(_FR_EDGE_PUNCT)
                if part:
                    out.append(part)
        return out

    def _lemmatise(self, tokens):
        if self._simplemma is None:
            try:
                import simplemma
            except ImportError as e:
                raise ImportError(
                    'The fr profile was configured with lemmatiser=simplemma '
                    'but simplemma is not installed (pip install simplemma).'
                ) from e
            self._simplemma = simplemma
        return [self._simplemma.lemmatize(t, lang='fr') for t in tokens]

    def svm_clean(self, text):
        """P3: expects TEI-stripped text; lowercase, normalise apostrophes
        and ligatures, replace non-word characters with spaces (no gluing),
        tokenise, split elisions, remove French stopwords, optionally
        lemmatise via simplemma."""
        text = self._normalise(text.lower())
        text = self._svm_strip_re.sub(' ', text)
        tokens = word_tokenize(text, language='french')
        tokens = [part.strip("'")
                  for token in tokens
                  for part in _split_elisions(token)]
        tokens = [t for t in tokens if t and t not in self.stopwords]
        if self.lemmatiser == 'simplemma':
            tokens = self._lemmatise(tokens)
        return ' '.join(tokens)

    def normalise_novel_key(self, key):
        import unicodedata
        return unicodedata.normalize('NFC', key)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

_PROFILE_CLASSES = {'en': EnglishProfile, 'fr': FrenchProfile}
_cache = {}


def read_project_language_config(project_name):
    """Read projects/<name>/language. Returns (code, overrides dict).
    A missing file means ("en", {}): existing projects stay English."""
    path = f'./projects/{project_name}/language'
    if not os.path.exists(path):
        return 'en', {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    code = lines[0]
    if code not in _PROFILE_CLASSES:
        raise ValueError(f'{path}: unknown language {code!r} '
                         f'(expected one of {sorted(_PROFILE_CLASSES)})')
    overrides = {}
    for line in lines[1:]:
        key, _, value = line.partition('=')
        if key.strip() != 'lemmatiser' or not value:
            raise ValueError(f'{path}: cannot parse override line {line!r}')
        overrides['lemmatiser'] = value.strip()
    return code, overrides


def get_profile(code=None, **overrides):
    """Return the profile for `code`, or for the current project when
    `code` is None (reading ./.current_project like the rest of the
    pipeline, then projects/<name>/language)."""
    if code is None:
        from util import get_project_name
        code, file_overrides = read_project_language_config(get_project_name())
        file_overrides.update(overrides)
        overrides = file_overrides
    key = (code, tuple(sorted(overrides.items())))
    if key not in _cache:
        _cache[key] = _PROFILE_CLASSES[code](**overrides)
    return _cache[key]
