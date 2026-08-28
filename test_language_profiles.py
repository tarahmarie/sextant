#!/usr/bin/env python3
"""Unit tests for language_profiles.py. Run: python3 test_language_profiles.py

The heavyweight English no-op proof lives in equivalence_harness.py (it
asserts byte-identity over every real split file). These tests cover the
French profile's documented decisions and a spot-check that the English
profile matches an independent copy of the historical code.
"""

import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from language_profiles import get_profile, _split_elisions

passed = 0


def check(name, condition):
    global passed
    assert condition, f'FAIL: {name}'
    passed += 1
    print(f'  ok: {name}')


fr = get_profile('fr')
en = get_profile('en')

print('French elision:')
check("l'escabeau splits", _split_elisions("l'escabeau") == ['l', 'escabeau'])
check("qu'elle splits", _split_elisions("qu'elle") == ['qu', 'elle'])
check("jusqu'à splits", _split_elisions("jusqu'à") == ['jusqu', 'à'])
check("d'une splits", _split_elisions("d'une") == ['d', 'une'])
check("chained d'l'autre splits fully", _split_elisions("d'l'autre") == ['d', 'l', 'autre'])
check("aujourd'hui kept whole", _split_elisions("aujourd'hui") == ["aujourd'hui"])
check("quelqu'un kept whole", _split_elisions("quelqu'un") == ["quelqu'un"])
check("dialect ch'aime kept whole", _split_elisions("ch'aime") == ["ch'aime"])
check("O'Brien kept whole", _split_elisions("O'Brien") == ["O'Brien"])
check("capitalised L'Homme splits", _split_elisions("L'Homme") == ['L', 'Homme'])

print('French ligatures and accents:')
toks = [t for t, _ in fr.kwic_tokenize('le cœur et l’œuvre de ma sœur')]
check('œ maps to oe', toks == ['le', 'coeur', 'et', 'l', 'oeuvre', 'de', 'ma', 'soeur'])
toks = [t for t, _ in fr.kwic_tokenize('où est la forêt déjà brûlée')]
check('accents preserved', toks == ['où', 'est', 'la', 'forêt', 'déjà', 'brûlée'])
check('multiplication sign not a letter', [t for t, _ in fr.kwic_tokenize('3×4')] == [])

print('French P2 hapax tokeniser:')
toks = fr.hapax_tokenize("« L’escabeau, dit-elle, — c’est peut-être moi. »")
check('elision split, guillemets dropped, hyphens kept',
      toks == ['L', 'escabeau', 'dit-elle', 'c', 'est', 'peut-être', 'moi'])
check('case preserved for the counting quirk', toks[0] == 'L')

print('French P3 SVM path:')
out = fr.svm_clean("Qu'elle avait vu l'homme, jusqu'à la forêt !")
check('clitics and stopwords removed, accents kept, no gluing',
      out == 'vu homme forêt')
out = fr.svm_clean('les temps et les fois passèrent')
check('no lemmatisation by default (inflection preserved)',
      out == 'temps fois passèrent')
check('lemmatiser parameter is visible', fr.lemmatiser == 'none')

print('English profile matches historical code (spot check):')
SAMPLE = ("It was the best of times, it was the worst of times; the ladies' "
          "societies had 3 × 4 cœurs and mustn't-grumble airs.")
# Independent verbatim copy of the pre-refactor do_svm.py implementation:
_stop = set(stopwords.words('english'))
_lem = WordNetLemmatizer()
_old = re.sub(r"[^a-zA-Z0-9\sÀ-ÿ]", "", SAMPLE.lower())
_old_tokens = [t for t in word_tokenize(_old) if t not in _stop]
_old_out = ' '.join(_lem.lemmatize(t) for t in _old_tokens)
check('svm_clean identical', en.svm_clean(SAMPLE) == _old_out)
# Pre-refactor extract_hapaxes.py tokenizer:
_old_kwic = [(m.group().lower(), m.start()) for m in
             re.finditer(r"[a-zA-ZÀ-ÿ]+(?:'[a-zA-Z]+)?", SAMPLE)]
check('kwic_tokenize identical', en.kwic_tokenize(SAMPLE) == _old_kwic)
# Pre-refactor hapaxes_1tM.py tokenizer:
_table = str.maketrans('', '', string.punctuation)
_old_hap = [w.translate(_table) for w in word_tokenize(SAMPLE)]
check('hapax_tokenize identical', en.hapax_tokenize(SAMPLE) == _old_hap)

print('Novel keys:')
import unicodedata
_nfc = unicodedata.normalize('NFC', 'FRA00501\u2014S\u00e9gur')
_nfd = unicodedata.normalize('NFD', _nfc)
check('en uses NFKD (historical behaviour, decomposes accents)',
      en.normalise_novel_key(_nfc) == unicodedata.normalize('NFKD', _nfc)
      and en.normalise_novel_key(_nfc) != _nfc)
check('fr uses NFC (composes filesystem NFD forms back)',
      fr.normalise_novel_key(_nfd) == _nfc)

print(f'\nAll {passed} checks passed.')
