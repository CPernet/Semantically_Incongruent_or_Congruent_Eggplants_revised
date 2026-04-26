"""
Phonological property utilities.

Uses the `pronouncing` package, which is built on the CMU Pronouncing
Dictionary, to extract phonological characteristics of English words.

Metrics extracted
-----------------
phonemes        : ARPABET phoneme string (first pronunciation entry)
n_phonemes      : total number of phonemes
n_syllables     : number of syllables (counted from stressed vowel nuclei)
onset_phoneme   : first phoneme (consonant or vowel onset)
"""

import re
from typing import Optional

import pronouncing


def get_phones(word: str) -> Optional[str]:
    """Return the ARPABET phone string for the first pronunciation of *word*.

    Parameters
    ----------
    word:
        The English word to look up.

    Returns
    -------
    str or None
        Space-separated ARPABET phones, or ``None`` if the word is not in
        the CMU Pronouncing Dictionary.
    """
    phones_list = pronouncing.phones_for_word(word.lower())
    if phones_list:
        return phones_list[0]
    return None


def get_num_phonemes(word: str) -> Optional[int]:
    """Return the number of phonemes in *word*.

    Parameters
    ----------
    word:
        The English word to look up.

    Returns
    -------
    int or None
        Number of phonemes, or ``None`` if the word is not found.
    """
    phones = get_phones(word)
    if phones is None:
        return None
    return len(phones.split())


def get_num_syllables(word: str) -> Optional[int]:
    """Return the number of syllables in *word*.

    Parameters
    ----------
    word:
        The English word to look up.

    Returns
    -------
    int or None
        Syllable count, or ``None`` if the word is not found.
    """
    phones = get_phones(word)
    if phones is None:
        return None
    return pronouncing.syllable_count(phones)


def get_onset_phoneme(word: str) -> Optional[str]:
    """Return the first (onset) phoneme of *word*.

    The onset is the phoneme that begins the pronunciation; stress digits
    are stripped from the returned token so that vowel-initial words return
    the bare vowel label.

    Parameters
    ----------
    word:
        The English word to look up.

    Returns
    -------
    str or None
        Onset phoneme without stress digit, or ``None`` if not found.
    """
    phones = get_phones(word)
    if phones is None:
        return None
    first = phones.split()[0]
    return re.sub(r"\d", "", first)


def get_phonology_for_word(word: str) -> dict:
    """Return a dictionary of all phonological metrics for *word*.

    Parameters
    ----------
    word:
        The English word to look up.

    Returns
    -------
    dict
        Keys: ``word``, ``phonemes``, ``n_phonemes``, ``n_syllables``,
        ``onset_phoneme``.
    """
    phones = get_phones(word)
    n_phonemes = len(phones.split()) if phones else None
    n_syllables = pronouncing.syllable_count(phones) if phones else None
    onset = None
    if phones:
        onset = re.sub(r"\d", "", phones.split()[0])
    return {
        "word": word,
        "phonemes": phones,
        "n_phonemes": n_phonemes,
        "n_syllables": n_syllables,
        "onset_phoneme": onset,
    }


def get_phonology_for_words(words: list[str]) -> list[dict]:
    """Return phonological metrics for each word in *words*.

    Parameters
    ----------
    words:
        Sequence of English words.

    Returns
    -------
    list[dict]
        One record per word with keys ``word``, ``phonemes``,
        ``n_phonemes``, ``n_syllables``, ``onset_phoneme``.
    """
    return [get_phonology_for_word(w) for w in words]
