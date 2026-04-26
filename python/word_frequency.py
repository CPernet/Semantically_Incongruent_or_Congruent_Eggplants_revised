"""
Word lexical frequency utilities.

Uses the `wordfreq` package to retrieve word frequency norms.  Frequencies
are returned on the Zipf scale (Zipf = log10(freq per billion words) + 3),
where values above ~3 are common and values below ~1 are very rare.

Reference: Van Heuven et al. (2014). SUBTLEX-UK.
"""

import wordfreq


def get_zipf_frequency(word: str, lang: str = "en") -> float:
    """Return the Zipf-scale frequency of *word* in language *lang*.

    Parameters
    ----------
    word:
        The word to look up.
    lang:
        BCP 47 language code (default ``"en"`` for English).

    Returns
    -------
    float
        Zipf frequency (0.0 if the word is unknown to the database).
    """
    return wordfreq.zipf_frequency(word, lang)


def get_word_frequency(word: str, lang: str = "en") -> float:
    """Return the raw frequency (proportion of tokens) of *word*.

    Parameters
    ----------
    word:
        The word to look up.
    lang:
        BCP 47 language code (default ``"en"`` for English).

    Returns
    -------
    float
        Estimated frequency as a proportion (0.0 if unknown).
    """
    return wordfreq.word_frequency(word, lang)


def get_frequency_for_words(words: list[str], lang: str = "en") -> list[dict]:
    """Return a list of frequency records for each word in *words*.

    Parameters
    ----------
    words:
        Sequence of words to look up.
    lang:
        BCP 47 language code.

    Returns
    -------
    list[dict]
        Each element contains ``word``, ``zipf_frequency``, and
        ``raw_frequency`` keys.
    """
    results = []
    for word in words:
        results.append(
            {
                "word": word,
                "zipf_frequency": get_zipf_frequency(word, lang),
                "raw_frequency": get_word_frequency(word, lang),
            }
        )
    return results
