"""
Main stimuli processing pipeline.

Reads a stimuli CSV file, computes all linguistic metrics for each
sentence/word, and writes a CSV with added metric columns.

Expected input CSV columns
--------------------------
At minimum the CSV must contain one of the following:

* ``sentence``      – the full sentence string
* ``critical_word`` – (optional) the word of primary interest (e.g. the
                      sentence-final or critical word)
* ``context``       – (optional) sentence fragment preceding the critical
                      word; if absent it is derived automatically

All other columns in the input are preserved unchanged in the output.

Output columns added
--------------------
Word-level metrics (applied to *critical_word* when present):
  cw_zipf_frequency         Zipf-scale lexical frequency
  cw_raw_frequency          Raw corpus frequency (proportion of tokens)
  cw_phonemes               ARPABET phoneme string
  cw_n_phonemes             Number of phonemes
  cw_n_syllables            Number of syllables
  cw_onset_phoneme          First (onset) phoneme
  cw_surprisal_bits         Surprisal of critical word given context (bits)
  cw_n_letters              Number of letters

Sentence-level metrics:
  sent_mean_surprisal       Mean token surprisal across the sentence (bits)
  sent_perplexity           Sentence perplexity (2 ** mean_surprisal)
  sent_n_tokens             Number of sub-word tokens
  sent_context_target_sim   Cosine similarity: sentence context vs. critical
                            word embedding (semantic congruency proxy)

Usage
-----
python process_stimuli.py stimuli.csv --output stimuli_metrics.csv

The script will download the GPT-2 and BERT model weights on first run
(~500 MB) and cache them in the default HuggingFace cache directory.
"""

import argparse
import logging
import sys

import pandas as pd

from phonology import get_phonology_for_word
from sentence_metrics import SentenceMetrics
from surprisal import SurprisalModel
from word_frequency import get_zipf_frequency, get_word_frequency

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_context(row: pd.Series) -> str:
    """Derive context string from a DataFrame row.

    If the row has a non-empty ``context`` column, use that.
    Otherwise split the sentence and remove the critical word from the end.
    """
    if "context" in row.index and pd.notna(row["context"]) and str(row["context"]).strip():
        return str(row["context"]).strip()

    sentence = str(row.get("sentence", "")).strip()
    critical = str(row.get("critical_word", "")).strip().lower()

    if not critical or not sentence:
        return sentence

    words = sentence.split()
    # Walk backwards to find the critical word and return everything before it
    for i in range(len(words) - 1, -1, -1):
        if words[i].strip(".,;:!?\"'").lower() == critical:
            return " ".join(words[:i])
    return sentence


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------


def process_stimuli(
    df: pd.DataFrame,
    surprisal_model_name: str = "gpt2",
    bert_model_name: str = "bert-base-uncased",
    device: str | None = None,
) -> pd.DataFrame:
    """Compute all linguistic metrics for each row in *df*.

    Parameters
    ----------
    df:
        Input DataFrame with at least a ``sentence`` column.
    surprisal_model_name:
        HuggingFace causal LM model name (for surprisal).
    bert_model_name:
        HuggingFace masked LM model name (for sentence embeddings).
    device:
        Torch device (``"cpu"`` or ``"cuda"``); auto-detected if ``None``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional metric columns appended.
    """
    log.info("Loading surprisal model: %s", surprisal_model_name)
    surp_model = SurprisalModel(model_name=surprisal_model_name, device=device)

    log.info("Loading sentence metrics model: %s", bert_model_name)
    sent_model = SentenceMetrics(model_name=bert_model_name, device=device)

    out = df.copy()

    has_critical_word = "critical_word" in df.columns

    for idx, row in df.iterrows():
        sentence = str(row.get("sentence", "")).strip()
        if not sentence:
            log.warning("Row %d has no sentence; skipping.", idx)
            continue

        # ---- sentence-level metrics ----
        log.info("[%d/%d] Sentence surprisal: %s", idx + 1, len(df), sentence[:60])
        mean_surp = surp_model.sentence_surprisal(sentence)
        out.at[idx, "sent_mean_surprisal"] = mean_surp
        out.at[idx, "sent_perplexity"] = 2**mean_surp

        # Token count (sub-word tokens)
        n_tok = len(surp_model.tokenizer.tokenize(sentence))
        out.at[idx, "sent_n_tokens"] = n_tok

        # ---- critical-word metrics ----
        if has_critical_word:
            critical = str(row.get("critical_word", "")).strip()
            if not critical:
                continue

            # Lexical frequency
            out.at[idx, "cw_zipf_frequency"] = get_zipf_frequency(critical)
            out.at[idx, "cw_raw_frequency"] = get_word_frequency(critical)
            out.at[idx, "cw_n_letters"] = len(critical)

            # Phonology
            phon = get_phonology_for_word(critical)
            out.at[idx, "cw_phonemes"] = phon["phonemes"]
            out.at[idx, "cw_n_phonemes"] = phon["n_phonemes"]
            out.at[idx, "cw_n_syllables"] = phon["n_syllables"]
            out.at[idx, "cw_onset_phoneme"] = phon["onset_phoneme"]

            # Surprisal of critical word given context
            context = _get_context(row)
            if context:
                cw_surp = surp_model.word_surprisal(context, critical)
                out.at[idx, "cw_surprisal_bits"] = cw_surp
            else:
                out.at[idx, "cw_surprisal_bits"] = None

            # Sentence-level context-target similarity
            s_metrics = sent_model.sentence_metrics(sentence, critical)
            out.at[idx, "sent_context_target_sim"] = s_metrics.get(
                "context_target_similarity"
            )

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute linguistic metrics for EEG stimuli.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Path to the input stimuli CSV file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the output CSV file (default: input_metrics.csv).",
    )
    parser.add_argument(
        "--surprisal-model",
        default="gpt2",
        help="HuggingFace causal LM for surprisal (default: gpt2).",
    )
    parser.add_argument(
        "--bert-model",
        default="bert-base-uncased",
        help="HuggingFace masked LM for sentence embeddings "
             "(default: bert-base-uncased).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. 'cpu' or 'cuda' (auto-detected by default).",
    )
    args = parser.parse_args(argv)

    output_path = args.output
    if output_path is None:
        stem = args.input.removesuffix(".csv")
        output_path = stem + "_metrics.csv"

    log.info("Reading stimuli from: %s", args.input)
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    log.info("Stimuli loaded: %d rows, columns: %s", len(df), list(df.columns))

    result = process_stimuli(
        df,
        surprisal_model_name=args.surprisal_model,
        bert_model_name=args.bert_model,
        device=args.device,
    )

    result.to_csv(output_path, index=False)
    log.info("Metrics written to: %s", output_path)


if __name__ == "__main__":
    main()
