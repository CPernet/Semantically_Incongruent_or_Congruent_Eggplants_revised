"""
Next-word prediction and word surprisal utilities.

Uses a causal language model (GPT-2 by default) from the Hugging Face
`transformers` library to compute *surprisal* – the negative log-probability
(in bits) of each token given the preceding context.

  surprisal(w_i | w_1..w_{i-1})  =  -log2 P(w_i | w_1..w_{i-1})

High surprisal means the word was unexpected; low surprisal means it was
highly predictable.  For N400/ERP research this value closely tracks the
classical *cloze probability* measure (but is derived from a language model
rather than human completion norms).

Usage example
-------------
>>> from surprisal import SurprisalModel
>>> model = SurprisalModel()
>>> model.sentence_surprisal("The cat sat on the mat")
[..., ...]  # list of (token, surprisal_bits) tuples
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SurprisalModel:
    """Wrapper around a causal LM for computing per-token surprisal.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (default ``"gpt2"``).
    device:
        Torch device string.  Defaults to ``"cuda"`` when a GPU is available,
        otherwise ``"cpu"``.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def token_surprisals(self, sentence: str) -> list[tuple[str, float]]:
        """Return per-token surprisal values for *sentence*.

        The first token is assigned ``None`` because there is no prior
        context to condition on (the model cannot produce a meaningful
        probability for it).

        Parameters
        ----------
        sentence:
            A complete sentence string.

        Returns
        -------
        list[tuple[str, float | None]]
            Each element is ``(token_string, surprisal_bits)``.  The first
            token has ``surprisal = None``.
        """
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab)

        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        results: list[tuple[str, float | None]] = [(tokens[0], None)]

        for i in range(1, len(tokens)):
            token_id = input_ids[0, i].item()
            log_p = log_probs[i - 1, token_id].item()
            surprisal_bits = -log_p / math.log(2)
            results.append((tokens[i], surprisal_bits))

        return results

    def word_surprisal(self, context: str, target_word: str) -> float:
        """Return the surprisal of *target_word* given *context*.

        The target word is appended to the context and the surprisal is
        summed over all sub-word tokens that make up the target word.
        This is useful for computing the surprisal of a specific word
        (e.g. the critical word in an EEG sentence) given the preceding
        sentence fragment.

        Parameters
        ----------
        context:
            The preceding sentence fragment (not including the target word).
        target_word:
            The word whose surprisal is to be computed.

        Returns
        -------
        float
            Total surprisal (in bits) of *target_word* given *context*.
        """
        full = context.rstrip() + " " + target_word.strip()
        context_ids = self.tokenizer.encode(context, return_tensors="pt").to(
            self.device
        )
        full_ids = self.tokenizer.encode(full, return_tensors="pt").to(self.device)

        n_context = context_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(full_ids, labels=full_ids)
            logits = outputs.logits  # (1, seq_len, vocab)

        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)

        surprisal_total = 0.0
        for i in range(n_context, full_ids.shape[1]):
            token_id = full_ids[0, i].item()
            log_p = log_probs[i - 1, token_id].item()
            surprisal_total += -log_p / math.log(2)

        return surprisal_total

    def sentence_surprisal(self, sentence: str) -> float:
        """Return the mean per-token surprisal for *sentence*.

        The first token is excluded from the average because it has no
        prior context.

        Parameters
        ----------
        sentence:
            A complete sentence string.

        Returns
        -------
        float
            Mean surprisal in bits per token (excluding the first token).
        """
        token_surps = self.token_surprisals(sentence)
        values = [s for _, s in token_surps if s is not None]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def sentence_perplexity(self, sentence: str) -> float:
        """Return the perplexity of *sentence* under the model.

        Parameters
        ----------
        sentence:
            A complete sentence string.

        Returns
        -------
        float
            Perplexity (2 ** mean-surprisal).
        """
        return 2 ** self.sentence_surprisal(sentence)
