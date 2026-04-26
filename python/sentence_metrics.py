"""
Sentence-level metrics utilities.

Computes sentence-level representations and similarity scores using
masked language models (BERT) via the Hugging Face `transformers` library.

Metrics provided
----------------
sentence_embedding      : mean-pooled BERT [CLS] embedding for a sentence
cosine_similarity       : cosine similarity between two sentence embeddings
context_word_similarity : similarity between sentence context and a target
                          word embedding (proxy for semantic congruency)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class SentenceMetrics:
    """Sentence-level metrics using a masked language model (BERT).

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (default ``"bert-base-uncased"``).
    device:
        Torch device string.  Defaults to CUDA when available.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> np.ndarray:
        """Return a mean-pooled embedding vector for *text*.

        Tokens are embedded using the last hidden state of the model and
        mean-pooled (excluding ``[CLS]`` and ``[SEP]``).

        Parameters
        ----------
        text:
            Any string (word, phrase, or sentence).

        Returns
        -------
        numpy.ndarray
            1-D array of shape ``(hidden_size,)``.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pool over non-special tokens (positions 1..-1)
        hidden = outputs.last_hidden_state[0, 1:-1, :]
        if hidden.shape[0] == 0:
            hidden = outputs.last_hidden_state[0]
        embedding = hidden.mean(dim=0).cpu().numpy()
        return embedding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Return the cosine similarity between the embeddings of *text_a*
        and *text_b*.

        Parameters
        ----------
        text_a, text_b:
            Strings to compare.

        Returns
        -------
        float
            Cosine similarity in [-1, 1].
        """
        emb_a = self.embed_text(text_a)
        emb_b = self.embed_text(text_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

    def context_target_similarity(
        self, context: str, target_word: str
    ) -> float:
        """Return the cosine similarity between the embedding of *context*
        and the embedding of *target_word*.

        This serves as a model-derived measure of semantic congruency:
        higher values indicate that the target word is semantically
        consistent with the sentence context.

        Parameters
        ----------
        context:
            The sentence fragment preceding the target word.
        target_word:
            The critical (target) word.

        Returns
        -------
        float
            Cosine similarity in [-1, 1].
        """
        return self.cosine_similarity(context, target_word)

    def sentence_metrics(
        self, sentence: str, critical_word: Optional[str] = None
    ) -> dict:
        """Return a dictionary of sentence-level metrics.

        Parameters
        ----------
        sentence:
            The full sentence string.
        critical_word:
            Optional – if provided, compute context/target similarity using
            all words *before* the critical word as context.

        Returns
        -------
        dict
            Keys: ``sentence``, ``n_tokens``, ``context_target_similarity``
            (only if *critical_word* is supplied).
        """
        tokens = self.tokenizer.tokenize(sentence)
        result: dict = {
            "sentence": sentence,
            "n_tokens": len(tokens),
        }

        if critical_word is not None:
            # Build context = sentence up to (but not including) critical word
            words = sentence.split()
            critical_lower = critical_word.lower()
            # Find last occurrence of the critical word (case-insensitive)
            context_words = words
            for i in range(len(words) - 1, -1, -1):
                if words[i].strip(".,;:!?\"'").lower() == critical_lower:
                    context_words = words[:i]
                    break
            context = " ".join(context_words)
            if context:
                result["context_target_similarity"] = self.context_target_similarity(
                    context, critical_word
                )
            else:
                result["context_target_similarity"] = None

        return result
