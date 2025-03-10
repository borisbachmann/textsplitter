"""
The module provides classes for built-in sentence splitting methods as well as
a protocol for custom sentence segmenters to implement.

The SaT sentence splitter and models are described in:
Frohmann, M. et al. (2024): "Segment Any Text: A Universal Approach for Robust,
Efficient and Adaptable Sentence Segmentation", in: Al-Onaizan, Y. et. al.
(eds.): "Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing", pp. 11908--11941
("https://aclanthology.org/2024.emnlp-main.665")

and

Minixhofer, B. et al. (2023): "Where{'}s the Point? Self-Supervised
Multilingual Punctuation-Agnostic Sentence Segmentation", in: "Proceedings of
the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers)", pp. 7215--7235
(https://aclanthology.org/2023.acl-long.398)
"""

from typing import List, Protocol

import pysbd
import spacy
from tqdm.auto import tqdm
from wtpsplit import SaT


# Protocol for all Sentencizer backends (segmenters) to implement
class SentSegmenterProtocol(Protocol):
    """
    Protocol for custom sentence segmenters to implement.

    Args:
        data: List[str]: List of strings to split into sentences.

    Returns:
        List[List[str]]: List of lists of sentences as strings with one
            list of sentences for each input string.
    """
    def __call__(self,
                 data: List[str],
                 **kwargs
                 ) -> List[List[str]]:
        ...


# built-in sentence segmenters
class SpacySentSegmenter:
    """
    spaCy-based sentence splitter. Uses spaCy's dependency parser to split
    sentences. Initializes with a spaCy language model name.

    Note: The spaCy language model must be pre-downloaded on the system.
    """
    def __init__(self,
                 language_model: str):
        self._nlp = spacy.load(language_model)

    def __call__(self,
                 data: List[str],
                 **kwargs
                 ) -> List[List[str]]:
        """
        Split a list of strings into a list of lists containing sentences as
        strings. If show_progress is True, a tqdm progress bar is shown.

        Args:
            data: List[str]: List of strings to split into sentences.
            show_progress: bool: Show progress bar if True.

        Returns:
            List[List[str]]: List of lists of sentences as strings with one
                list of sentences for each input string.
        """
        show_progress = kwargs.get("show_progress", False)
        if show_progress:
            docs = self._nlp.pipe(tqdm(data, total=len(data)))
        else:
            docs = self._nlp.pipe(data)
        sentences = [[s.text for s in doc.sents] for doc in docs]
        return sentences

class SatSentSegmenter:
    """
    SaT-based sentence splitter. Uses the SaT sentence splitter for the
    wtsplit package to split sentences. Initializes with a SaT model name.

    Note: If not installed, the SaT model will be downloaded from the
    Hugging Face model hub.
    """
    def __init__(self,
                 model: str):
        self._sat = SaT(model)

    def __call__(self,
                 data: List[str],
                 **kwargs
                 ) -> List[List[str]]:
        """
        Split a list of strings into a list of lists containing sentences as
        strings. If show_progress is True, a tqdm progress bar is shown.

        Args:
            data: List[str]: List of strings to split into sentences.
            show_progress: bool: Show progress bar if True.

        Returns:
            List[List[str]]: List of lists of sentences as strings with one
                list of sentences for each input string.
        """
        show_progress = kwargs.get("show_progress", False)
        if show_progress:
            sentences = list(tqdm(self._sat.split(data), total=len(data)))
        else:
            sentences = list(self._sat.split(data))
        return sentences

class PysbdSentSegmenter:
    """
    PySBD-based sentence splitter. Uses the rules-based PySBD sentence splitter
    to split sentences. Initializes with an ISO language code.
    """
    def __init__(self,
                 language: str):
        self._seg = pysbd.Segmenter(language=language,
                                   clean=False,
                                   char_span=False)

    def __call__(self,
                 data: List[str],
                 **kwargs
                 ) -> List[List[str]]:
        """
        Split a list of strings into a list of lists containing sentences as
        strings. If show_progress is True, a tqdm progress bar is shown.

        Args:
            data: List[str]: List of strings to split into sentences.
            show_progress: bool: Show progress bar if True.

        Returns:
            List[List[str]]: List of lists of sentences as strings with one
                list of sentences for each input string.
        """
        show_progress = kwargs.get("show_progress", False)
        if show_progress:
            sentences = [self._seg.segment(t) for t in tqdm(data)]
        else:
            sentences = [self._seg.segment(t) for t in data]
        return sentences


# Mapping of segmenter names to built-in segmenter classes
SENT_SEGMENTER_MAP = {
    "pysbd": PysbdSentSegmenter,
    "sat":   SatSentSegmenter,
    "spacy": SpacySentSegmenter
}
