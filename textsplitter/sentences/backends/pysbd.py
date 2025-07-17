from typing import List

import pysbd
from tqdm.auto import tqdm

from textsplitter import constants as pkg_const


class PysbdSentSegmenter:
    """
    PySBD-based sentence splitter. Uses the rules-based PySBD sentence splitter
    to split sentences. Initializes with an ISO language code.
    """
    def __init__(self,
                 language: str):
        self._seg = pysbd.Segmenter(
            language=language or pkg_const.DEFAULT_LANGUAGE["ISO 639-1"],
            clean=False,
            char_span=False
        )

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
