from typing import List

import spacy
from tqdm.auto import tqdm


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
