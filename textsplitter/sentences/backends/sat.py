from typing import List

from tqdm.auto import tqdm
from wtpsplit import SaT


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
