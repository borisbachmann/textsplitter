from typing import Protocol, List
from .pysbd import PysbdSentSegmenter

DEFAULT_SENT_BACKEND = PysbdSentSegmenter

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
