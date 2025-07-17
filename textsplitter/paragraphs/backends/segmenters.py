from typing import Protocol, List
from .clean import CleanParaSegmenter

# Standard segmenter to use if no segmenter is specified
DEFAULT_PARA_BACKEND = CleanParaSegmenter


# Protocol for all paragraph backends (segmenters) to implement
class ParaSegmenterProtocol(Protocol):
    """
    Protocol for custom paragraph segmenters to implement.

    Args:
        data (List[str]): List of strings to split into sentences.

    Returns:
        List[List[str]]: List of lists of sentences as strings with one
            list of sentences for each input string.
    """
    def __call__(self,
                 data: List[str],
                 *args,
                 **kwargs
                 ) -> List[List[str]]:
        ...
