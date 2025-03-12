from typing import Protocol, List, Dict

from .split import SplitTokenSegmenter
from .spacy import SpacyTokenSegmenter

# Mapping of all segmenters: Put all available segmenters here
TOKEN_SEGMENTER_MAP = {
    "spacy": SpacyTokenSegmenter,
    "split": SplitTokenSegmenter
}


# Protocoll for new segmenters: Implement this protocol for new segmenters
class TokenSegmenterProtocol(Protocol):
    """
    Protocol for custom token segmenters to implement.

    Args:
        texts: List[str]: List of texts to split into tokens.

    Returns:
        List[List[Dict]]: List of lists of tokens as dictionaries with one
            list of tokens for each input text and token information in dicts.
            Dicts have to include at least the following keys:
            - "token_id": Token ID
            - "token": Token text
    """
    def __call__(self,
                 texts: List[str],
                 show_progress: bool = False,
                 **kwargs
                 ) -> List[List[Dict]]:
        ...
