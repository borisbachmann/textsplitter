import re
from typing import Optional, List

from tqdm.auto import tqdm

from ..patterns import PARAGRAPH_PATTERN


class RegexParaSegmenter:
    """
    Regex-based paragraph segmenter that splits text into paragraphs based
    upon a regex pattern. Uses default pattern if no custom pattern is provided.

    Args:
        pattern: Optional[str]: Custom regex pattern to split paragraphs. If
        None, uses default pattern.
    """
    def __init__(self, pattern: Optional[str] = None):
        self.pattern = pattern or PARAGRAPH_PATTERN

    def __call__(self,
                 data: List[str],
                 show_progress: bool = False
                 ) -> List[List[str]]:
        """
        Split a list of strings into a list of lists containing paragraphs as
        strings based on a regex pattern.

        Args:
            data: List[str]: List of strings to split into paragraphs.

        Returns:
            List[List[str]]: List of lists of paragraphs as strings with one
                list of paragraphs for each input string.
        """
        if show_progress:
            return [self._split(text)
                    for text in tqdm(data, desc="Splitting paragraphs")]
        else:
            return [self._split(text) for text in data]

    def _split(self, text: str) -> List[str]:
        """
        Split a string into paragraphs based on a regex pattern. Returns a list
        of paragraphs with removed leading and trailing whitespace.

        Args:
            text: str: Text to split into paragraphs.
        """
        paragraphs = re.split(self.pattern, text)
        paragraphs = [p.strip() for p in paragraphs]
        return paragraphs
