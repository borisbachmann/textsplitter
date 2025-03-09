import re
from typing import Optional, Dict, List

from tqdm.auto import tqdm

from .constants import BULLETS
from .patterns import (PARAGRAPH_PATTERN_SIMPLE, ENUM_PATTERN_NO_DATE_DE,
                       PARAGRAPH_PATTERN)

class CleanParagrapher:
    """
    Linebreak-based paragraph segmenter that applies rules to re-merge
    bullet-point and enumerated groups into paragraphs.
    """

    def __init__(self, specs: Optional[Dict] = None):
        if specs is None:
            specs = {}

        self.paragraph_pattern = specs.get("paragraph_pattern",
                                           PARAGRAPH_PATTERN_SIMPLE)
        self.enum_pattern = specs.get("enum_pattern", ENUM_PATTERN_NO_DATE_DE)
        self.bullets = specs.get("bullets", BULLETS)

    def __call__(self,
                 data: List[str],
                 merge_bullets: Optional[bool] = True,
                 show_progress: bool = False,
                 ) -> List[List[str]]:
        """
        Split a list of strings into a list of lists containing paragraphs as
        strings. If merge_bullets is True, bullet points and enumerations will
        be kept together if the preceding paragraph end signals semantic
        continuation (i.e. ends with "," or ":").

        Args:
            data: List[str]: List of strings to split into paragraphs.
            show_progress: bool: Show progress bar if True.

        Returns:
            List[List[str]]: List of lists of paragraphs as strings with one
                list of paragraphs for each input string.
        """
        if show_progress:
            return [self._split(text, merge_bullets)
                    for text in tqdm(data, desc="Splitting paragraphs")]
        else:
            return [self._split(text, merge_bullets) for text in data]

    def _split(self,
               text: str,
               merge_bullets: Optional[bool] = True,
               ) -> List[str]:
        """
        From a string of text, return a list of cleaned paragraphs as strings.
        By default, the function splits paragraphs at line breaks and tries to
        perserve bullet-point and enumarted groups for later semantic processing.
        Groups are preserved if the preceding paragraph end signals semantic
        continuation (i.e. ends with "," or ":"). Line breaks within
        bullet-point groups are preserved to allow for later-on handling like
        splitting etc.

        Args:
            text: str: Text to split into paragraphs.
            merge_bullets: Optional[bool]: Merge bullet-point groups if True.
        """
        paragraphs = re.split(self.paragraph_pattern, text)
        cleaned_paragraphs = [p.strip() for p in paragraphs]
        cleaned_paragraphs = [p for p in cleaned_paragraphs if len(p) > 0]

        if merge_bullets:
            merged_paragraphs = []
            current_group = []
            for p in cleaned_paragraphs:
                if not current_group:
                    current_group.append(p)
                else:
                    if (current_group[0].endswith(tuple([":", ","])) or
                        current_group[0][-1].isalpha()) and (
                            p.startswith(tuple(self.bullets))
                            or re.match(self.enum_pattern, p)
                    ):
                        current_group.append(p)
                    else:
                        merged_paragraphs.append(current_group)
                        current_group = [p]

            # Don't forget to merge the last group
            if current_group:
                merged_paragraphs.append(current_group)

            merged_paragraphs = ["\n".join(group) for group in merged_paragraphs]
        else:
            merged_paragraphs = cleaned_paragraphs

        return merged_paragraphs


class RegexParagrapher:
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

# Mapping of paragraph segmenter names to segmenter classes
PARA_SEGMENTER_MAP = {
    "clean": CleanParagrapher,
    "regex": RegexParagrapher
}
