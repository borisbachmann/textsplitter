import re
from typing import List

from ..constants import BULLETS
from ..patterns import (PARAGRAPH_PATTERN, PARAGRAPH_PATTERN_SIMPLE,
                        ENUM_PATTERN_NO_DATE)


def split_regex_paragraphs(text: str,
                           paragraph_pattern: re.Pattern = PARAGRAPH_PATTERN
                           ) -> List[str]:
    """Split a string into paragraphs based on a regex pattern. Returns a list
    of paragraphs with removed leading and trailing whitespace."""
    paragraphs = re.split(paragraph_pattern, text)
    paragraphs = [p.strip() for p in paragraphs]
    return paragraphs


def split_clean_paragraphs(
        text: str,
        merge_bullets: bool = True,
        paragraph_pattern: re.Pattern = PARAGRAPH_PATTERN_SIMPLE,
        enum_pattern=ENUM_PATTERN_NO_DATE,
        bullets=BULLETS
        ) -> List[str]:
    """From a string of text, return a list of cleaned paragraphs as strings.
    By default, the function splits paragraphs at line breaks and tries to
    perserve bullet-point and enumarted groups for later semantic processing.
    Groups are preserved if the preceding paragraph end signals semantic
    continuation (i.e. ends with "," or ":"). Line breaks within bullet-point
    groups are preserved to allow for later-on handling like splitting etc.
    """
    paragraphs = re.split(paragraph_pattern, text)
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
                    p.startswith(tuple(bullets)) or re.match(enum_pattern, p)):
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
