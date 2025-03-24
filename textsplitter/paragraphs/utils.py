from typing import Tuple

from textsplitter.utils import find_substring_indices

def make_indices_from_paragraph(
        paragraph: str,
        text: str
        ) -> Tuple[int, int]:
    """
    Reconstruct indices of paragraph span in original text.

    Can handle paragraphs that consist of synthetically merged original
    line-break-based paragraphs (as e.g. in the "clean" paragrapher backend.
    Strings retrieved from these indices might then not be identical to the
    input paragraph string.

    Args:
        paragraph (str): Paragraph to reconstruct indices for.
        text (str): Original text containing paragraph.

    Returns:
        Tuple[int, int]: Start and end indices of paragraph in original text.
    """
    # Split paragraph into lines as some paragraphers might re-merge original
    # paragraphs into a single string
    original_paragraphs = paragraph.splitlines()
    start_p = original_paragraphs[0]
    end_p = original_paragraphs[-1]
    start_idx = find_substring_indices(text, [start_p])[0][0]
    end_idx = find_substring_indices(text, [end_p])[0][1]

    return start_idx, end_idx