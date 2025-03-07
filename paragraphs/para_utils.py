from text_splitter.utils import find_substring_indices

def make_indices_from_paragraph(paragraph, text):
    """Reconstruct indices of paragraph span in original text."""
    # Split paragraph into lines as some paragraphers might re-merge original
    # paragraphs into a single string
    original_paragraphs = paragraph.splitlines()
    start_p = original_paragraphs[0]
    end_p = original_paragraphs[-1]
    start_idx = find_substring_indices(text, [start_p])[0][0]
    end_idx = find_substring_indices(text, [end_p])[0][1]

    return start_idx, end_idx