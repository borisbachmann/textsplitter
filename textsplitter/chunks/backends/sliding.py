from typing import List


def generate_sentence_n_grams(
        sentences: List[str],
        length: int,
        enforce: bool = False
        ) -> List[List[str]]:
    """
    Return a list of chunks as lists of sentences based upon a sliding
    window of specified length over the input sentences. Resulting chunks
    will overlap.

    Args:
        sentences (List[str]): A list of sentences.
        length (int): The length of the sliding window.
        enforce (bool): If True, return a single chunk for input lists
            that are shorter than the specified length. (default: False)

    Returns:
        List[List[str]]: A list of chunks as lists of sentences per chunk.
    """

    if (
        not isinstance(sentences, list)
        or not all(isinstance(s, str) for s in sentences)
    ):
        raise ValueError("sentences must be a list of strings.")

    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer.")

    if enforce and len(sentences) < length:
        return [sentences]

    return [sentences[i: i + length] for i
            in range(len(sentences) - length + 1)]


class SlidingSentChunker:
    """
    Chunker class that creates chunks by sliding a window of specified length
    over a list of sentences. Returns a list of chunks as lists of sentences
    covering all consecutive sentence groups of specified length in the input.
    Returned chunks will overlap.
    """
    chunker_type = "simple"

    def __init__(self):
        pass

    def __call__(self,
                 sentences : List[str],
                 length : int,
                 enforce : bool = False
                 ) -> List[List[str]]:
        """
        Return a list of chunks as lists of sentences based upon a sliding
        window of specified length over the input sentences. Resulting chunks
        will overlap.

        Args:
            sentences (List[str]): A list of sentences.
            length (int): The length of the sliding window.
            enforce (bool): If True, return a single chunk for input lists
                that are shorter than the specified length. (default: False)

        Returns:
            List[List[str]]: A list of chunks as lists of sentences per chunk.
        """
        return generate_sentence_n_grams(sentences, length, enforce)

