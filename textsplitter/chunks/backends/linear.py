from typing import Callable, List, Optional

from numpy.typing import NDArray

from ..utils import calculate_similarity
from ..constants import DEFAULT_MAX_LENGTH, DEFAULT_THRESHOLD, DEFAULT_METRIC


def linear_chunking(
        sentences: List[str],
        embeddings: List[NDArray],
        length_metric: Callable,
        similarity_metric: Optional[str] = DEFAULT_METRIC,
        max_length: Optional[int] = DEFAULT_MAX_LENGTH,
        threshold: Optional[float] = DEFAULT_THRESHOLD
        ) -> List[List[str]]:
    """"
    Create semantic chunks from a list of sentences and corresponding
    embeddings. Chunks are created by linearly moving through the list of
    sentences and adding up sentences until a certain similiarity threshold or
    maximum length is breached. Similarity can be calculated as either the
    pairwise similarity of consecutive sentences or the cumulative similarity
    of all sentences in the current chunk.

    Args:
        sentences (List[str]): A list of consecutive sentences
        embeddings (List[NDArray]): A list of embeddings in the same order as
            original sentences
        length_metric (Callable): A function that takes a sentence and returns
            its length
        similarity_metric (str): Whether similarity should be calculated
            'pairwise' or 'cumulative'
        max_length (Optional[int]): The maximum length of the chunk according
            to the specified metric.
        threshold (Optional[float]): The minimum similarity for sentences to be
            added to the current chunk.

    Returns:
        List[List[str]: A list with chunks as lists of sentences.
    """

    def check_threshold_reached():
        if similarity_metric == "cumulative":
            return cumulative_similarity / len(current_chunk) < threshold
        return similarity < threshold

    def start_new_chunk():
        nonlocal current_chunk, current_length, cumulative_similarity
        current_chunk = [sent]
        current_length = sent_length
        cumulative_similarity = 0

    def build_chunk():
        nonlocal current_length
        current_chunk.append(sent)
        current_length += sent_length

    chunks = []
    current_chunk = []
    current_length = 0
    cumulative_similarity = 0

    for i, sent in enumerate(sentences):
        sent_length = length_metric(sent)

        if current_length + sent_length <= max_length:

            if current_chunk:
                similarity = calculate_similarity(
                    embeddings[i],
                    embeddings[i - 1])
                cumulative_similarity += similarity

                if check_threshold_reached():
                    chunks.append(current_chunk)
                    start_new_chunk()
                else:
                    build_chunk()
            else:
                build_chunk()
        else:
            chunks.append(current_chunk)
            start_new_chunk()

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class LinearEmbeddingChunker:
    """
    Chunker class that wraps around the linear chunking technique. Linear
    chunking splits a list of sentences into chunks bis aggregating consecutive
    sentences until either a certain maximum length is reached or a similarity
    threshold is exceeded. Linear chunking allows to create chunks with a
    precise maximum length based upon a specific metric (e.g. number of
    tokens, characters etc.). Maximum length and similarity threshold values
    are specified when calling the chunker.

    Args:
        length_metric (callable): A callable that takes a sentence as input and
            returns its length as a numerical value. If initialized via the
            Chunker class, the Chunker's tokenizer is used.
        similarity_metric (str): The similarity metric to measure against the
            threshold. Either 'pairwise' (i.e. similarity between two
            consecutive sentences) or 'cumulative' (i.e. average similarity
            between all sentences in the chunk). Defaults to 'pairwise'.
    """
    chunker_type = "embedding"

    def __init__(self,
                 length_metric: callable,
                 similarity_metric: str = DEFAULT_METRIC
                 ):
        self.length_metric = length_metric
        self.similarity_metric = similarity_metric

    def __call__(self,
                 sentences : List[str],
                 embeddings: List[NDArray],
                 **kwargs
                 ) -> List[List[str]]:
        """
        Call the linear chunking technique to create chunks from a list of
        consecutive sentences and corresponding embeddings.

        Args:
            sentences (List[str]): List of sentences to be chunked.
            embeddings (List[NDArray]): List of embeddings for each sentence.
            **kwargs: Additional keyword arguments to be passed to the linear
                chunking function:
                - max_length (int): Maximum length of a chunk.
                - threshold (float): Similarity threshold for chunking.
        """
        return linear_chunking(sentences=sentences,
                               embeddings=embeddings,
                               length_metric=self.length_metric,
                               similarity_metric=self.similarity_metric,
                               **kwargs)
