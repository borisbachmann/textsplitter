from typing import Callable, List, Optional, Union

from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from ..embeddings import EmbeddingModel
from ..utils import calculate_similarity

# linear-chunking specific constants
DEFAULT_MAX_LENGTH = 512                # maximal length of produced chunks
DEFAULT_THRESHOLD = 0.3                 # similarity threshold for chunking
DEFAULT_METRIC = "pairwise"             # strategy to check threshold


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


class LinearChunkerBackend:
    """
        Chunker class that wraps around the linear chunking technique. Linear
        chunking splits a list of sentences into chunks bis aggregating consecutive
        sentences until either a certain maximum length is reached or a similarity
        threshold is exceeded. Linear chunking allows to create chunks with a
        precise maximum length based upon a specific metric (e.g. number of
        tokens, characters etc.). Maximum length and similarity threshold values
        are specified when calling the chunker.

        Args:
            model (Optional[str]): transformer model as a string or an instance of
                EmbeddingModel or SentenceTransformer. If a string, it must refer to
                a valid model from Hugging Face.
            length_metric (callable): A callable that takes a sentence as input and
                returns its length as a numerical value. If initialized via the
                Chunker class, the Chunker's tokenizer is used.
            similarity_metric (str): The similarity metric to measure against the
                threshold. Either 'pairwise' (i.e. similarity between two
                consecutive sentences) or 'cumulative' (i.e. average similarity
                between all sentences in the chunk). Defaults to 'pairwise'.
        """
    chunker_type = "simple"

    def __init__(
            self,
            model: str,
            length_metric: Callable = None,
            similarity_metric: str = DEFAULT_METRIC
    ):
        self.model = self._load_model(model)
        self.tokenizer = self.model.tokenizer
        self.length_metric = length_metric or self._calculate_length
        self.similarity_metric = similarity_metric

    def __call__(
            self,
            sentences: List[str],
            **kwargs
    ) -> List[List[str]]:
        # Create embeddings internally
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        chunks = linear_chunking(sentences=sentences,
                                 embeddings=embeddings,
                                 length_metric=self.length_metric,
                                 similarity_metric=self.similarity_metric,
                                 **kwargs)
        return chunks

    def _load_model(self,
                    model: Union[str, EmbeddingModel, SentenceTransformer]
                    ) -> Union[EmbeddingModel, SentenceTransformer]:
        """
        Load the internal transformer model used for the generation of
        embeddings and length calculations. Model can be specified as either a
        string, a SentenceTransformer instance or an EmbeddingModel instance
        which wraps any model from Hugging Face into a high-level interface
        similar to SentenceTransformer. If a string is passed, it must refer
        to a valid Hugging Face model and will be used to create an
        EmbeddingModel instance.

        Args:
            model (Union[str, EmbeddingModel, SentenceTransformer]): Model to
                be used. If a string, it must specify a valid model from Hugging
                Face.

        Returns:
            Union[EmbeddingModel, SentenceTransformer]: Model as an instance
                that mirrors the SentenceTransformer interface for the purposes
                of the EmbeddingChunker's methods.
        """
        if isinstance(model, str):
            return EmbeddingModel(model)
        elif (isinstance(model, EmbeddingModel) or
              isinstance(model, SentenceTransformer)):
            return model
        else:
            raise ValueError("Model must be a string or an instance of "
                             "EmbeddingModel or SentenceTransformer.")

    def _calculate_length(self, sentence: str) -> int:
        tokens = self.tokenizer(sentence)
        return len(tokens["input_ids"])
