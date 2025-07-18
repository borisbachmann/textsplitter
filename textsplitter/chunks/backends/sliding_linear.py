from typing import Callable, List, Optional, Union
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from ..embeddings import EmbeddingModel
from ..utils import calculate_similarity
from ..constants import DEFAULT_MAX_LENGTH, DEFAULT_THRESHOLD, DEFAULT_METRIC


def semantic_sliding_chunking(
        sentences: List[str],
        embeddings: List[NDArray],
        length_metric: Callable,
        similarity_metric: Optional[str] = DEFAULT_METRIC,
        max_length: Optional[int] = DEFAULT_MAX_LENGTH,
        threshold: Optional[float] = DEFAULT_THRESHOLD,
        stride: int = 1,
        ) -> List[List[str]]:

    def check_threshold_reached():
        if similarity_metric == "cumulative":
            return cumulative_similarity / len(current_chunk) < threshold
        return similarity < threshold

    chunks = []
    for center_i in range(0, len(sentences), stride):
        current_chunk = [sentences[center_i]]
        current_length = length_metric(sentences[center_i])
        cumulative_similarity = 0

        left_i = center_i - 1
        right_i = center_i + 1
        direction = "left"

        while True:
            # to track if we managed to add anything
            old_length = len(current_chunk)

            if direction == "left" and left_i >= 0:
                left_sent = sentences[left_i]
                new_length = current_length + length_metric(left_sent)

                if not current_length > max_length:
                    similarity = calculate_similarity(
                        embeddings[left_i],
                        embeddings[left_i + 1])
                    cumulative_similarity += similarity

                    if not check_threshold_reached():
                        current_chunk.insert(0, left_sent)
                        current_length = new_length

                left_i -= 1
                direction = "right"

            elif direction == "right" and right_i < len(sentences):
                right_sent = sentences[right_i]
                new_length = current_length + length_metric(right_sent)

                if not current_length > max_length:
                    similarity = calculate_similarity(
                        embeddings[right_i],
                        embeddings[right_i - 1])
                    cumulative_similarity += similarity

                    if not check_threshold_reached():
                        current_chunk.append(right_sent)
                        current_length = new_length

                right_i += 1
                direction = "left"

            if len(current_chunk) == old_length:
                chunks.append(current_chunk)
                break

    return chunks



class SlidingEmbeddingChunker:
    """
    An embedding-based chunker class that implements a sliding-window approach:
    The window is defined by a maximum length and a similarity threshold,
    expanding in both directions from a central sentence until either one
    is breached. The window is then shifted by a stride and a new chunk is
    created around the new center.
    """
    chunker_type = "embedding"

    def __init__(self,
                 length_metric: Callable,
                 similarity_metric: str = DEFAULT_METRIC
                 ):
        self.length_metric = length_metric
        self.similarity_metric = similarity_metric

    def __call__(self,
                 sentences: List[str],
                 embeddings: List[NDArray],
                 **kwargs
                 ) -> List[List[str]]:
        """
        Call the sliding window chunking technique to create overlapping
        chunks from the list of sentences and embeddings.

        Args:
            sentences (List[str]): The text split into consecutive sentences.
            embeddings (List[NDArray]): The embeddings aligned to sentences.
            **kwargs: Additional keyword arguments (e.g. `max_length`, `threshold`,
                `lookbehind`, `lookahead`) to pass on.
        """
        return semantic_sliding_chunking(
            sentences=sentences,
            embeddings=embeddings,
            length_metric=self.length_metric,
            similarity_metric=self.similarity_metric,
            **kwargs
        )

class SlidingLinearChunkerBackend:
    """
    An embedding-based chunker class that implements a sliding-window approach:
    The window is defined by a maximum length and a similarity threshold,
    expanding in both directions from a central sentence until either one
    is breached. The window is then shifted by a stride and a new chunk is
    created around the new center.
    """
    chunker_type = "simple"

    def __init__(self,
            model: str,
            length_metric: Callable = None,
            similarity_metric: str = DEFAULT_METRIC
    ):
        self.model = self._load_model(model)
        self.tokenizer = self.model.tokenizer
        self.length_metric = length_metric or self._calculate_length
        self.similarity_metric = similarity_metric

    def __call__(self,
                 sentences: List[str],
                 **kwargs
                 ) -> List[List[str]]:
        """
        Call the sliding window chunking technique to create overlapping
        chunks from the list of sentences and embeddings.

        Args:
            sentences (List[str]): The text split into consecutive sentences.
            embeddings (List[NDArray]): The embeddings aligned to sentences.
            **kwargs: Additional keyword arguments (e.g. `max_length`, `threshold`,
                `lookbehind`, `lookahead`) to pass on.
        """
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        chunks = semantic_sliding_chunking(
            sentences=sentences,
            embeddings=embeddings,
            length_metric=self.length_metric,
            similarity_metric=self.similarity_metric,
            **kwargs
        )
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
