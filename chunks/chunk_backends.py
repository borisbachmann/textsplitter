"""
This module provides classes for built-in chunking methods. Classes of this
type are used to wrap around different embedding-based chunking techniques.
They do not have to implement the chunking logic themselves, but provide a
common interface defined by the EmbeddingChunker class. They take a list of
sentences and corresponding embeddings as input and return a list of chunks
as lists of sentences within each chunk.

Further chunking techniques can be added by implementing the EmbeddingChunker
protocol and adding the new class to the CHUNKER_MAP dictionary.
"""

from typing import List, Protocol

from numpy._typing import NDArray
from text_splitter.chunks.techniques.graph_chunking import graph_chunking
from text_splitter.chunks.techniques.linear_chunking import linear_chunking
from text_splitter.constants import DEFAULT_METRIC, DEFAULT_RES_MULTIPLIER


class EmbeddingChunker(Protocol):
    """
    Protocol class for embedding-based chunking techniques. Classes can be
    initiated with any kind of arguments and should implement the __call__
    method to chunk a list of sentences and corresponding embeddings into a
    list of chunks as lists of sentences within each chunk.
    """
    def __init__(self, **kwargs) -> None:
        ...

    def __call__(
            self,
            sentences: List[str],
            embeddings: List[NDArray],
            **kwargs
    ) -> List[List[str]]:
        """
        Take a list of sentences and corresponding embeddings and return a list
        chunks as lists of consecutive sentences within each chunk based upon
        some embedding-based chunking technique.
        """
        ...

# built-in types implementing the EmbeddingChunker protocol

class LinearChunker:
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


class GraphChunker:
    """
    Chunker class that wraps around the graph chunking technique. Graph chunking
    splits a list of sentences into chunks by creating a similarity graph
    between sentences and then applying a graph-based clustering algorithm to
    group sentences into chunks. Clusters are extracted by finding Louvain
    communities in the graph.

    Currently accepts a length_metric callable that is not used in the  current
    implementation but required for compatibility with the Chunker class.

    Args:
        length_metric (callable): A callable that takes a sentence as input and
            returns its length as a numerical value. If initialized via the
            Chunker class, the Chunker's tokenizer is used.
    """
    def __init__(self,
                 length_metric: callable = None
                 ):
        self.length_metric = length_metric
        pass

    def __call__(self,
                 sentences : List[str],
                 embeddings: List[NDArray],
                 **kwargs
                 ) -> List[List[str]]:
        """
        Call the graph chunking technique to create chunks from a list of
        consecutive sentences and corresponding embeddings.

        Args:
            sentences (List[str]): List of sentences to be chunked.
            embeddings (List[NDArray]): List of embeddings for each sentence.
            **kwargs: Additional keyword arguments to be passed to the graph
                chunking function:
                - K (int): Number of preceeding and following sentences to
                    connect in the graph.
                - resolution (float): Resolution parameter for the Louvain
                    community detection algorithm.

        Returns:
            List[List[str]]: List of chunks as lists of sentences within each
                chunk.
        """
        goal_length = kwargs.pop("goal_length", None)
        res_multiplier = kwargs.pop("res_multiplier", DEFAULT_RES_MULTIPLIER)

        if goal_length is not None:
            if isinstance(goal_length, int):
                resolution = len(sentences) / (goal_length * res_multiplier)
                return graph_chunking(sentences=sentences,
                                      embeddings=embeddings,
                                      resolution=resolution,
                                      **kwargs)
            else:
                raise ValueError("Invalid goal length value. Please provide an "
                                 "integer value or None.")
        else:
            return graph_chunking(sentences=sentences,
                                  embeddings=embeddings,
                                  **kwargs)

# Mapping of paragraph segmenter names to segmenter classes
# To extend the chunker with additional chunking techniques, add the new
# technique to the CHUNKER_MAP dictionary.
CHUNKER_MAP = {
    "linear": LinearChunker,
    "graph": GraphChunker
}
