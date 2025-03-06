from typing import List

from numpy._typing import NDArray
from text_splitter.chunks.techniques.graph_chunking import graph_chunking
from text_splitter.chunks.techniques.linear_chunking import linear_chunking
from text_splitter.constants import DEFAULT_METRIC


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
