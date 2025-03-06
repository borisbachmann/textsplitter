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
        return graph_chunking(sentences=sentences,
                              embeddings=embeddings,
                              **kwargs)

# Mapping of paragraph segmenter names to segmenter classes
CHUNKER_MAP = {
    "linear": LinearChunker,
    "graph": GraphChunker
}
