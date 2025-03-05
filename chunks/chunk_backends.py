from typing import List

from numpy._typing import NDArray
from text_splitter.chunks.chunkers.graph_chunker import graph_chunking
from text_splitter.chunks.chunkers.linear_chunker import linear_chunking
from text_splitter.constants import DEFAULT_METRIC


class LinearChunker:
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
    def __init__(self,
                 length_metric: callable
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
