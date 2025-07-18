"""
Further chunking techniques can be added by implementing the
EmbeddingBackendProtocol and adding the new class to the CHUNKER_MAP dictionary.
"""
from typing import Protocol, List

import numpy as np

from .sliding_linear import SlidingEmbeddingChunker
from .graph import GraphEmbeddingChunker
from .linear import LinearEmbeddingChunker
from .sliding import SlidingSentChunker

# Mapping of all chunkers: Put all available chunkers here
CHUNK_BACKENDS_MAP = {
    "embedding": {"linear": LinearEmbeddingChunker,
                  "graph": GraphEmbeddingChunker,
                  "semantic_sliding": SlidingEmbeddingChunker,
                   },
    "simple": {"sliding": SlidingSentChunker}
}

# Protocols for new chunkers:
class EmbeddingChunkerProtocol(Protocol):
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
            embeddings: List[np.ndarray],
            **kwargs
    ) -> List[List[str]]:
        """
        Take a list of sentences and corresponding embeddings and return a list
        chunks as lists of consecutive sentences within each chunk based upon
        some embedding-based chunking technique.
        """
        ...

class SimpleChunkerProtocol(Protocol):
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
            **kwargs
    ) -> List[List[str]]:
        """
        Take a list of sentences and corresponding embeddings and return a list
        chunks as lists of consecutive sentences within each chunk based upon
        some embedding-based chunking technique.
        """
        ...
