from typing import List, Callable, Union

import math
from collections import defaultdict

import community as community_louvain
import networkx as nx
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from ..embeddings import EmbeddingModel
from ..utils import calculate_similarity, find_overlap, sigmoid
from ..constants import (DEFAULT_K, DEFAULT_RESOLUTION, DEFAULT_RANDOM_STATE,
                         DEFAULT_RES_MULTIPLIER)


def graph_chunking(
        sentences: list,
        embeddings: list,
        K: int = DEFAULT_K,
        resolution: float = DEFAULT_RESOLUTION,
        random_state: int = DEFAULT_RANDOM_STATE
        ) -> list:
    """Chunking strategy that accounts for a lookbehind and lookahead of K
    embeddings to avoid splitting at single "filling" embeddings.

    Adopted from: https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1
    """

    if len(sentences) == 1:
        return [sentences]

    tiles = get_tiles(embeddings, K, resolution, random_state)
    chunks = [sentences[tile[0]:tile[-1]+1] for tile in tiles]
    return chunks

def get_tiles(embeddings, K, resolution, random_state):
    # Create a similarity graph based on the extracted embeddings
    graph = create_similarity_graph(embeddings, K)

    # Initialize a Graph object from Networkx
    G = nx.Graph()
    for node in graph:
        G.add_edge(node[0], node[1], weight=node[2])

    # Partition the graph into communities using the Louvain method
    partition = community_louvain.best_partition(G, resolution=resolution,
                                                 weight='weight', random_state=
                                                 random_state)

    # Organize the embeddings into their respective communities (tiles)
    tiles = defaultdict(list)
    for k, v in partition.items():
        tiles[v].append(k)

    # Convert the defaultdict to a list and sort it
    tiles = list(tiles.values())
    tiles.sort()

    # Compact the clusters to remove range overlaps
    tiles = compact_clusters(tiles)

    # Original
    # Generate the text tiling in the form [0, th1, th2.... thN, len(embeddings)-1]
    # return [c[0] for c in tiles] + [tiles[-1][-1]]

    # Ammended: Return tiles directly instead
    return tiles

# adopted from https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1
def create_similarity_graph(embeddings, K):
    """
    This function creates a graph of sentence similarities given a list of
    input embeddings corresponding to sentences. Each is connected with the
    following K sentences' embeddings in the list. The similarity between
    each pair of embeddings is calculated and then used to weight the
    edge between the embeddings in the graph.

    Parameters:
    - embeddings: a list of embeddings (strings) for which the similarity graph
    is to be created.
    - K: an integer indicating the number of following embeddings to be
    considered for each sentence.

    Returns:
    - result: a list of weighted edges in the graph. Each edge is represented
    by a list of three elements: the index of the first sentence, the index of
    the second sentence, and the weight of the edge.
    """
    result = []
    couples = []

    # The outer loop iterates over all embeddings.
    for i in range(len(embeddings)):
        l = 0
        # The inner loop iterates over K embeddings following the sentence i.
        for j in range(i + 1, min(i + 1 + K, len(embeddings))):
            # Collecting pairs of embeddings and assign a decreasing weight to the edge connecting them.
            # This reflects the intuition that closer embeddings are more likely to be similar.
            couples.append((embeddings[i], embeddings[j]))
            result.append([i, j, math.exp(-l / 2)])  # weight decreases as we move further away
            l += 1

    # Couples are split into two separate lists which are fed to the similarity function
    a, b = zip(*couples)
    similarities = get_similarity_scores(a, b)
    # The similarity score for each pair of embeddings is incorporated into the edge weight.
    for i, s in enumerate(similarities):
        result[i][2] *= s

    # The final result is a list of weighted edges in the similarity graph.
    return result

# adopted from: https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1
def get_similarity_scores(emb_a, emb_b):
    """
    This function calculates the similarity scores between pairs of sentence
    embeddings.

    Parameters:
    - emb_a, emb_b: Lists of embeddings. They must have the same  length, and
      the comparison is made between corresponding pairs (i.e., emb_a[i] is
      compared with emb_b[i]).

    Returns:
    - similarities: A list of similarity scores for each pair of text chunks.
      The score ranges from 0 (no similarity) to 1 (identical).
    """
    similarity_func = calculate_similarity

    similarities = []
    for i in range(len(emb_a)):
        # since cosine distance 1 - cosine similarity, we can use the cosine
        # similarity directly
        similarity = similarity_func(emb_a[i], emb_b[i])
        sigmoid_similairty = sigmoid(similarity)  # apply sigmoid to force similarity between 0 and 1.
        similarities.append(sigmoid_similairty)

    return similarities


def compact_clusters(clusters):
    """
    This function takes a list of clusters and compacts them, eliminating range overlaps.
    It does this by iteratively merging overlapping clusters together until there are no more overlaps.
    The approach taken is to minimize the number of elements that need to be moved during the merge.

    Parameters:
    - clusters: a list of lists, where each sublist represents a cluster of elements.

    Returns:
    - compact_clusters: a list of compacted clusters. The clusters are sorted, and the elements within each cluster are also sorted.
    """
    compact_clusters = []
    while len(clusters):
        curr_cl = clusters.pop(0)
        if not curr_cl:
            pass
        for i in range(len(clusters)):
            target_cl = clusters[i]
            # find_overlap() returns the range overlaps and number of overlaps between two clusters
            overlap, n_1_in_2, n_2_in_1 = find_overlap(target_cl, curr_cl)
            if overlap:
                # The code block here decides which cluster to merge the overlapping elements into.
                # It aims to minimize the amount of element transfer. If it's equally easy to merge into both,
                # it merges into the current cluster.
                if n_1_in_2 < n_2_in_1 or n_2_in_1 == 0:
                    curr_cl.extend(overlap)
                    curr_cl = list(set(curr_cl))
                    clusters[i] = list(set(target_cl)-set(overlap))
                else:
                    target_cl.extend(overlap)
                    target_cl = list(set(target_cl))
                    curr_cl = list(set(curr_cl)-set(overlap))
                if not curr_cl:
                    break
        # After examining all clusters, if the current cluster still has elements, it's added to the compacted list.
        if len(curr_cl):
            # remove any possible duplicates
            compact_clusters.append(list(set(curr_cl)))

    # compact_clusters

    for cl in compact_clusters:
        cl.sort()
    compact_clusters.sort()
    return compact_clusters


class GraphEmbeddingChunker:
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
    chunker_type = "embedding"

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


class GraphChunkerBackend:
    """
    Chunker class that wraps around the graph chunking technique. Graph chunking
    splits a list of sentences into chunks by creating a similarity graph
    between sentences and then applying a graph-based clustering algorithm to
    group sentences into chunks. Clusters are extracted by finding Louvain
    communities in the graph.

    Currently accepts a length_metric callable that is not used in the  current
    implementation but required for compatibility with the Chunker class.

    Args:
        model (Optional[str]): transformer model as a string or an instance of
            EmbeddingModel or SentenceTransformer. If a string, it must refer to
            a valid model from Hugging Face.
    """
    chunker_type = "simple"

    def __init__(self,
                 model: str = None,
                 ):
        self.model = self._load_model(model)

    def __call__(self,
                 sentences : List[str],
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
                embeddings = self.model.encode(
                    sentences, show_progress_bar=False
                )
                resolution = len(sentences) / (goal_length * res_multiplier)
                chunks = graph_chunking(
                    sentences=sentences,
                    embeddings=embeddings,
                    resolution=resolution,
                    **kwargs
                )
            else:
                raise ValueError("Invalid goal length value. Please provide an "
                                 "integer value or None.")
        else:
            embeddings = self.model.encode(
                sentences, show_progress_bar=False
            )
            chunks = graph_chunking(
                sentences=sentences,
                embeddings=embeddings,
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
