import math
from collections import defaultdict

import community as community_louvain
import networkx as nx

from .chunk_utils import calculate_similarity, find_overlap, sigmoid


def graph_chunking(
        sentences: list,
        embeddings: list,
        K: int = 5,
        resolution: float = 1.0,
        model=None
        ) -> list:
    """Chunking strategy that accounts for a lookbehind and lookahead of K
    embeddings to avoid splitting at single "filling" embeddings.

    Adopted from: https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1
    """

    if len(sentences) == 1:
        return [sentences]

    tiles = get_tiles(embeddings, K, resolution, model)
    chunks = [sentences[tile[0]:tile[-1]+1] for tile in tiles]
    return chunks

def get_tiles(embeddings, K, resolution, model=None):
    # Create a similarity graph based on the extracted embeddings
    graph = create_similarity_graph(embeddings, K, model=model)

    # Initialize a Graph object from Networkx
    G = nx.Graph()
    for node in graph:
        G.add_edge(node[0], node[1], weight=node[2])

    # Partition the graph into communities using the Louvain method
    partition = community_louvain.best_partition(G, resolution=resolution,
                                                 weight='weight', randomize=False)

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
def create_similarity_graph(embeddings, K, model):
    """
    This function creates a graph of sentence similarities given a list of
    input embeddings. Each sentence is connected with the following K embeddings
    in the list. The similarity between each pair of embeddings is calculated
    using the provided model, and this similarity is then used to weight the
    edge between the embeddings in the graph.

    Parameters:
    - embeddings: a list of embeddings (strings) for which the similarity graph
    is to be created.
    - K: an integer indicating the number of following embeddings to be
    considered for each sentence.
    - model: a string indicating the model to be used for calculating sentence
    similarity.

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
    similarities = get_similarity_scores(a, b, model)
    # The similarity score for each pair of embeddings is incorporated into the edge weight.
    for i, s in enumerate(similarities):
        result[i][2] *= s

    # The final result is a list of weighted edges in the similarity graph.
    return result

# adopted from: https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1
def get_similarity_scores(vec_a, vec_b, model):
    """
    This function calculates the similarity scores between pairs of text chunks
    using the specified model. It supports several types of models, including
    BERT, sequence matcher, Jaccard index, and transformers from HuggingFace.

    Parameters:
    - vec_a, vec_b: Lists of text chunks to be compared. They must have the same
      length, and the comparison is made between corresponding pairs (i.e.,
      vec_a[i] is compared with vec_b[i]).
    - model: A string indicating the type of model to be used for the
      comparison. Current valid values are "bert", "seqmatch", "jaccard", and
      the name of any transformer available from HuggingFace.

    Returns:
    - similarities: A list of similarity scores for each pair of text chunks.
      The score ranges from 0 (no similarity) to 1 (identical).
    """

    # ORIGINAL IMPLEMENTATION OF MODELS COMMENTED OUT FOR THE TIME BEING
    # vec_a = [x.lower() for x in vec_a]  # convert vector a to lowercase
    # vec_b = [x.lower() for x in vec_b]  # convert vector b to lowercase
    #
    # if model == "bert":
    #     # BERTScore returns three values: Precision, Recall, and F1 Score
    #     # Here we use F1 Score as the similarity measure
    #     _, _, f1_score = score(vec_a, vec_b, lang='en', model_type='bert-base-uncased')
    #
    #     # f1_score is a tensor with the F1 score for each pair of embeddings.
    #     # Since we only have one pair, we take the first (and only) element.
    #     similarities = [f1_score[i].item() for i in range(len(vec_a))]
    # elif model == "seqmatch":
    #     translation_table = str.maketrans('', '', string.punctuation)
    #     similarities = []
    #     for xa, xb in zip(vec_a, vec_b):
    #         xa = xa.translate(translation_table)
    #         xb = xb.translate(translation_table)
    #         sm = SequenceMatcher(None, xa, xb).ratio()
    #         similarities.append(sm)
    # elif model == "jaccard":
    #     translation_table = str.maketrans('', '', string.punctuation)
    #     similarities = []
    #     for xa, xb in zip(vec_a, vec_b):
    #         a_words = set(xa.translate(translation_table).split())
    #         b_words = set(xb.translate(translation_table).split())
    #         js = len(a_words & b_words) / len(a_words | b_words)
    #         similarities.append(js)
    # else:
    #     # any transformer chosen from HuggingFace can be used here
    #     similarities = []
    #     tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/"+model)
    #     model = AutoModel.from_pretrained("sentence-transformers/"+model)
    #     embeddings_a = []
    #     embeddings_b = []
    #
    #     # Compute embeddings for each string in the lists
    #     for i in range(len(vec_a)):
    #         a_tokens = tokenizer(vec_a[i], padding=True, truncation=True, max_length=256, return_tensors='pt')
    #         b_tokens = tokenizer(vec_b[i], padding=True, truncation=True, max_length=256, return_tensors='pt')
    #
    #         with torch.no_grad():
    #             embeddings_a.append(model(**a_tokens).last_hidden_state.mean(dim=1))
    #             embeddings_b.append(model(**b_tokens).last_hidden_state.mean(dim=1))
    #
    #     # Compute cosine similarity for each pair of strings
    #     similarities = []
    #     for i in range(len(embeddings_a)):
    #         similarity = 1 - cosine(embeddings_a[i][0], embeddings_b[i][0])
    #         similarities.append(similarity)

    similarity_func = calculate_similarity

    similarities = []
    for i in range(len(vec_a)):
        # since cosine distance 1 - cosine similarity, we can use the cosine
        # similarity directly
        similarity = similarity_func(vec_a[i], vec_b[i])
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
