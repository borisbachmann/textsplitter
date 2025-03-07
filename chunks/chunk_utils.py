import re
from typing import Union, Optional

import math

from numpy.typing import NDArray
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

from text_splitter.utils import find_substring_indices
from ..embeddings import EmbeddingModel


class TokenCounter:
    """
    Count tokens in a text using a Hugging Face tokenizer or SentenceTransformer.
    """

    def __init__(self,
                 model: Union[str, EmbeddingModel, SentenceTransformer]):
        self.tokenizer = self._make_tokenizer(model)

    def __call__(self, text: str) -> int:
        tokens = self.tokenizer(text)
        return len(tokens["input_ids"])

    def _make_tokenizer(self,
                        model: Union[str, EmbeddingModel, SentenceTransformer]):
        """Create a tokenizer from a model. If string, load tokenizer from
        Hugging Face, if EmbeddingModel or SentenceTransformer, use the
        tokenizer attribute."""
        if isinstance(model, str):
            return AutoTokenizer.from_pretrained(model)
        elif isinstance(model, EmbeddingModel):
            return model.tokenizer
        elif isinstance(model, SentenceTransformer):
            return model.tokenizer
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")


def calculate_similarity(
        embedding_1: NDArray,
        embedding_2: NDArray) \
        -> float:
    """Calculate cosine similarity between two embeddings."""
    return cosine_similarity([embedding_1], [embedding_2])[0][0]


# adpted from https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1
def bert_similarity(
        sentence_1: str,
        sentence_2: str
        ) -> float:
    """Calculate similarity between two sentences using BERT."""
    # BERTScore returns three values: Precision, Recall, and F1 Score
    # Here we use F1 Score as the similarity measure
    sentence_1, sentence_2 = [sentence_1], [sentence_2]
    _, _, f1_score = score(sentence_1, sentence_2, lang='de',
                           model_type='bert-base-uncased')

    # f1_score is a tensor with the F1 score for each pair of sentences.
    # Since we only have one pair, we take the first (and only) element.
    similarities = [f1_score[i].item() for i in range(len(sentence_1))]

    return similarities[0]


# adopted from https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1
def vec_intersection(vec1, vec2, preserve_order=False):
    if preserve_order:
        set2 = frozenset(vec2)
        return [x for x in vec1 if x in set2]
    else:
        vec1 = list(vec1)
        vec2 = list(vec2)
        vec1.sort()
        vec2.sort()
        t = []
        l1 = len(vec1)
        l2 = len(vec2)
        i = j = 0
        while i < l1 and j < l2:
            if vec1[i] == vec2[j]:
                t.append(vec1[i])
                i += 1
                j += 1
            elif vec1[i] < vec2[j]:
                i += 1
            elif vec1[i] > vec2[j]:
                j += 1
        return t

# adopted from https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1
def vec_complement(vec1, vec2, preserve_order=False):
    if preserve_order:
        return [x for x in vec1 if x not in vec2]
    else:
        vec1 = list(vec1)
        vec2 = list(vec2)
        vec1.sort()
        vec2.sort()
        t = []
        l1 = len(vec1)
        l2 = len(vec2)
        i = j = 0
        while i < l1 and j < l2:
            if vec1[i] == vec2[j]:
                i += 1
                j += 1
            elif vec1[i] < vec2[j]:
                t.append(vec1[i])
                i += 1
            elif vec1[i] > vec2[j]:
                j += 1
        t.extend(vec1[i:])
        return list(t)

# adopted from https://towardsdatascience.com/text-tiling-done-right-building-solid-foundations-for-your-personal-llm-e70947779ac1
def find_overlap(vector1, vector2):
    # Added check for empty vectors which raised errors
    if not vector1 or not vector2:
        return [], 0, 0

    min_v1, max_v1 = min(vector1), max(vector1)
    min_v2, max_v2 = min(vector2), max(vector2)

    one_in_two = [num for num in vector1 if min_v2 <= num <= max_v2]
    two_in_one = [num for num in vector2 if min_v1 <= num <= max_v1]
    overlap = one_in_two + two_in_one

    return overlap, len(one_in_two), len(two_in_one)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

if __name__ == '__main__':
    sentence_1 = "Ich gehe nach Hause."
    sentence_2 = "Ich gehe heim."
    print("BERT similarity:", bert_similarity(sentence_1, sentence_2))


def make_text_from_chunk(
        chunk: list,
        ensure_separators: Optional[bool] = False
        ) -> str:
    """Reconstruct text data from a chunk."""

    def add_separator(text):
        if not re.search(r'[.!?]["\']*$', text):
            return text + "."
        return text

    texts = [sent.strip() for sent in chunk]
    if ensure_separators:
        texts = [add_separator(text) for text in texts]
    return" ".join([text for text in texts])


def make_indices_from_chunk(chunk, text):
    """Reconstruct indices of chunk span in original text."""
    start_sent = chunk[0]
    end_sent = chunk[-1]
    start_idx = find_substring_indices(text, [start_sent])[0][0]
    end_idx = find_substring_indices(text, [end_sent])[0][1]
    return start_idx, end_idx
