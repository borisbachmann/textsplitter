from typing import Callable, List, Optional

from numpy.typing import NDArray

from ..utils import calculate_similarity
from ..constants import DEFAULT_MAX_LENGTH, DEFAULT_THRESHOLD, DEFAULT_METRIC


def linear_chunking(
        sentences: List[str],
        embeddings: List[NDArray],
        length_metric: Callable,
        similarity_metric: Optional[str] = DEFAULT_METRIC,
        max_length: Optional[int] = DEFAULT_MAX_LENGTH,
        threshold: Optional[float] = DEFAULT_THRESHOLD
        ) -> List[List[str]]:

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
