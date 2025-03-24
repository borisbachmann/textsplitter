from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    """
    A simple wrapper class for creating Embeddings with the Hugging Face
    Transformers library.

    model_name (str): Name of the model to be used for creating embeddings.
        Must be a valid model name from the Hugging Face Transformers library.
    """

    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts: list, batch_size: int = 32, show_progress_bar=True) -> np.ndarray:
        """
        Create embeddings for a list of texts using the classes specified model.
        """
        embeddings_list = []
        range_iter = range(0, len(texts), batch_size)

        iterator = tqdm(range_iter, desc="Batches") if show_progress_bar \
                   else range_iter

        for start_idx in iterator:
            batch_texts = texts[start_idx:start_idx + batch_size]
            embeddings_list.append(self._process_batch(batch_texts))

        return np.concatenate(embeddings_list, axis=0)

    def _process_batch(self, batch: list) -> np.ndarray:
        inputs = self.tokenizer(batch, padding=True, truncation=True,
                                return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def __repr__(self):
        return f"EmbeddingModel('{self.model_name}')"

    def __str__(self):
        return f"EmbeddingModel('{self.model_name}')"


def create_embeddings(
        sentences: List[str],
        model: Union[str, EmbeddingModel, SentenceTransformer]
        ) -> Optional[Union[list, np.ndarray]]:
    if isinstance(model, str):
        model = EmbeddingModel(model)
    return model.encode(sentences, show_progress_bar=False)
