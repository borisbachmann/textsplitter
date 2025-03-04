from typing import List, Union

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from text_splitter import EmbeddingModel
from text_splitter.chunks.chunk_handling import make_text_from_chunk
from text_splitter.chunks.linear_chunker import linear_chunking
from text_splitter.constants import DEFAULT_METRIC, DEFAULT_MAX_LENGTH, DEFAULT_THRESHOLD
from text_splitter.sentences.sent_handling import SentenceModule


class LinearChunker:
    def __init__(self, model, **specs):
        self.model = self._load_model(model)
        self.tokenizer = self.model.tokenizer

        # Load internal SentenceModule
        sent_specs = {"sentencizer": specs.get("sentencizer", {}),
                      "show_progress": False}
        para_specs = {"paragrapher": specs.get("paragrapher", "clean"),
                      "drop_placeholders": specs.get("drop_placeholders", [])}
        self.sentencizer = SentenceModule(sent_specs, para_specs)

        self.length_metric = specs.get("length_metric", self._calculate_length)
        self.similarity_metric = specs.get("similarity_metric", DEFAULT_METRIC)
        self.max_length = specs.get("max_length", DEFAULT_MAX_LENGTH)
        self.threshold = specs.get("threshold", DEFAULT_THRESHOLD)

    def split(self,
              data: List[str],
              show_progress: bool = False,
              ) -> List[List[str]]:
        if show_progress:
            sentences = [self.sentencizer.split(t) for t in tqdm(data, desc="Splitting sentences")]
            embeddings = [self._create_embeddings(s) for s in tqdm(sentences, desc="Creating embeddings")]
            chunks = [linear_chunking(sentences=s, embeddings=e,
                                      length_metric=self.length_metric,
                                      similarity_metric=self.similarity_metric,
                                      max_length=self.max_length,
                                      threshold=self.threshold)
                      for s, e in tqdm(zip(sentences, embeddings),
                                       desc="Chunking",
                                       total=len(sentences))
                      ]
            chunks = [[make_text_from_chunk(c) for c in chunk_list]
                      for chunk_list in chunks]
        else:
            sentences = [self.sentencizer.split(t) for t in data]
            embeddings = [self._create_embeddings(s) for s in sentences]
            chunks = [linear_chunking(sentences=s,
                                     embeddings=e,
                                     length_metric=self.length_metric,
                                     similarity_metric=self.similarity_metric,
                                     max_length=self.max_length,
                                     threshold=self.threshold)
                      for s, e in zip(sentences, embeddings)
                      ]
            chunks = [[make_text_from_chunk(c) for c in chunk_list]
                      for chunk_list in chunks]
        return chunks

    def _create_embeddings(self, sentence, show_progress=False):
        return self.model.encode(sentence, show_progress_bar=show_progress)

    def _calculate_length(self, sentence):
        tokens = self.tokenizer(sentence)
        return len(tokens["input_ids"])

    def _load_model(self, model):
        return load_model(model)

class GraphChunker:
    def __init__(self, model, **specs):
        self.model = self._load_modal(model)
        print(specs)

        def _load_model(self, model):
            return model
        pass


def load_model(model: Union[str, EmbeddingModel, SentenceTransformer]
               ) -> Union[EmbeddingModel, SentenceTransformer, str]:
    if isinstance(model, str):
        return EmbeddingModel(model)
    elif (isinstance(model, EmbeddingModel) or
          isinstance(model, SentenceTransformer)):
        return model
    else:
        raise ValueError("Model must be a string or an instance of "
                         "EmbeddingModel or SentenceTransformer.")



# Mapping of paragraph segmenter names to segmenter classes
CHUNK_SEGMENTER_MAP = {
    "linear": LinearChunker,
    "graph": GraphChunker
}
