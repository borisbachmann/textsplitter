from typing import Union, List, Protocol, Optional, Dict, Any

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from .chunk_backends import CHUNKER_MAP
from ..embeddings import EmbeddingModel
from ..sentences.sent_handling import SentenceModule
from ..utils import uniform_depth


class ChunkerProtocoll(Protocol):
    """
    Protocol for custom chunkers to implement.

    Args:
        data: Union[List[str], List[List[str]]]: List of strings or list of
            lists of strings to split into chunks.
        show_progress: bool: Show progress bar if True.

    Returns:
        List[List[str]]: List or list of lists of chunks as strings with one
            list of chunks for each input string or list of strings.
    """
    def __call__(self,
                 data: Union[List[str], List[List[str]]],
                 show_progress: bool = False
                 ) -> List[List[str]]:
        ...

class Chunker:
    """
    Chunker class that wraps around different chunk segmenters. Takes a text
    or alternatively a lists of texts and returns a list of chunks per text
    either in a flat list or a list of lists. Initialized with the name of a
    built-in chunker and a corresponding set of parameters. Alternatively
    initiated with a custom callable that implements the ChunkerProtocol.
    transformer model name.

    Args:
        segmenter: Union[str, SegmenterProtocol]: Name of a built-in segmenter
            or a custom segmenter callable implementing the SegmenterProtocol.
        language_or_model: Optional[str]: Language or model name for built-in
            segmenters.
    """

    def __init__(self,
                 model: Union[str, EmbeddingModel, SentenceTransformer],
                 specs: Dict[str, Any]):
        self.model = self._load_model(model)
        self.tokenizer = self.model.tokenizer

        # Load internal SentenceModule
        sent_specs = {"sentencizer": specs.get("sentencizer", ("pysbd", "de")),
                      "show_progress": False}
        para_specs = {"paragrapher": specs.get("paragrapher", "clean"),
                      "drop_placeholders": specs.get("drop_placeholders", [])}
        self.sentencizer = SentenceModule(sent_specs, para_specs)

        # Load working parameters
        chunker = specs.get("chunker", "linear")
        chunker_specs = specs.get("chunker_specs", {})
        chunker_specs["length_metric"] = self._calculate_length

        self._chunker = self._load_chunker(chunker, chunker_specs)

    def split(self,
              data: Union[str, List[str]],
              show_progress: bool = False,
              compile: bool = True,
              postprocess: bool = True,
              **chunker_kwargs
              ) -> Union[List[List[str]], List[str]]:
        if isinstance(data, str):
            # wrap to ensure that the segmenter receives a list
            chunks = self._split([data], **chunker_kwargs)
            if compile:
                chunks = self._compile_chunks(chunks)
            if postprocess:
                chunks = self._postprocess(chunks)
            # unwrap to return a single list
            return chunks[0]
        if isinstance(data, list):
            if not data:
                return []
            if all([isinstance(e, str) for e in data]):
                chunks = self._split(data, show_progress,
                                     **chunker_kwargs)
                chunks = self._postprocess(chunks)
                if compile:
                    chunks = self._compile_chunks(chunks)
                if postprocess:
                    chunks = self._postprocess(chunks)
                return chunks
        raise ValueError("Data must be either string or list of strings only.")

    def _split(self,
               data: List[str],
               show_progress: bool = False,
               **chunker_kwargs
               ) -> List[List[str]]:
        if show_progress:
            sentences = [self.sentencizer.split(t)
                         for t in tqdm(data, desc="Splitting sentences")]
            embeddings = [self._create_embeddings(s)
                          for s in tqdm(sentences, desc="Creating embeddings")]
            chunks = [self._chunker(sentences=s,
                                    embeddings=e,
                                    **chunker_kwargs)
                      for s, e in tqdm(zip(sentences, embeddings),
                                       desc="Chunking",
                                       total=len(sentences))
                      ]
        else:
            sentences = [self.sentencizer.split(t) for t in data]
            embeddings = [self._create_embeddings(s) for s in sentences]
            chunks = [self._chunker(sentences=s,
                                    embeddings=e,
                                    **chunker_kwargs)
                      for s, e in zip(sentences, embeddings)
                      ]
        return chunks

    def _create_embeddings(self, sentence, show_progress=False):
        return self.model.encode(sentence, show_progress_bar=show_progress)

    def _postprocess(self,
                     chunks: (Union[List[str], List[List[str]],
                              List[List[List[str]]]])
                     ) -> (Union[List[str], List[List[str]],
                           List[List[List[str]]]]):
        """
        Clear away irregularities in the sentence lists produced by different
        sentence segmenters. Removes leading and trailing whitespace and empty
        strings.
        """
        depth = uniform_depth(chunks)
        if depth == 1:
            return [c.strip() for c in chunks if c.strip()]
        if depth == 2:
            return [[c.strip() for c in chunk if c.strip()]
                    for chunk in chunks]
        if depth == 3:
            return [[[s.strip() for s in chunk if s.strip()] for chunk in chunk_list]
                    for chunk_list in chunks]

    def _compile_chunks(self, chunks):
        depth = uniform_depth(chunks)
        if depth == 2:
            return [make_text_from_chunk(c) for c in chunks]
        if depth == 3:
            return [[make_text_from_chunk(c) for c in chunk_list]
                    for chunk_list in chunks]

    def _calculate_length(self, sentence):
        tokens = self.tokenizer(sentence)
        return len(tokens["input_ids"])

    def _load_model(self, model):
        return load_model(model)

    def _load_chunker(self,
                      chunker: Union[str, ChunkerProtocoll],
                      chunker_specs: Dict[str, Any]
                      ):
        if isinstance(chunker, str):
            if chunker not in CHUNKER_MAP:
                raise ValueError(f"Invalid segmenter '{chunker}'. "
                                 f"Must be in: {list(CHUNKER_MAP.keys())}.")
            return CHUNKER_MAP[chunker](**chunker_specs)
        elif callable(chunker):
            return chunker
        else:
            raise ValueError("Chunker must be a string or callable. Custom "
                             "callables must implement the ChunkerProtocoll.")

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

def make_text_from_chunk(
        chunk: list
        ) -> str:
    """Reconstruct text data from a chunk."""
    texts = [sent.strip() for sent in chunk]
    return" ".join([text for text in texts])