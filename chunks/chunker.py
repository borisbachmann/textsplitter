from typing import Union, List, Protocol, Dict, Any

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers.models.cvt.convert_cvt_original_pytorch_checkpoint_to_pytorch import embeddings

from .chunk_backends import CHUNKER_MAP
from .chunk_utils import make_text_from_chunk
from ..embeddings import EmbeddingModel
from ..sentences.sent_handling import SentenceSegmenter
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
        self.sentencizer = SentenceSegmenter(sent_specs, para_specs)

        # Load working parameters
        chunker = specs.get("chunker", "linear")
        chunker_specs = specs.get("chunker_specs", {})
        chunker_specs["length_metric"] = self._calculate_length

        self._chunker = self._load_chunker(chunker, chunker_specs)

    def split(self,
              data: Union[str, List[str]],
              compile: bool = True,
              postprocess: bool = True,
              **kwargs
              ) -> Union[List[List[str]], List[str]]:

        ensure_separators = kwargs.pop("ensure_separators", False)
        show_progress = kwargs.pop("show_progress", False)

        if isinstance(data, str):
            # wrap to ensure that the segmenter receives a list
            chunks = self._split([data], **kwargs)
            if compile:
                chunks = self._compile_chunks(chunks, ensure_separators)
            if postprocess:
                chunks = self._postprocess(chunks)
            # unwrap to return a single list
            return chunks[0]
        if isinstance(data, list):
            if not data:
                return []
            if all([isinstance(e, str) for e in data]):
                chunks = self._split(data, show_progress,
                                     **kwargs)
                chunks = self._postprocess(chunks)
                if compile:
                    chunks = self._compile_chunks(chunks, ensure_separators)
                if postprocess:
                    chunks = self._postprocess(chunks)
                return chunks
        raise ValueError("Data must be either string or list of strings only.")

    def _split(self,
               data: List[str],
               show_progress: bool = False,
               **kwargs
               ) -> List[List[str]]:
        drop_placeholders = kwargs.pop("drop_placeholders", [])

        sentences = self.sentencizer.split_list(
            data, drop_placeholders=drop_placeholders,
            show_progress=show_progress)
        embeddings = self._create_embeddings(
            sentences, show_progress=show_progress)

        if show_progress:
            chunks = [self._chunker(sentences=s,
                                    embeddings=e,
                                    **kwargs)
                      for s, e in tqdm(zip(sentences, embeddings),
                                       desc="Chunking sentences",
                                       total=len(sentences))
                      ]
        else:
            chunks = [self._chunker(sentences=s,
                                    embeddings=e,
                                    **kwargs)
                      for s, e in zip(sentences, embeddings)
                      ]
        return chunks

    def _create_embeddings(self, sentences, show_progress=False):
        if show_progress:
            iterator = tqdm(sentences, desc="Creating embeddings")
        else:
            iterator = sentences
        return [self.model.encode(sent_list, show_progress_bar=False)
                for sent_list in iterator]

    def _postprocess(self,
                     chunks: (Union[List[str], List[List[str]],
                              List[List[List[str]]]])
                     ) -> Union[List[str], List[List[str]],
                                List[List[List[str]]]]:
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

    def _compile_chunks(self, chunks, ensure_separators):
        depth = uniform_depth(chunks)
        if depth == 2:
            return [make_text_from_chunk(c, ensure_separators=ensure_separators)
                    for c in chunks]
        if depth == 3:
            return [[make_text_from_chunk(c, ensure_separators=ensure_separators)
                     for c in chunk_list]
                    for chunk_list in chunks]

    def _calculate_length(self, sentence):
        tokens = self.tokenizer(sentence)
        return len(tokens["input_ids"])

    def _load_model(self, model: Union[str, EmbeddingModel, SentenceTransformer]
                   ) -> Union[EmbeddingModel, SentenceTransformer, str]:
        if isinstance(model, str):
            return EmbeddingModel(model)
        elif (isinstance(model, EmbeddingModel) or
              isinstance(model, SentenceTransformer)):
            return model
        else:
            raise ValueError("Model must be a string or an instance of "
                             "EmbeddingModel or SentenceTransformer.")

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
