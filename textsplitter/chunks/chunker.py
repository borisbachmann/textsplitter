"""
This module includes basic Chunker classes that wrap around different chunking
techniques. Chunkers are used as internal objects within the ChunkHandler class
to split texts into chunks. The ChunkerProtocoll defines the interface for all
chunkers to work with the ChunkHandler.

The DummyChunker is a placeholder chunker that returns the original text as a
single chunk, the EmbeddingChunker is a more complex chunker that uses an
embedding model to split texts into sentences which are then turned back into
chunks.

Other chunkers that can be added can rely on different chunking
techniques. All have to be able to handle specs for internal paragraph and
sentence segmenters upon intialization, even if they do not use them.
"""

from typing import Protocol, Union, List, Optional, Dict, Any

from numpy._typing import NDArray
from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm

from.backends import (EmbeddingChunkerProtocol, SimpleChunkerProtocol,
                      CHUNK_BACKENDS_MAP)
from .embeddings import EmbeddingModel
from .utils import make_text_from_chunk
from ..sentences.handling import SentenceHandler
from textsplitter.utils import uniform_depth


class ChunkerProtocol(Protocol):
    """
    Protocol for custom chunkers to implement.

    Args:
        args: positional arguments for initialization of the chunker.
        kwargs: keyword arguments for initialization of the chunker.
        """
    def __init__(self, *args, **kwargs):
        ...

    def split(self,
              data: Union[List[str], List[List[str]]],
              compile: bool = True,
              postprocess: bool = True,
              show_progress: Optional[bool] = False,
              ) -> Union[List[List[str]], List[str]]:
        """
        Take a list of strings or a list of lists of strings and return a list
        of chunks as lists of strings.

        Args:
            data (Union[List[str], List[List[str]]]): List of strings or a list
                of lists of strings to split into chunks.
            compile (bool, optional): If True, the chunker will use its _compile
                method to compile the chunks and return a list of strings.
            postprocess (bool, optional): If True, the chunker will use its
                _postprocess method to postprocess the chunk strings before
                returning them.
            show_progress (bool, optional): If True, the chunker will show a
                progress bar when chunking more than one string.

        Returns:
            Union[List[List[str]], List[str]]: List of chunks as lists of
                strings (uncompiled) or a list of strings (compiled). Compiled
                chunks have to include uncompiled strings with the first and
                last strings identical with the beinning and end of the complete
                chunk to enable span identification by the ChunkHandler.
        """
        ...

    def _compile_chunks(self,
                        chunks: Union[List[List[str]], List[List[List[str]]]],
                        ensure_separators: bool = False
                        ) -> Union[List[str], List[List[str]]]:
        """
        Compile consecutive chunk substrings into single chunk strings.

        Compiled chunks have to include uncompiled strings with the first and
        last strings identical with the beinning and end of the complete chunk
        to enable span identification by the ChunkHandler.

        Args:
            chunks (Union[List[List[str]], List[List[List[str]]]): List of
                chunks as lists of strings or a list of lists of chunks as lists
                of ensure_separators: bool: Argument to be caught and ignored.
            ensure_separators (bool, optional): Argument to place separator
                punctuation between chunk substrings if needed by this techique.
                Default is False.

        Returns:
            Union[List[str], List[List[str]]]: List of chunks as strings or a
                list of lists of chunks as strings with exactly one string per
                chunk.
        """
        ...

    def _postprocess(self,
                     data: List[List[str]],
                     **kwargs
                     ) -> List[List[str]]:
        """
        Apply some postprocessing if needed by the chunking technique. May also
        return unchanged data if not needed by chunking technique.

        Args:
            data (List[List[str]]): List of chunks as lists of strings.
            **kwargs: Additional keyword arguments as needed for postprocessing.

        Returns:
            List[List[str]]: List of chunks as lists of strings with
                postprocessing applied.
        """


class DummyChunker:
    """
    Dummy chunker that returns a list with the original text as a solitary
    chunk. Catches all arguments upon initialization but does not use them.

    The class mirrors the behavior of other chunkers but does not actually
    perform any chunking. Its main purpose is to serve as a functional
    placeholder in lightweight initializations of the TextSplitter class.

    Args:
        *args: Additional positional arguments to be caught and ignored.
        **kwargs: Additional keyword arguments to be caught and ignored.
    """
    def __init__(self, *args, **kwargs):
        pass

    def split(self,
              data: Union[str, List[str]],
              compile: bool = True,
              postprocess: bool = True,
              show_progress: Optional[bool] = False,
              *args,
              **kwargs
              ) -> Union[List[List[str]], List[str]]:
        """
        For a string or a list of strings, return exactly one chunk per string.
        Catches all kinds of arguments but does not use them.

        Args:
            data: Union[str, List[str]]: Text or list of texts to split into
                chunks.
            compile: bool: Compile chunks into a single string if True.
            postprocess: bool: Postprocess chunks if True.
            show_progress: bool: Show progress bar if True and input data is
                in list format.
            **kwargs: Additional keyword arguments to be caught and ignored.

        Returns:
            Union[List[List[str]], List[str]]: List of chunks as strings or a
                list of lists of chunks as strings with exactly one string per
                input string.
        """
        if isinstance(data, str):
            # wrap to ensure that the segmenter receives a list
            chunks = self._split([data])
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
                chunks = self._split(data, show_progress)
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
               **kwargs
               ) -> List[List[List[str]]]:
        """
        Takes a list of strings and returns a nested list with each string
        wrapped in two layers of lists (mirroring lists of chunks which
        themselves contain lists of strings).

        Args:
            data: List[str]: List of strings to split to dummy split into chunks.
            show_progress: bool: Show progress bar if True.
            kwargs: Additional keyword arguments for the chunker to be caught
                and ignored.

        Returns:
            List[List[List[str]]]: List of chunks as lists of strings with
                exactly one string per chunk.
        """

        if show_progress:
            iterator = tqdm(data, desc="Creating Dummy Chunks")
        else:
            iterator = data
        # for each text, simply create a list of chunks with one chunk
        # containing a list of sentences in the chunk with exactly one sentence
        chunks = [[[d]] for d in iterator]
        return chunks

    def _compile_chunks(self,
                        chunks: Union[List[List[str]], List[List[List[str]]]],
                        ensure_separators: bool = False
                        ) -> Union[List[str], List[List[str]]]:
        """
        For nested lists of depth 2 or 3, peel away exactly one layer of lists.

        Args:
            chunks (Union[List[List[str]], List[List[List[str]]]): List of
                chunks as lists of strings or a list of lists of chunks as lists
                of ensure_separators: bool: Argument to be caught and ignored.

        Returns:
            Union[List[str], List[List[str]]]: List of chunks as strings or a
                list of lists of chunks as strings with exactly one string per
                chunk.
        """
        return [inner_list for outer_list in chunks
                for inner_list in outer_list]

    def _postprocess(self,
                     data: List[List[str]],
                     **kwargs
                     ) -> List[List[str]]:
        """
        Return data as received and ignore any additional arguments.

        Args:
            data (List[List[str]]): List of chunks as lists of strings.
            **kwargs: Additional keyword arguments to be caught and ignored.

        Returns:
            List[List[str]]: List of chunks as lists of strings.
        """
        return data


class EmbeddingChunker:
    """
    Chunker class that wraps around different embedding-based chunk segmenters.
    Takes a text or alternatively a lists of texts and returns a list of chunks
    per text either in a flat list or a list of lists. Initialized with the name
    of a built-in chunker and a corresponding set of parameters. Alternatively
    initiated with a custom callable that implements the EmbeddingBackendProtocol
    and a transformer model name.

    Args:
        model (Optional[str]): transformer model as a string or an instance of
            EmbeddingModel or SentenceTransformer. If a string, it must refer to
            a valid model from Hugging Face.
        specs (Optional[Dict[str, Any]]): Specifications for the chunker and
            internal segmenters. Must include:
            - "chunker": Name of the chunker to be used. Can be either a string
                referring to a built-in chunker or a custom callable implementing
                the EmbeddingChunkerProtocol.
            - "chunker_specs": Additional specifications to be passed when
                instantiating the chunker.
            - sent_specs: Specifications for the internal SentenceHandler.
            - para_specs: Specifications for the internal ParagraphHandler.
    """

    def __init__(
            self,
            model: Union[str, EmbeddingModel, SentenceTransformer],
            specs: Optional[Dict[str, Any]] = None
            ):
        self.model = self._load_model(model)
        self.tokenizer = self.model.tokenizer


        specs = specs or {}

        # Load internal SentenceModule
        sentencizer = specs.get("sentencizer", None)
        para_specs = {"paragrapher": specs.get("paragrapher", "clean"),
                      "drop_placeholders": specs.get("drop_placeholders", [])}
        self.sentencizer = SentenceHandler(sentencizer, para_specs)

        # Load working parameters
        chunker = specs.get("chunker", "linear")
        chunker_specs = specs.get("chunker_specs", {})
        chunker_specs["length_metric"] = self._calculate_length

        self._chunker = self._load_chunker(chunker, chunker_specs)

    def split(self,
              data: Union[str, List[str]],
              compile: bool = True,
              postprocess: bool = True,
              show_progress: Optional[bool] = False,
              **kwargs
              ) -> Union[List[List[str]], List[str]]:
        """
        Return chunks for a single string or a list of strings as one list of
        strings per input string.

        Args:
            data (Union[str, List[str]]): Text or list of texts to split into
                chunks.
            compile (bool): Compile chunks into a single string if True.
            postprocess (bool): Postprocess chunks if True.
            show_progress (bool): Show progress bar if True and input data is
                of list format.
            **kwargs: Additional keyword arguments for the internal chunker.

        Returns:
            Union[List[List[str]], List[str]]: List of chunks as strings or a
                list of lists of chunks as strings.
        """

        ensure_separators = kwargs.pop("ensure_separators", False)

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
               ) -> List[List[List[str]]]:
        """
        Private function handling the core splitting logic. Splits text data
        into sentences with the internal sentencizer, creates embeddings for
        each sentence and groups sentences into chunks based upon the internal
        chunker.

        Args:
            data (List[str]): List of strings to split to split into chunks.
            show_progress (bool): Show progress bar if True.
            kwargs: Additional keyword arguments for the chunker.

        Returns:
            List[List[List[str]]]: List of chunks as lists of strings.
        """

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

    def _create_embeddings(
            self,
            sentences: Union[List[str], List[List[str]]],
            show_progress: bool = False
            ) -> Union[List[NDArray], List[List[NDArray]]]:
        """
        Create embeddings for each string in data based upon the internal
        transformer model.

        Args:
            sentences (Union[List[str], List[List[str]]]): List or list of lists
                of sentences.
            show_progress (bool): Show progress bar if True.

        Returns:
            Union[List[NDArray], List[List[NDArray]]]: List or list of lists
                of embeddings with one embedding per sentence.
        """
        if show_progress:
            iterator = tqdm(sentences, desc="Creating embeddings")
        else:
            iterator = sentences
        return [self.model.encode(sent_list, show_progress_bar=False)
                for sent_list in iterator]

    def _compile_chunks(self,
                        chunks: Union[List[List[str]], List[List[List[str]]]],
                        ensure_separators: bool = False
                        ) -> Union[List[str], List[List[str]]]:
        """
        Compile chunks received as individual sentences into single strings. To
        avoid merging paragraphs (originally separated by line breaks) into
        unstructured word salad, a separator full stop can be inserted
        optionally.

        Args:
            chunks (Union[List[List[str]], List[List[List[str]]]]): Chunks as
                lists of sentences per chunk.
            ensure_separators (bool): Insert a full stop at the end of each
                sentence that not clearly ends in sentence-ending punctuation

        Returns:
            Union[List[str], List[List[str]]]: Chunks as single strings.
        """
        depth = uniform_depth(chunks)
        if depth == 2:
            return [make_text_from_chunk(c, ensure_separators=ensure_separators)
                    for c in chunks]
        if depth == 3:
            return [[make_text_from_chunk(c, ensure_separators=ensure_separators)
                     for c in chunk_list]
                    for chunk_list in chunks]

    def _postprocess(self,
                     chunks: (Union[List[str], List[List[str]],
                              List[List[List[str]]]])
                     ) -> Union[List[str], List[List[str]],
                                List[List[List[str]]]]:
        """
        Clear away irregularities in the sentence lists produced by different
        sentence segmenters.

        Removes leading and trailing whitespace and empty strings in the current
        implementation. (Might be modified in the future to accept kwargs for
        different postprocessing steps.)

        Args:
            chunks (Union[List[str], List[List[str]], List[List[List[str]]]]):
                Compiled or uncompiled chunks (lists of sentences or single
                strings per chunk)

        Returns:
            Union[List[str], List[List[str]], List[List[List[str]]]]: Chunks in
                the same basic structure as input chunks but with empty chunks
                and trailing or leading whitespace removed.
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

    def _calculate_length(
            self,
            sentence: str
            ) -> int:
        """
        Use the internal model's tokenizer to calculate the length of a sentence
        as the number of its tokens.

        Note: Tokens in this implementation are tokens as perceived by
        transformer models which don't translate into tokens in the traditional
        NLP sense (where one word is one token). The logic behind this choice
        is to enable generation of chunks which fit into the embedding length
        of or are of equal length with regard to transformer models.

        Args:
            sentence (str): Sentence to calculate length of.

        Returns:
            int: The number of tokens in the sentence.
        """
        tokens = self.tokenizer(sentence)
        return len(tokens["input_ids"])

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

    def _load_chunker(self,
                      chunker: Union[str, EmbeddingChunkerProtocol],
                      chunker_specs: Dict[str, Any]
                      ) -> EmbeddingChunkerProtocol:
        """
        Parse chunker specifications to create the internal chunker object that
        handles the compilation of sentences and embeddings into chunks. Can
        parse either a string specifying a built-in chunker or a custom callable
        that implements the EmbeddingChunker protocol.

        Args:
            chunker (Union[str, EmbeddingChunkerProtocol]): Chunker to be
                loaded.
            chunker_specs (Dict[str, Any]): Additional specifications to be
                passed at initialization to the chunker.

        Returns:
            EmbeddingChunkerProtocol: Callable that implements the
                EmbeddingChunkerProtocoll.
        """
        map = CHUNK_BACKENDS_MAP["embedding"]
        if isinstance(chunker, str):
            if chunker not in map:
                raise ValueError(f"Invalid segmenter '{chunker}'. "
                                 f"Must be in: {list(map.keys())}.")
            return map[chunker](**chunker_specs)
        elif callable(chunker):
            return chunker(**chunker_specs)
        else:
            raise ValueError("Chunker must be a string or callable. Custom "
                             "callables must implement the "
                             "EmbeddingChunkerProtocol.")


class SimpleChunker:
    """
    Chunker class that wraps around different chunk segmenters that work without
    sentence embeddings.seTakes a text or alternatively a lists of texts and
    returns a list of chunks per text either in a flat list or a list of lists.
    Initialized with the name of a built-in chunker and a corresponding set of
    parameters. Alternatively initiated with a custom callable that implements
    the SimpleChunkerProtocol.

    Args:
        specs (Optional[Dict[str, Any]]): Specifications for the chunker and
            internal segmenters. Must include:
            - "chunker": Name of the chunker to be used. Can be either a string
                referring to a built-in chunker or a custom callable implementing
                the EmbeddingChunkerProtocol.
            - "chunker_specs": Additional specifications to be passed when
                instantiating the chunker.
            - sent_specs: Specifications for the internal SentenceHandler.
            - para_specs: Specifications for the internal ParagraphHandler.
    """

    def __init__(
            self,
            specs: Optional[Dict[str, Any]] = None
            ):

        specs = specs or {}

        # Load internal SentenceModule
        sent_specs = {"sentencizer": specs.get("sentencizer", ("pysbd", "de")),
                      "show_progress": False}
        para_specs = {"paragrapher": specs.get("paragrapher", "clean"),
                      "drop_placeholders": specs.get("drop_placeholders", [])}
        self.sentencizer = SentenceHandler(sent_specs, para_specs)

        # Load working parameters
        chunker = specs.get("chunker", "sliding")
        chunker_specs = specs.get("chunker_specs", {})

        self._chunker = self._load_chunker(chunker, chunker_specs)

    def split(self,
              data: Union[str, List[str]],
              compile: bool = True,
              postprocess: bool = True,
              show_progress: Optional[bool] = False,
              **kwargs
              ) -> Union[List[List[str]], List[str]]:
        """
        Return chunks for a single string or a list of strings as one list of
        strings per input string.

        Args:
            data (Union[str, List[str]]): Text or list of texts to split into
                chunks.
            compile (bool): Compile chunks into a single string if True.
            postprocess (bool): Postprocess chunks if True.
            show_progress (bool): Show progress bar if True and input data is
                of list format.
            **kwargs: Additional keyword arguments for the internal chunker.

        Returns:
            Union[List[List[str]], List[str]]: List of chunks as strings or a
                list of lists of chunks as strings.
        """

        ensure_separators = kwargs.pop("ensure_separators", False)

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
               ) -> List[List[List[str]]]:
        """
        Private function handling the core splitting logic. Splits text data
        into sentences with the internal sentencizer, creates embeddings for
        each sentence and groups sentences into chunks based upon the internal
        chunker.

        Args:
            data (List[str]): List of strings to split to split into chunks.
            show_progress (bool): Show progress bar if True.
            kwargs: Additional keyword arguments for the chunker.

        Returns:
            List[List[List[str]]]: List of chunks as lists of strings.
        """

        drop_placeholders = kwargs.pop("drop_placeholders", [])

        sentences = self.sentencizer.split_list(
            data, drop_placeholders=drop_placeholders,
            show_progress=show_progress)

        if show_progress:
            chunks = [self._chunker(sentences=s,
                                    **kwargs)
                      for s in tqdm((sentences),
                                       desc="Chunking sentences")
                      ]
        else:
            chunks = [self._chunker(sentences=s,
                                    **kwargs)
                      for s in sentences
                      ]

        return chunks

    def _compile_chunks(self,
                        chunks: Union[List[List[str]], List[List[List[str]]]],
                        ensure_separators: bool = False
                        ) -> Union[List[str], List[List[str]]]:
        """
        Compile chunks received as individual sentences into single strings. To
        avoid merging paragraphs (originally separated by line breaks) into
        unstructured word salad, a separator full stop can be inserted
        optionally.

        Args:
            chunks (Union[List[List[str]], List[List[List[str]]]]): Chunks as
                lists of sentences per chunk.
            ensure_separators (bool): Insert a full stop at the end of each
                sentence that not clearly ends in sentence-ending punctuation

        Returns:
            Union[List[str], List[List[str]]]: Chunks as single strings.
        """
        depth = uniform_depth(chunks)
        if depth == 2:
            return [make_text_from_chunk(c, ensure_separators=ensure_separators)
                    for c in chunks]
        if depth == 3:
            return [[make_text_from_chunk(c, ensure_separators=ensure_separators)
                     for c in chunk_list]
                    for chunk_list in chunks]

    def _postprocess(self,
                     chunks: (Union[List[str], List[List[str]],
                              List[List[List[str]]]])
                     ) -> Union[List[str], List[List[str]],
                                List[List[List[str]]]]:
        """
        Clear away irregularities in the sentence lists produced by different
        sentence segmenters.

        Removes leading and trailing whitespace and empty strings in the current
        implementation. (Might be modified in the future to accept kwargs for
        different postprocessing steps.)

        Args:
            chunks (Union[List[str], List[List[str]], List[List[List[str]]]]):
                Compiled or uncompiled chunks (lists of sentences or single
                strings per chunk)

        Returns:
            Union[List[str], List[List[str]], List[List[List[str]]]]: Chunks in
                the same basic structure as input chunks but with empty chunks
                and trailing or leading whitespace removed.
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

    def _load_chunker(self,
                      chunker: Union[str, SimpleChunkerProtocol],
                      chunker_specs: Dict[str, Any]
                      ) -> SimpleChunkerProtocol:
        """
        Parse chunker specifications to create the internal chunker object that
        handles the compilation of sentences and embeddings into chunks. Can
        parse either a string specifying a built-in chunker or a custom callable
        that implements the EmbeddingChunker protocol.

        Args:
            chunker (Union[str, SimpleChunkerProtocol]): Chunker to be
                loaded.
            chunker_specs (Dict[str, Any]): Additional specifications to be
                passed at initialization to the chunker.

        Returns:
            EmbeddingChunkerProtocol: Callable that implements the
                EmbeddingChunkerProtocoll.
        """
        map = CHUNK_BACKENDS_MAP["simple"]
        if isinstance(chunker, str):
            if chunker not in map:
                raise ValueError(f"Invalid segmenter '{chunker}'. "
                                 f"Must be in: {list(map.keys())}.")
            return map[chunker](**chunker_specs)
        elif callable(chunker):
            return chunker(**chunker_specs)
        else:
            raise ValueError("Chunker must be a string or callable. Custom "
                             "callables must implement the "
                             "SimpleChunkerProtocol.")
