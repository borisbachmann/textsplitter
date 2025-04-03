"""
Module for handling chunking of text data. The main class ChunkHandler handles
chunking of different data types (strings, lists of strings, pandas Series and
DataFrames). It provides an interface that wraps around an internal Chunker
object that handles the actual chunking process and ensures that the output is
formatted correctly.

If no appropriate chunking specs are passed, the ChunkHandler will default to
using a DummyChunker that simply splits the input data into chunks of a fixed
size. The ChunkHandler can also be initialized with a custom chunker.
"""
from typing import Dict, Any, Union, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from .backends import EmbeddingChunkerProtocol, CHUNK_BACKENDS_MAP
from .chunker import (DummyChunker, ChunkerProtocol, EmbeddingChunker,
                      SimpleChunker)
from .utils import make_indices_from_chunk
from textsplitter.utils import add_id
from ..dataframes import columns, cast_to_df

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()

class ChunkHandler:
    def __init__(
            self,
            chunk_specs: Optional[Dict[str, Any]] = None,
            para_specs: Optional[Dict[str, Any]] = None,
            sent_specs: Optional[Dict[str, Any]] = None):
        """
        Constructor merges spec dictionaries, parses them,
        then loads the appropriate chunker.
        """
        parsed_specs = self._parse_specs(chunk_specs, para_specs, sent_specs)
        self.chunker = self._load_chunker(*parsed_specs)

    @ staticmethod
    def _parse_specs(chunk_specs: Optional[Dict[str, Any]] = None,
                     para_specs: Optional[Dict[str, Any]] = None,
                     sent_specs: Optional[Dict[str, Any]] = None
                     ) -> Tuple[str, Optional[str], Dict[str, Any],
                                Dict[str, Any]]:
        """
        Merge all specs and extract required parameters for
        initialization of the chunker.

        Args:
            chunk_specs (Optional[Dict[str, Any]]): Chunker specs
            para_specs (Optional[Dict[str, Any]]): Paragrapher specs
            sent_specs (Optional[Dict[str, Any]]): Sentencizer specs

        Returns:
            Tuple[str, Optional[str], Dict[str, Any], Dict[str, Any]]:
                chunker, chunker_type, required_params, remaining_specs for
                chunker initialization
        """
        # Handle missing specs
        chunk_specs = chunk_specs or {}
        para_specs = para_specs or {}
        sent_specs = sent_specs or {}

        chunker = chunk_specs.get("chunker", None)
        chunker_type = ChunkHandler._determine_chunker_type(chunk_specs)
        # if chunker_type:
        #     chunker_map = CHUNKER_REGISTRY[chunker_type]["chunkers"]

        required = (CHUNKER_REGISTRY[chunker_type]["required_params"]
                    if chunker_type else [])

        if not chunker and not chunker_type:
            print("No valid chunker specified. Using dummy chunker.")
            chunker = "dummy"
        elif not chunker and chunker_type:
            chunker = CHUNKER_REGISTRY[chunker_type]["default_backend"]
            print(
                f"No chunker specified. Defaulting to '{chunker}' with "
                f"specified parameters."
            )
        # catch invalid chunkers
        elif isinstance(chunker, str) and chunker_type:
            # verify call to built-in chunkers
            chunker_map = CHUNKER_REGISTRY.get(
                chunker_type, CHUNKER_REGISTRY["simple"]
            )["chunkers"]
            if chunker not in chunker_map:
                raise ValueError(
                    f"Invalid chunker specified: '{chunker}' is not a "
                    f"built-in type."
                )
        elif isinstance(chunker, str) and not chunker_type:
            # if chunker is valid but chunker type could not be
            # determined, either chunker type does not require parameters  or
            # some have to be missing
            for current_type, config in CHUNKER_REGISTRY.items():
                if chunker in config["chunkers"]:
                    missing_params = [
                        param for param in
                        config["required_params"]
                        if param not in chunk_specs
                        ]
                    if missing_params:
                        raise ValueError(
                            f"Chunker type '{current_type}' specified but "
                            f"missing required parameters: '{missing_params}'."
                        )
                    chunker_type = current_type
        elif chunker is not None and not callable(chunker):
            raise ValueError(
                "Chunker must be a string or callable implementing the "
                "ChunkerProtocol."
            )

        # extract required parameters from dict:
        required_params = {key: chunk_specs.pop(key) for key in required}

        # merge para_specs and sent_specs into remaining_specs
        remaining_specs = chunk_specs | para_specs | sent_specs

        return chunker, chunker_type, required_params, remaining_specs

    @staticmethod
    def _determine_chunker_type(
            specs: Dict[str, Any]
            ) -> Optional[str]:
        """Try to determine the type of chunker to use based on specs.

        Args:
            specs (Dict[str, Any]): Chunker specs

        Return:
            Optional[str]: Chunker type if derivable from dict, else None
        """
        for chunker_type, config in CHUNKER_REGISTRY.items():
            if config["required_params"]:
                if all(param in specs for param in config["required_params"]):
                    return chunker_type

    @ staticmethod
    def _load_chunker(chunker: Union[str, ChunkerProtocol,
                                     EmbeddingChunkerProtocol],
                      chunker_type: Optional[str],
                      required_specs: Dict[str, Any],
                      remaining_specs: Dict[str, Any]
                      ) -> Union[EmbeddingChunker, DummyChunker, callable]:
        """
        Based upon specs load the appropriate chunker with the correct
        parameters.

        Args:
            chunker (Union[str, ChunkerProtocol, EmbeddingChunkerProtocol]):
                Chunker to load
            chunker_type (Optional[str]): Type of chunker if known chunker class
            required_specs (Dict[str, Any]): Required parameters for known
                chunker classes
            remaining_specs (Dict[str, Any]): Remaining parameters for chunker
        """
        if chunker == "dummy":
            return DummyChunker()
        if isinstance(chunker, str):
            chunker_class = CHUNKER_REGISTRY[chunker_type]["class"]
            return chunker_class(*required_specs.values(), remaining_specs)
        elif callable(chunker):
            if chunker_type is not None:
                # if a chunker type could be determined, use the callable as
                # backend and wrap it in appropriate class
                chunker_class = CHUNKER_REGISTRY[chunker_type]["class"]
                return chunker_class(*required_specs.values(), remaining_specs)
            if chunker_type is None:
                # if no chunker type could be determined, the use the callable
                # directly
                return callable(remaining_specs)

    def split(self,
              text: str,
              as_tuples: bool = False,
              include_span: bool = False,
              **kwargs
              ) -> list:
        """Split a string containing natural language data into chunks. Returns
        a list of chunks as strings. Optionally, return a list of tuples also
        including chunk index and/or start and end indices of chunks in the
        original text.

        Args:
            text (str): Text to split into chunks
            as_tuples (bool): Return chunks as tuples with index and text
            include_span (bool): Include start and end indices of chunks
            kwargs (dict): Additional arguments for chunker

        Returns:
            list: List of chunks as strings or tuples
        """
        if include_span:
            ensure_separators = kwargs.pop("ensure_separators", False)
            chunks = self.chunker.split(text,
                                        compile=False,
                                        postprocess=False,
                                        **kwargs)
            # prune for empty chunks
            chunks = [c for c in chunks if c]
            indices = [make_indices_from_chunk(c, text) for c in chunks]
            chunks = self.chunker._compile_chunks(
                chunks, ensure_separators=ensure_separators)
            chunks = self.chunker._postprocess(chunks)
            chunks = list(zip(indices, chunks))
        else:
            chunks = self.chunker.split(text,
                                        compile=True,
                                        postprocess=True,
                                        **kwargs)

        if as_tuples:
            chunks = add_id(chunks)

        return chunks

    def split_list(self,
                   texts: list,
                   as_tuples: bool = False,
                   include_span: bool = False,
                   **kwargs
                   ) -> list:
        """
        Split a list of strings containing natural language data into chunks.
        Returns a list of chunks per text as lists of strings. Optionally,
        returns lists of tuples also including chunk index and/or start and end
        indices of chunks in the original text.

        Args:
            texts (list): List of texts to split into chunks
            as_tuples (bool): Return chunks as tuples with index and text
            include_span (bool): Include start and end indices of chunks
            kwargs (dict): Additional arguments for chunker

        Returns:
            Union[List[List[str]], List[List[Tuple[int, str]]],
                List[List[Tuple[Tuple[int, int], str]]]]: List of chunks
                per text as list of strings or tuples including paragraph ids
                and span indices.
        """
        show_progress = kwargs.get("show_progress", False)

        if include_span:
            ensure_separators = kwargs.pop("ensure_separators", False)
            chunks = self.chunker.split(texts,
                                        compile=False,
                                        postprocess=False,
                                        **kwargs)
            if show_progress:
                iterator = tqdm(zip(texts, chunks),
                                desc="Adding span indices",
                                total=len(texts))
            else:
                iterator = zip(texts, chunks)

            indices = [[make_indices_from_chunk(c, text) for c in chunk_list]
                       for text, chunk_list in iterator]

            chunks = self.chunker._compile_chunks(
                chunks, ensure_separators=ensure_separators)
            chunks = self.chunker._postprocess(chunks)
            chunks = [list(zip(index_list, chunk_list))
                      for index_list, chunk_list in zip(indices, chunks)]
        else:
            chunks = self.chunker.split(texts,
                                        compile=True,
                                        postprocess=True,
                                        **kwargs)

        if as_tuples:
            chunks = [add_id(chunk_list) for chunk_list in chunks]

        return chunks

    def split_df(self,
                 input_df: pd.DataFrame,
                 text_column: str = columns.TEXT_COL,
                 drop_text: bool = True,
                 keep_orig: list = None,
                 mathematical_ids: bool = False,
                 include_span: bool = False,
                 **kwargs
                 ) -> pd.DataFrame:
        """
        Split texts in a pandas DataFrame column into chunks. Returns a df
        with chunks as rows, including chunk text, chunk index, and optionally
        start and end indices of chunks in the original text. Original text id
        and number of chunks in text and, optionally, text string are kept in
        the output.

        Args:
            input_df (pd.DataFrame): DataFrame containing data data
            text_column (str): Name of the column containing data data
            drop_text (bool): Whether to drop the original data column
            keep_orig (bool): List of original columns to keep in output
            mathematical_ids (bool): whether to increment chunk IDs by 1 to
                avoid 0
            include_span (bool): Include start and end indices of chunks
            kwargs (dict): Additional arguments for chunker

        Returns:
            pd.DataFrame: DataFrame with chunks as rows
        """

        texts = input_df[text_column].tolist()
        chunks = self.split_list(texts,
                                 as_tuples=True,
                                 include_span=include_span,
                                 **kwargs
                                 )

        return cast_to_df(
            input_df=input_df,
            segments=chunks,
            base_column=columns.CHUNK_COL,
            text_column=text_column,
            drop_text=drop_text,
            keep_orig=keep_orig,
            mathematical_ids=mathematical_ids,
            include_span=include_span
        )

# Registry of all chunkers types: Put all available chunkers here
CHUNKER_REGISTRY = {
"embedding": {
    "required_params": ["model"],
    "default_backend": "linear",
    "class": EmbeddingChunker,
    "chunkers": CHUNK_BACKENDS_MAP["embedding"]
    },
"simple": {
    "required_params": [],
    "default_backend": "sliding",
    "class": SimpleChunker,
    "chunkers": CHUNK_BACKENDS_MAP["simple"]
    }
}