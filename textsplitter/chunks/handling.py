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

from .backends import EmbeddingChunkerProtocol, SimpleChunkerProtocol, CHUNK_BACKENDS_MAP
from .chunker import (DummyChunker, ChunkerProtocol, EmbeddingChunker,
                      SimpleChunker)
from .utils import make_indices_from_chunk
from textsplitter.utils import add_id
from ..dataframes import columns, cast_to_df
from ..paragraphs import ParaSegmenterProtocol
from ..sentences import SentSegmenterProtocol

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()

class ChunkHandler:
    def __init__(
            self,
            chunk_backend: Optional[
                Union[
                    SimpleChunkerProtocol,
                    EmbeddingChunkerProtocol,
                ]
            ] = None,
            sent_backend: Optional[SentSegmenterProtocol] = None,
            para_backend: Optional[ParaSegmenterProtocol] = None,
    ):
        """
        Constructor merges spec dictionaries, parses them,
        then loads the appropriate chunker.
        """
        self.splitter = self._load_chunker(
            chunk_backend, sent_backend, para_backend)

    @ staticmethod
    def _load_chunker(
            chunk_backend: Optional[
                Union[
                    SimpleChunkerProtocol,
                    EmbeddingChunkerProtocol,
                ]
            ] = None,
            sent_backend: Optional[SentSegmenterProtocol] = None,
            para_backend: Optional[ParaSegmenterProtocol] = None,
    ) -> Union[EmbeddingChunker, SimpleChunker, DummyChunker]:
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
        if chunk_backend is None:
            print("No chunker specified. Using dummy chunker.")
            return DummyChunker()
        else:
            chunker_type = getattr(chunk_backend, "chunker_type", None)
            chunker_class = CHUNKER_REGISTRY.get(chunker_type, None)
            if chunker_class is None:
                raise ValueError(
                    f"Unknown chunker type: {chunker_type}. "
                    "Available chunkers types are: "
                    f"{list(CHUNKER_REGISTRY.keys())}."
                )
            return chunker_class(chunk_backend, sent_backend, para_backend)

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
            chunks = self.splitter.split(text,
                                         compile=False,
                                         postprocess=False,
                                         **kwargs)
            # prune for empty chunks
            chunks = [c for c in chunks if c]
            indices = [make_indices_from_chunk(c, text) for c in chunks]
            chunks = self.splitter._compile_chunks(
                chunks, ensure_separators=ensure_separators)
            chunks = self.splitter._postprocess(chunks)
            chunks = list(zip(indices, chunks))
        else:
            chunks = self.splitter.split(text,
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
            chunks = self.splitter.split(texts,
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

            chunks = self.splitter._compile_chunks(
                chunks, ensure_separators=ensure_separators)
            chunks = self.splitter._postprocess(chunks)
            chunks = [list(zip(index_list, chunk_list))
                      for index_list, chunk_list in zip(indices, chunks)]
        else:
            chunks = self.splitter.split(texts,
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
    "embedding": EmbeddingChunker,
    "simple": SimpleChunker,
}