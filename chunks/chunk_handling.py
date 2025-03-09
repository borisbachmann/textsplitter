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
from typing import Dict, Any, Union, Optional

import pandas as pd
from tqdm.auto import tqdm

from .chunker import EmbeddingChunker, DummyChunker
from .utils import make_indices_from_chunk
from ..constants import TEXT_COL, CHUNK_COL
from ..utils import add_id, cast_to_df

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()

class ChunkHandler:
    def __init__(self, chunk_specs, para_specs, sent_specs):
        specs = self._compile_specs(chunk_specs, para_specs, sent_specs)
        self.chunker = self._load_chunker(specs)

    def _compile_specs(self,
                       chunk_specs: Optional[Dict[str, Any]] = None,
                       para_specs: Optional[Dict[str, Any]] = None,
                       sent_specs: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
        chunk_specs = chunk_specs or {}
        para_specs = para_specs or {}
        sent_specs = sent_specs or {}

        # merge all specs
        specs = chunk_specs | para_specs | sent_specs

        # ensure default chunker is set
        if not callable(specs.get("chunker", None)):
            if "model" in specs:
                print("No chunker specified. Using linear chunker.")
                specs.setdefault("chunker", "linear")
            else:
                print("No chunker or model specified. Using dummy chunker.")
                specs.setdefault("chunker", "dummy")

        return specs

    def _load_chunker(self,
                      specs: Dict[str, Any]
                      ) -> Union[EmbeddingChunker, DummyChunker, callable]:
        """
        Based upon specs load the appropriate chunker with the correct
        parameters.
        """
        # Loading callable is not implemented yet – resolve how to handle para
        # and sent specs in specs
        if callable(specs["chunker"]):
            return specs["chunker"](specs)
        elif specs["chunker"] == "dummy":
            return DummyChunker()
        else:
            return EmbeddingChunker(specs)

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
            text: str: Text to split into chunks
            as_tuples: bool: Return chunks as tuples with index and text
            include_span: bool: Include start and end indices of chunks
            kwargs: dict: Additional arguments for chunker

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
        """Split a list of strings containing natural language data into chunks.
        Returns a list of chunks per text as lists of strings. Optionally,
        returns lists of tuples also including chunk index and/or start and end
        indices of chunks in the original text.

        Args:
            texts: list: List of texts to split into chunks
            as_tuples: bool: Return chunks as tuples with index and text
            include_span: bool: Include start and end indices of chunks
            kwargs: dict: Additional arguments for chunker

        Returns:
            list: List of chunks as lists strings or tuples
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
                 column: str = TEXT_COL,
                 drop_text: bool = True,
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
            input_df: pd.DataFrame containing data data
            column: name of the column containing data data
            drop_text: whether to drop the original data column
            mathematical_ids: whether to increment chunk IDs by 1 to avoid 0
            include_span: whether to include start and end indices of chunks
            kwargs: dict: Additional arguments for chunker

        Returns:
            pd.DataFrame: DataFrame with chunks as rows
        """
        texts = input_df[column].tolist()
        chunks = self.split_list(texts,
                                 as_tuples=True,
                                 include_span=include_span,
                                 **kwargs
                                 )

        return cast_to_df(
            input_df=input_df,
            segments=chunks,
            base_column=CHUNK_COL,
            text_column=column,
            drop_text=drop_text,
            mathematical_ids=mathematical_ids,
            include_span=include_span
        )
