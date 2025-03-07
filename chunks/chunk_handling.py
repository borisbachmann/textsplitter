"""
Module for handling chunking of text data. The main class ChunkSegmenter handles
chunking of different data types (strings, lists of strings, pandas Series and
DataFrames). It provides an interface that wraps around an internal Chunker
object that handles the actual chunking process and ensures that the output is
formatted correctly.

The module also includes a DummyChunkSegmenter class that is loaded if the
high-level TextSplitter class is initialized without chunking capabilities for
lightweight sentence and paragraph splitting. The DummyChunkSegmenter returns
exactly one chunk (the input text) for each input text in the otherwise same
format as the ChunkSegmenter.
"""

import pandas as pd
from tqdm.auto import tqdm

from .chunker import Chunker
from .chunk_utils import make_indices_from_chunk
from ..constants import (TEXT_COL, CHUNK_COL, CHUNKS_COL, CHUNK_N_COL,
                         CHUNK_ID_COL, CHUNK_SPAN_COL)
from ..utils import column_list, add_id

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()

class ChunkSegmenter:
    def __init__(self, chunk_specs, para_specs, sent_specs):
        model = chunk_specs.get("model")
        specs = self._compile_specs(chunk_specs, para_specs, sent_specs)
        self.chunker = Chunker(model, specs)

    def _compile_specs(self, chunk_specs, para_specs, sent_specs):
        specs = {}

        # resolve chunking specs
        specs["chunker"] = chunk_specs.get("chunker", "linear")
        chunk_specs = {k: v for k, v in chunk_specs.items()
                       if k not in ["chunker", "model"]}
        specs["chunk_specs"] = chunk_specs

        # resolve paragraphing and sentencizing specs
        if para_specs:
            specs.update(para_specs)
        if sent_specs:
            specs.update(sent_specs)

        return specs

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
            chunks = self.chunker._compile_chunks(chunks, ensure_separators)
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

            chunks = self.chunker._compile_chunks(chunks, ensure_separators)
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

        df = input_df.copy()

        df[CHUNKS_COL] = pd.Series(self.split_list(df[column].tolist(),
                                                   as_tuples=True,
                                                   include_span=include_span,
                                                   **kwargs
                                                   )
                                   )

        df = df.explode(CHUNKS_COL).reset_index(drop=True)

        # uunpack chunk data into separate columns
        chunks_df = pd.DataFrame(df[CHUNKS_COL].tolist())
        if include_span:
            df[[CHUNK_ID_COL, CHUNK_SPAN_COL, CHUNK_COL]] = chunks_df
        else:
            df[[CHUNK_ID_COL, CHUNK_COL]] = chunks_df

        # count number of chunks in each text
        df[CHUNK_N_COL] = df.groupby(column)[column].transform("size")

        if mathematical_ids:
            df[CHUNK_ID_COL] = df[CHUNK_ID_COL].map(lambda x: x + 1)

        # keep only desired columns for output dataframe
        columns = [c for c in column_list(CHUNK_COL, column) if c in df.columns]

        if drop_text:
            columns.remove(column)

        return df[columns]

class DummyChunkSegmenter:
    def __init__(self):
        pass

    def split(self,
              text: str,
              as_tuples: bool = False,
              include_span: bool = False
              ) -> list:
        chunks = [text]

        if include_span:
            indices = [(0, len(text))]
            chunks = list(zip(indices, chunks))

        if as_tuples:
            chunks = add_id(chunks)

        return chunks

    def split_df(self,
                 input_df: pd.DataFrame,
                 column: str = TEXT_COL,
                 drop_text: bool = True,
                 mathematical_ids: bool = False,
                 include_span: bool = False
                 ) -> pd.DataFrame:
        df = input_df.copy()
        df[CHUNK_ID_COL] = 0
        df[CHUNK_COL] = df[column]
        if include_span:
            df[CHUNK_SPAN_COL] = (0, len(df[column]))
        df[CHUNK_N_COL] = 1

        if mathematical_ids:
            df[CHUNK_ID_COL] = df[CHUNK_ID_COL].map(lambda x: x + 1)

        columns = [c for c in column_list(CHUNK_COL, column) if c in df.columns]

        if drop_text:
            columns.remove(column)

        return df[columns]
