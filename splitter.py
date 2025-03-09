"""
This module contains the main class TextSplitter, which is a high-level wrapper
class around different text-splitting techniques (sentences, paragraphs, and
chunks) represented by corresponding module classes. It provides a unified
interface that digests input data and distributes it to the underlying module
methods. When initialized, it takes specs for each segment type and creates
instances of the corresponding module classes accordingly.

Currently, three types of segmenters are supported:
- Sentences: handled internally by a SentenceModule instance
- Paragraphs: handled internally by a ParagraphModule instance
- Chunks: handled internally by a ChunkModule instance
"""

from typing import Optional, Union, List

import pandas as pd

from tqdm.auto import tqdm

from .dataframes.columns import TEXT_COL
from .chunks.chunk_handling import ChunkHandler
from .paragraphs.para_handling import ParagraphHandler
from .sentences.sent_handling import SentenceHandler

# register pandas
tqdm.pandas()

class TextSplitter:
    """
    Class to split text into sentences, paragraphs, and chunks. TextSplitter
    processes text data in different formats and returns the split text in a
    corresponding format: Lists for strings and lists of strings, pandas Series
    for pandas Series, and pandas DataFrames for pandas DataFrames.

    A high-level wrapper class around different text-splitting techniques
    (sentences, paragraphs, and chunks). It maintains instances of sentence,
    paragraph, and chunk segmenters and provides a unified interface to them.

    Args:
        sentence_specs (Optional[dict]): Specifications for the sentence
            segmenter. See SentenceModule for details.
        paragraph_specs (Optional[dict]): Specifications for the paragraph
            segmenter. See ParagraphModule for details.
        chunking_specs (Optional[dict]): Specifications for the chunk segmenter.
            See ChunkModule for details.
    """
    def __init__(
            self,
            sentence_specs: Optional[dict] = None,
            paragraph_specs: Optional[dict] = None,
            chunking_specs: Optional[dict] = None,
            ):
        # initialize paragraphing attributes
        self.paragrapher = ParagraphHandler(paragraph_specs)

        # initialize sentencizing attributes
        self.sentencizer = SentenceHandler(sentence_specs, paragraph_specs)

        # initialize chunking attributes
        self.chunker = ChunkHandler(chunking_specs, paragraph_specs,
                                    sentence_specs)

        self.processors = {
            "sentences": self.sentencizer,
            "paragraphs": self.paragrapher,
            "chunks": self.chunker
        }

    # present for backwards compatibility. Use explicit methods instead.
    def __call__(
            self,
            data: Union[str, list, pd.Series, pd.DataFrame],
            mode: str,
            column: Optional[str] = TEXT_COL,
            as_tuples: Optional[bool] = False,
            include_span: Optional[bool] = False,
            mathematical_ids: Optional[bool] = False,
            drop_text: Optional[bool] = True,
            **kwargs
            ) -> Union[List[str], List[List[str]], pd.Series, pd.DataFrame]:
        """
        Split data with the appropriate segmenter module based on mode.

        Args:
            data (Union[str, list, pd.Series, pd.DataFrame]): Text data to
                split.
            mode (str): Type of segment unit to split the text into.
            column (Optional[str]): Column name if data is a DataFrame
                (default is specified by TEXT_COL, currently "text").
            as_tuples (Optional[bool]): Whether to return tuples with id and
                text (only for data other than DataFrame, default False).
            include_span (Optional[bool]): Include span information if True
                (default False).
            mathematical_ids (Optional[bool]): Increment IDs by 1 (starting at 1
                rather than 0) if True (default False). Applies only when
                as_tuples is True or data is a DataFrame.
            drop_text (Optional[bool]): In a dataframe, drop the original text
                column if True (default True)
            **kwargs: Additional keyword arguments for the segmenter.

        Returns:
            Union[str, list, pd.Series, pd.DataFrame]: Data split into segments
                based on the mode.

        Raises:
            ValueError: If mode is not supported.
            TypeError: If data type is not supported.
        """

        return self._process_data(data=data, mode=mode, column=column,
                                  as_tuples=as_tuples, include_span=include_span,
                                  mathematical_ids=mathematical_ids,
                                  drop_text=drop_text, **kwargs)

    def sentences(self,
                  data: Union[str, list, pd.Series, pd.DataFrame],
                  column: Optional[str] = TEXT_COL,
                  as_tuples: Optional[bool] = False,
                  include_span: Optional[bool] = False,
                  mathematical_ids: Optional[bool] = False,
                  drop_text: Optional[bool] = True,
                  **kwargs
                  ) -> Union[List[str], List[List[str]], pd.Series, pd.DataFrame]:
        """
        Split text into sentences.

        Args:
            data (Union[str, list, pd.Series, pd.DataFrame]): Text data to
                split.
            column (Optional[str]): Column name if data is a DataFrame
                (default is specified by TEXT_COL, currently "text").
            as_tuples (Optional[bool]): Whether to return tuples with id and
                text (only for data other than DataFrame, default False).
            include_span (Optional[bool]): Include span information if True
                (default False).
            mathematical_ids (Optional[bool]): Increment IDs by 1 (starting at 1
                rather than 0) if True (default False). Applies only when
                as_tuples is True or data is a DataFrame.
            drop_text (Optional[bool]): In a dataframe, drop the original text
                column if True (default True)
            **kwargs: Additional keyword arguments for the sentence segmenter.

        Returns:
            Union[str, list, pd.Series, pd.DataFrame]: Data split into sentences.

        Raises:
            ValueError: If mode is not supported.
            TypeError: If data type is not supported.
        """
        return self._process_data(data=data, mode="sentences", column=column,
                                  as_tuples=as_tuples, include_span=include_span,
                                  mathematical_ids=mathematical_ids,
                                  drop_text=drop_text, **kwargs)

    def paragraphs(self,
                   data: Union[str, list, pd.Series, pd.DataFrame],
                   column: Optional[str] = TEXT_COL,
                   as_tuples: Optional[bool] = False,
                   include_span: Optional[bool] = False,
                   mathematical_ids: Optional[bool] = False,
                   drop_text: Optional[bool] = True,
                   **kwargs
                   ) -> Union[List[str], List[List[str]], pd.Series,
                              pd.DataFrame]:
        """
        Split text into paragraphs.

        Args:
            data (Union[str, list, pd.Series, pd.DataFrame]): Text data to
                split.
            column (Optional[str]): Column name if data is a DataFrame
                (default is specified by TEXT_COL, currently "text").
            as_tuples (Optional[bool]): Whether to return tuples with id and
                text (only for data other than DataFrame, default False).
            include_span (Optional[bool]): Include span information if True
                (default False).
            mathematical_ids (Optional[bool]): Increment IDs by 1 (starting at 1
                rather than 0) if True (default False). Applies only when
                as_tuples is True or data is a DataFrame.
            drop_text (Optional[bool]): In a dataframe, drop the original text
                column if True (default True)
            **kwargs: Additional keyword arguments for the paragraph segmenter.

        Returns:
            Union[str, list, pd.Series, pd.DataFrame]: Data split into
                paragraphs.

        Raises:
            ValueError: If mode is not supported.
            TypeError: If data type is not supported.
        """
        return self._process_data(data=data, mode="paragraphs", column=column,
                                  as_tuples=as_tuples, include_span=include_span,
                                  mathematical_ids=mathematical_ids,
                                  drop_text=drop_text, **kwargs)

    def chunks(self,
               data: Union[str, list, pd.Series, pd.DataFrame],
               column: Optional[str] = TEXT_COL,
               as_tuples: Optional[bool] = False,
               include_span: Optional[bool] = False,
               mathematical_ids: Optional[bool] = False,
               drop_text: Optional[bool] = True,
               **kwargs
               ) -> Union[List[str], List[List[str]], pd.Series, pd.DataFrame]:
        """
        Split text into chunks.

        Args:
            data (Union[str, list, pd.Series, pd.DataFrame]): Text data to
                split.
            column (Optional[str]): Column name if data is a DataFrame
                (default is specified by TEXT_COL, currently "text").
            as_tuples (Optional[bool]): Whether to return tuples with id and
                text (only for data other than DataFrame, default False).
            include_span (Optional[bool]): Include span information if True
                (default False).
            mathematical_ids (Optional[bool]): Increment IDs by 1 (starting at 1
                rather than 0) if True (default False). Applies only when
                as_tuples is True or data is a DataFrame.
            drop_text (Optional[bool]): In a dataframe, drop the original text
                column if True (default True)
            **kwargs: Additional keyword arguments for the chunk segmenter.

        Returns:
            Union[str, list, pd.Series, pd.DataFrame]: Data split into chunks.

        Raises:
            ValueError: If mode is not supported.
            TypeError: If data type is not supported.
        """
        return self._process_data(data=data, mode="chunks", column=column,
                                  as_tuples=as_tuples, include_span=include_span,
                                  mathematical_ids=mathematical_ids,
                                  drop_text=drop_text, **kwargs)

    def _process_data(self,
                      data: Union[str, list, pd.Series, pd.DataFrame],
                      mode: str,
                      column: Optional[str] = TEXT_COL,
                      as_tuples: Optional[bool] = False,
                      include_span: Optional[bool] = False,
                      mathematical_ids: Optional[bool] = False,
                      drop_text: Optional[bool] = True,
                      **kwargs
                      ) -> Union[List[str], List[List[str]], pd.Series,
                           pd.DataFrame]:
        """
        Process data with the appropriate segmenter module based on mode.

        Args:
            data (Union[str, list, pd.Series, pd.DataFrame]): Text data to
                split.
            mode (str): Type of segment unit to split the text into.
            column (Optional[str]): Column name if data is a DataFrame
                (default is specified by TEXT_COL, currently "text").
            as_tuples (Optional[bool]): Whether to return tuples with id and
                text  (only for data other than DataFrame, default False).
            include_span (Optional[bool]): Include span information if True
                (default False).
            mathematical_ids (Optional[bool]): Increment IDs by 1 (starting at 1
                rather than 0) if True (default False). Applies only when
                as_tuples is True or data is a DataFrame.
            drop_text (Optional[bool]): In a dataframe, drop the original text
                column if True (default True)
            **kwargs: Additional keyword arguments for the segmenter.


        Returns:
            Union[str, list, pd.Series, pd.DataFrame]: Data split into segments
                based on the mode.

        Raises:
            ValueError: If mode is not supported.
            TypeError: If data type is not supported.
        """
        processor = self.processors.get(mode)
        if processor is None:
            raise ValueError(f"Unsupported mode '{mode}'. "
                             f"Use one of {list(self.processors.keys())}.")

        common_args = {
            "as_tuples": as_tuples,
            "include_span": include_span,
            **kwargs
        }

        # helper functions to maintain readability
        def process_text(text):
            return processor.split(text=text, **common_args)

        def process_list(texts):
            return processor.split_list(texts=texts, **common_args)

        def process_series(series):
            return pd.Series(processor.split_list(texts=series.tolist(),
                                                  **common_args))

        def process_df(df):
            return processor.split_df(input_df=df, column=column,
                                      include_span=include_span,
                                      drop_text=drop_text,
                                      mathematical_ids=mathematical_ids,
                                      **kwargs)

        handlers = {
            str: process_text,
            list: process_list,
            pd.Series: process_series,
            pd.DataFrame: process_df
        }

        handler = handlers.get(type(data))
        if handler:
            return handler(data)

        raise TypeError(f"Unsupported data type: {type(data)}. "
                        f"Expected one of {list(handlers.keys())}.")
