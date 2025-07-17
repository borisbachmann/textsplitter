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

from typing import Optional, Union, List, Tuple

import pandas as pd

from tqdm.auto import tqdm

from .chunks.backends import EmbeddingChunkerProtocol, SimpleChunkerProtocol
from .dataframes import columns
from .chunks import ChunkHandler
from .paragraphs import ParagraphHandler, ParaSegmenterProtocol
from .sentences import SentenceHandler, SentSegmenterProtocol
from .tokens import TokenHandler, TokenFormats, TokenSegmenterProtocol

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
            sentence_specs: Optional[
                Union[
                    Tuple[
                        Union[SentSegmenterProtocol, None],
                        Union[ParaSegmenterProtocol, None]
                    ],
                    SentSegmenterProtocol
                ]
            ] = None,
            paragraph_specs:  Optional[ParaSegmenterProtocol] = None,
            chunking_specs: Optional[
                Union[
                    Tuple[
                        Union[EmbeddingChunkerProtocol, SimpleChunkerProtocol],
                        Union[SentSegmenterProtocol, None],
                        Union[ParaSegmenterProtocol, None]
                    ],
                    Union[EmbeddingChunkerProtocol, SimpleChunkerProtocol],
                ]
            ] = None,
            token_specs: Optional[TokenSegmenterProtocol] = None,
            ):
        # initialize paragraphing attributes
        self.paragrapher = ParagraphHandler(paragraph_specs)

        # initialize sentencizing attributes
        sent_backend, para_backend = (
            self._resolve_sentence_specs(sentence_specs)
        )
        self.sentencizer = SentenceHandler(sent_backend, para_backend)

        # initialize chunking attributes
        chunk_backend, sent_backend, para_backend = (
            self._resolve_chunk_specs(chunking_specs)
        )
        self.chunker = ChunkHandler(chunk_backend, sent_backend, para_backend)

        # initialize tokenizer attritubtes
        self.tokenizer = TokenHandler(token_specs)

        self.processors = {
            "sentences": self.sentencizer,
            "paragraphs": self.paragrapher,
            "chunks": self.chunker,
            "tokens": self.tokenizer
        }

    # present for backwards compatibility. Use explicit methods instead.
    def __call__(
            self,
            data: Union[str, list, pd.Series, pd.DataFrame],
            mode: str,
            column: Optional[str] = columns.TEXT_COL,
            as_tuples: Optional[bool] = False,
            include_span: Optional[bool] = False,
            mathematical_ids: Optional[bool] = False,
            drop_text: Optional[bool] = True,
            keep_orig: Optional[List[str]] = None,
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
            keep_orig (Optional[List[str]]): List of original dataframe columns
                to keep (default None).
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
                                  drop_text=drop_text, keep_orig=keep_orig,
                                  **kwargs)

    def _resolve_sentence_specs(self, sentence_specs):
        main_para_backend = self.paragrapher.splitter._backend
        if sentence_specs is not None:
            if isinstance(sentence_specs, tuple):
                sent_backend = sentence_specs[0]
                if sentence_specs[1] is not None:
                    para_backend = sentence_specs[1]
                else:
                    para_backend = main_para_backend
            elif callable(sentence_specs):
                sent_backend = sentence_specs
                para_backend = main_para_backend
            else:
                raise ValueError("Sentece specs must either be custom callable "
                                 "sentence backend or a tuple of sentence and "
                                 "paragraph backend.")
        else:
            sent_backend = None
            para_backend = main_para_backend

        return sent_backend, para_backend

    def _resolve_chunk_specs(
            self,
            chunking_specs
    ):
        main_sent_backend = self.sentencizer.splitter._backend
        main_sent_para_backend = self.sentencizer.para_splitter.splitter._backend
        if chunking_specs is not None:
            if isinstance(chunking_specs, tuple):
                chunk_backend, sent_backend, para_backend = chunking_specs
                if sent_backend is None:
                    sent_backend = main_sent_backend
                if para_backend is None:
                    para_backend = main_sent_para_backend
            elif callable(chunking_specs):
                chunk_backend = chunking_specs
                sent_backend = main_sent_backend
                para_backend = main_sent_para_backend
            else:
                raise ValueError("Sentece specs must either be custom callable "
                                 "sentence backend or a tuple of chunk, sentence and "
                                 "paragraph backend.")
        else:
            chunk_backend, sent_backend, para_backend = None, None, None

        return chunk_backend, sent_backend, para_backend

    def sentences(self,
                  data: Union[str, list, pd.Series, pd.DataFrame],
                  column: Optional[str] = columns.TEXT_COL,
                  as_tuples: Optional[bool] = False,
                  include_span: Optional[bool] = False,
                  mathematical_ids: Optional[bool] = False,
                  drop_text: Optional[bool] = True,
                  keep_orig: Optional[List[str]] = None,
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
            keep_orig (Optional[List[str]]): List of original dataframe columns
                to keep (default None).
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
                                  drop_text=drop_text, keep_orig=keep_orig,
                                  **kwargs)

    def paragraphs(self,
                   data: Union[str, list, pd.Series, pd.DataFrame],
                   column: Optional[str] = columns.TEXT_COL,
                   as_tuples: Optional[bool] = False,
                   include_span: Optional[bool] = False,
                   mathematical_ids: Optional[bool] = False,
                   drop_text: Optional[bool] = True,
                   keep_orig: Optional[List[str]] = None,
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
            keep_orig (Optional[List[str]]): List of original dataframe columns
                to keep (default None).
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
                                  drop_text=drop_text, keep_orig=keep_orig,
                                  **kwargs)

    def chunks(self,
               data: Union[str, list, pd.Series, pd.DataFrame],
               column: Optional[str] = columns.TEXT_COL,
               as_tuples: Optional[bool] = False,
               include_span: Optional[bool] = False,
               mathematical_ids: Optional[bool] = False,
               drop_text: Optional[bool] = True,
               keep_orig: Optional[List[str]] = None,
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
            keep_orig (Optional[List[str]]): List of original dataframe columns
                to keep (default None).
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
                                  drop_text=drop_text, keep_orig=keep_orig,
                                  **kwargs)

    def tokens(self,
               data: Union[str, list, pd.Series, pd.DataFrame],
               column: Optional[str] = columns.TEXT_COL,
               as_tuples: Optional[bool] = False,
               include_span: Optional[bool] = False,
               include_metadata: Optional[bool] = False,
               mathematical_ids: Optional[bool] = False,
               drop_text: Optional[bool] = True,
               keep_orig: Optional[List[str]] = None,
               **kwargs
               ) -> Union[List[TokenFormats], List[List[TokenFormats]],
                    pd.Series, pd.DataFrame]:
        return self._process_data(data=data, mode="tokens", column=column,
                                  as_tuples=as_tuples, include_span=include_span,
                                  include_metadata=include_metadata,
                                  mathematical_ids=mathematical_ids,
                                  drop_text=drop_text, keep_orig=keep_orig,
                                  **kwargs)

    def set_sentencizer(self,
                        sentence_specs: Optional[
                            Union[
                                Tuple[Union[
                                    SentSegmenterProtocol, None
                                ],
                                Union[ParaSegmenterProtocol, None]
                                ],
                                SentSegmenterProtocol
                            ]
                        ] = None
                        ):
        """Set internal sentencizer with new specs.

        Args:
            sentence_specs (Optional[dict]): Specifications for the sentence
                segmenter. See SentenceModule for details.
            paragraph_specs (Optional[dict]): Specifications for the
                sentencizer's internal paragraph segmenter.
                See SentenceModule for details.
        """
        # initialize sentencizing attributes
        sent_backend, para_backend = self._resolve_sentence_specs(sentence_specs)
        self.sentencizer = SentenceHandler(sent_backend, para_backend)

        self.processors["sentences"] = self.sentencizer


    def set_paragrapher(
            self,
            paragraph_specs: Optional[ParaSegmenterProtocol] = None
    ):
        """
        Set internal paragrapher with new specs.

        Args:
            paragraph_specs (Optional[dict]): Specifications for the paragraph
                segmenter.
        """
        # initialize paragraphing attributes
        self.paragrapher = ParagraphHandler(paragraph_specs)

        self.processors["paragraphs"] = self.paragrapher


    # def set_chunker(self,
    #                chunking_specs: Optional[dict] = None,
    #                paragraph_specs: Optional[dict] = None,
    #                sentence_specs: Optional[dict] = None):
    #    """
    #    Set internal chunker with new specs.
    #
    #    Args:
    #        chunking_specs (Optional[dict]): Specifications for the chunk
    #            segmenter.
    #        paragraph_specs (Optional[dict]): Specifications for the chunk
    #            segmenter's internal paragraph segmenter.
    #        sentence_specs (Optional[dict]): Specifications for the chunk
    #            segmenter's internal sentence segmenter.
    #    """
    #    # initialize chunking attributes
    #    self.chunker = ChunkHandler(chunking_specs, paragraph_specs,
    #                                sentence_specs)
    #
    #    self.processors["chunks"] = self.chunker


    def set_tokenizer(
            self,
            token_specs: Optional[TokenSegmenterProtocol] = None
    ):
        """
        Set internal tokenizer with new specs.

        Args:
            token_specs (Optional[dict]): Specifications for the tokenizer.
        """
        # initialize tokenizer attritubtes
        self.tokenizer = TokenHandler(token_specs)

        self.processors["tokens"] = self.tokenizer


    def _process_data(self,
                      data: Union[str, list, pd.Series, pd.DataFrame],
                      mode: str,
                      column: Optional[str] = columns.TEXT_COL,
                      as_tuples: Optional[bool] = False,
                      include_span: Optional[bool] = False,
                      mathematical_ids: Optional[bool] = False,
                      drop_text: Optional[bool] = True,
                      keep_orig: Optional[bool] = List[str],
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
            keep_orig (Optional[List[str]]): List of original dataframe columns
                to keep (default None).
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
            return processor.split_df(input_df=df, text_column=column,
                                      include_span=include_span,
                                      drop_text=drop_text,
                                      keep_orig=keep_orig,
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
