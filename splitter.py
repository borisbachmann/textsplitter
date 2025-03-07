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

from typing import Optional, Union

import pandas as pd

from tqdm.auto import tqdm

from .constants import TEXT_COL
from .chunks.chunk_handling import ChunkModule, DummyChunkModule
from .paragraphs.para_handling import ParagraphModule
from .sentences.sent_handling import SentenceModule

# register pandas
tqdm.pandas()

class TextSplitter:
    """
    Class to split text into sentences, paragraphs, and chunks. TextSplitter
    processes text data in different formats and returns the split text in a
    corresponding format: Lists for strings and lists of strings, pandas Series
    for pandas Series, and pandas DataFrames for pandas DataFrames.

    A high-level wrapper class around different text-splitting techniques
    (sentences, paragraphs, and chunks). This class can be initialized with
    specifications for each segment type and provides a unified interface to
    the underlying module classes.
    """
    def __init__(
            self,
            sentence_specs: Optional[dict] = None,
            paragraph_specs: Optional[dict] = None,
            chunking_specs: Optional[dict] = None,
            ):
        # iniitiate paragraphing attributes
        self.paragrapher = ParagraphModule(paragraph_specs)

        # initiate sentencizing attributes
        self.sentencizer = SentenceModule(sentence_specs, paragraph_specs)

        # initiate chunking attributes
        if chunking_specs is None:
            print("TextSplitter intitialized without chunking capabilities.")
            self.chunker = DummyChunkModule()
        else:
            if not "model" in chunking_specs:
                print ("No model specified for chunking. Splitter intitialized "
                       "without chunking capabilities.")
                self.chunker = DummyChunkModule()
            else:
                self.chunker = ChunkModule(chunking_specs, paragraph_specs,
                                           sentence_specs)

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
            ) -> Union[str, list, pd.Series, pd.DataFrame]:
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
                  ) -> Union[str, list, pd.Series, pd.DataFrame]:
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
                   ) -> Union[str, list, pd.Series, pd.DataFrame]:
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
               ) -> Union[str, list, pd.Series, pd.DataFrame]:
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
                      ) -> Union[list, pd.Series, pd.DataFrame]:
        """Process data with the appropriate segmenter module based on mode."""
        processors = {
            "sentences": self.sentencizer,
            "paragraphs": self.paragrapher,
            "chunks": self.chunker
        }

        processor = processors.get(mode)
        if processor is None:
            raise ValueError(f"Unsupported mode {mode}. "
                             f"Use on of {list[processors.keys()]}.")

        if isinstance(data, str):
            return processor.split(text=data,
                                   as_tuples=as_tuples,
                                   include_span=include_span,
                                   **kwargs
                                   )
        elif isinstance(data, list):
            return processor.split_list(texts=data,
                                        as_tuples=as_tuples,
                                        include_span=include_span,
                                        **kwargs
                                        )
        elif isinstance(data, pd.Series):
            return pd.Series(processor.split_list(texts=data.tolist(),
                                                  as_tuples=as_tuples,
                                                  include_span=include_span,
                                                  **kwargs
                                                  )
                             )
        elif isinstance(data, pd.DataFrame):
            return processor.split_df(input_df=data,
                                      column=column,
                                      drop_text=drop_text,
                                      include_span=include_span,
                                      mathematical_ids=mathematical_ids,
                                      **kwargs
                                      )
        else:
            raise ValueError(f"Data type not supported. Provided data is of "
                             f"type {type(data)}, but must be str, List[str], "
                             f"pd.Series or pd.DataFrame.")
