from typing import Optional, Union

import pandas as pd
import spacy

from tqdm.auto import tqdm

from .constants import TEXT_COL
from .chunks.chunk_handling import ChunkModule, DummyChunkModule
from .paragraphs.para_handling import ParagraphModule
from .sentences.sent_handling import SentenceModule

# register pandas
tqdm.pandas()


class TextSplitter:
    def __init__(
            self,
            as_tuples: Optional[bool] = False,
            include_span: Optional[bool] = False,
            sentence_specs: Optional[dict] = None,
            paragraph_specs: Optional[dict] = None,
            chunking_specs: Optional[dict] = None,
            mathematical_ids: Optional[bool] = False,
            drop_text: Optional[bool] = True,
            ):
        self.sentence_specs = sentence_specs
        self.as_tuples = as_tuples
        self.include_span = include_span

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

        self.mathematical_ids = mathematical_ids
        self.drop_text = drop_text

    def __call__(
            self,
            data: Union[str, list, pd.Series, pd.DataFrame],
            mode: str,
            column: Optional[str] = TEXT_COL,
            as_tuples: Optional[bool] = None,
            include_span: Optional[bool] = False,
            mathematical_ids: Optional[bool] = False,
            drop_text: Optional[bool] = True,
            **chunker_kwargs
            ) -> Union[str, list, pd.Series, pd.DataFrame]:

        if isinstance(data, str):
            return self._split(data=data, mode=mode,
                               as_tuples=as_tuples,
                               include_span=include_span,
                               **chunker_kwargs
                               )
        elif isinstance(data, list):
            return self._split_multiple(data=data, mode=mode,
                                        as_tuples=as_tuples,
                                        include_span=include_span,
                                        **chunker_kwargs
                                        )
        elif isinstance(data, pd.Series):
            return self._split_series(data=data, mode=mode,
                                      as_tuples=as_tuples,
                                      include_span=include_span,
                                      **chunker_kwargs
                                      )
        elif isinstance(data, pd.DataFrame):
            return self._split_df(data=data, mode=mode,
                                  as_tuples=as_tuples,
                                  include_span=include_span,
                                  column=column,
                                  mathematical_ids=mathematical_ids,
                                  drop_text=drop_text,
                                  **chunker_kwargs
                                  )
        else:
            raise ValueError(f"Data type not supported. Provided data is of "
                             f"type {type(data)}, but must be str, list, "
                             f"pd.Series or pd.DataFrame.")

    def sentences(self, *args, **kwargs
                  ) -> Union[str, list, pd.Series, pd.DataFrame]:
        return self(mode="sentences", *args, **kwargs)

    def paragraphs(self, *args, **kwargs
                   ) -> Union[str, list, pd.Series, pd.DataFrame]:
          return self(mode="paragraphs", *args, **kwargs)

    def chunks(self, *args, **kwargs
               ) -> Union[str, list, pd.Series, pd.DataFrame]:
        return self(mode="chunks", *args, **kwargs)

    def _split(self,
               data: str,
               mode: str,
               as_tuples: bool = False,
               include_span: bool = False,
               **chunker_kwargs
               ) -> list:
        if mode == "sentences":
            return self.sentencizer.split(text=data,
                                          as_tuples=as_tuples,
                                          include_span=include_span
                                          )
        elif mode == "paragraphs":
            return self.paragrapher.split(text=data,
                                          as_tuples=as_tuples,
                                          include_span=include_span
                                          )
        elif mode == "chunks":
            return self.chunker.chunk(text=data,
                                      as_tuples=as_tuples,
                                      include_span=include_span,
                                      **chunker_kwargs
                                      )

    def _split_multiple(self,
                        data: list,
                        mode: str,
                        as_tuples: bool = False,
                        include_span: bool = False,
                        **chunker_kwargs
                        ) -> list:
        if mode == "sentences":
            return [self.sentencizer.split(text=text,
                                          as_tuples=as_tuples,
                                          include_span=include_span
                                           )
                    for text in tqdm(data, desc="Splitting sentences")]
        elif mode == "paragraphs":
            return [self.paragrapher.split(text=text,
                                           as_tuples=as_tuples,
                                           include_span=include_span
                                           )
                    for text in tqdm(data, desc="Splitting paragraphs")]
        elif mode == "chunks":
            return [self.chunker.chunk(text=text,
                                        as_tuples=as_tuples,
                                        include_span=include_span,
                                        **chunker_kwargs
                                        )
                    for text in tqdm(data, desc="Chunking texts")]

    def _split_series(self,
                      data: pd.Series,
                      mode: str,
                      as_tuples: bool = False,
                      include_span: bool = False,
                      **chunker_kwargs
                      ) -> pd.Series:
        if mode in ["sentences"]:
            return data.progress_map(
                lambda text: self.sentencizer.split(text=text,
                                                    as_tuples=as_tuples,
                                                    include_span=include_span
                                                    ),
                desc="Splitting sentences"
            )

        if mode == "paragraphs":
            return data.progress_map(
                lambda text: self.paragrapher.split(text=text,
                                                    as_tuples=as_tuples,
                                                    include_span=include_span),
                desc="Splitting paragraphs"
            )

        elif mode == "chunks":
            return data.progress_map(
                lambda text: self.chunker.chunk(text=text,
                                                as_tuples=as_tuples,
                                                include_span=include_span
                                                             **chunker_kwargs
                                                ),
                desc="Chunking texts"
            )

    def _split_df(self,
                  data: pd.DataFrame,
                  mode: str,
                  column: str = TEXT_COL,
                  include_span: bool = False,
                  drop_text: bool = True,
                  mathematical_ids: bool = False,
                  **chunker_kwargs
                  ) -> pd.DataFrame:

        if mode == "sentences":
            df= self.sentencizer.split_df(input_df=data,
                                          column=column,
                                          drop_text=drop_text,
                                          include_span=include_span,
                                          mathematical_ids=mathematical_ids
                                          )
        elif mode == "paragraphs":
            df = self.paragrapher.split_df(input_df=data,
                                           column=column,
                                           drop_text=drop_text,
                                           include_span=include_span,
                                           mathematical_ids=mathematical_ids,
                                           )
        elif mode == "chunks":
            df = self.chunker.chunk_df(input_df=data,
                                        column=column,
                                        drop_text=drop_text,
                                        include_span=include_span,
                                        mathematical_ids=mathematical_ids,
                                        **chunker_kwargs
                                       )
        return df

if __name__ == "__main__":
    filename = "/Users/borisbachmann/sciebo/Forschung_cloud/5_Projekte/3_VW/3_WP 3/Quellen sortiert/2_Korpus/2020-11-30_707.txt"
    filename_2 = "/Users/borisbachmann/sciebo/Forschung_cloud/5_Projekte/3_VW/3_WP 3/Quellen sortiert/2_Korpus/2021-MM-DD_810.txt"

    with open(filename, "r") as file:
        text_1 = file.read()
    with open(filename_2, "r") as file:
        text_2 = file.read()

    nlp = spacy.load("de_core_news_sm")

    splitter_specs = {"as_tuples": True,
                      "paragraph_function": None,
                      "chunking_strategy": "graph",
                      "chunking_specs": {
                          "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
                      }

    print("Initializing spliiter")
    splitter = TextSplitter(nlp, **splitter_specs)

    print("Splitting text")
    splitted_text = splitter(text_1, mode="chunks", as_tuples=True)
    for element in splitted_text:
        print(element)

    print("Splitting list of texts")
    texts = [text_1, text_2]
    for list_ in splitter(texts, mode="chunks", as_tuples=True):
        print(list_)
