from typing import Optional, Union

import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer

from tqdm.auto import tqdm

from .embeddings import EmbeddingModel
from .constants import TEXT_COL, DEFAULT_STRATEGY
from .chunks.chunk_handling import (chunk_text, split_chunks,
                                   chunk_multiple_texts, chunk_text_series)
from .paragraphs.para_handling import make_paragraphs_from_text, split_paragraphs
from .sentences.sent_handling import make_sentences_from_text, split_sentences

tqdm.pandas()

class TextSplitter:
    def __init__(
            self,
            drop_empty: Optional[bool] = True,
            as_tuples: Optional[bool] = False,
            include_span: Optional[bool] = False,
            sentence_specs: Optional[dict] = None,
            paragraph_specs: Optional[dict] = None,
            chunking_strategy: Optional[str] = DEFAULT_STRATEGY,
            chunking_specs: Optional[dict] = None,
            mathematical_ids: Optional[bool] = False,
            drop_text: Optional[bool] = True,
            ):
        self.sentence_specs = sentence_specs
        self.drop_empty = drop_empty
        self.as_tuples = as_tuples
        self.include_span = include_span
        self.paragraph_specs = paragraph_specs
        self.chunking_strategy = chunking_strategy
        self.chunking_specs = chunking_specs
        if self.chunking_specs is not None:
            if 'model' in self.chunking_specs:
                self.chunking_specs['model'] = self._load_model(self.chunking_specs['model'])

        self.mathematical_ids = mathematical_ids
        self.drop_text = drop_text

    def __call__(
            self,
            data: Union[str, list, pd.Series, pd.DataFrame],
            mode: str,
            column: Optional[str] = None,
            drop_empty: Optional[bool] = None,
            as_tuples: Optional[bool] = None,
            include_span: Optional[bool] = False,
            paragraph_specs: Optional[dict] = None,
            sentence_specs: Optional[dict] = None,
            chunking_strategy: Optional[str] = None,
            chunking_specs: Optional[dict] = None,
            mathematical_ids: Optional[bool] = None,
            drop_text: Optional[bool] = None,
            ) -> Union[str, list, pd.Series, pd.DataFrame]:

        if isinstance(data, str):
            specs = self._compile_specs(drop_empty, as_tuples, include_span,
                                        paragraph_specs, sentence_specs,
                                        chunking_strategy, chunking_specs)
            return self._split(data=data, mode=mode, **specs)
        elif isinstance(data, list):
            specs = self._compile_specs(drop_empty, as_tuples, include_span,
                                        paragraph_specs, sentence_specs,
                                        chunking_strategy, chunking_specs)
            return self._split_multiple(data=data, mode=mode, **specs)
        elif isinstance(data, pd.Series):
            specs = self._compile_specs(drop_empty, as_tuples, include_span,
                                        paragraph_specs, sentence_specs,
                                        chunking_strategy, chunking_specs)
            return self._split_series(data=data, mode=mode, **specs)
        elif isinstance(data, pd.DataFrame):
            specs = self._compile_df_specs(drop_empty, as_tuples, include_span,
                                           paragraph_specs, sentence_specs,
                                           chunking_strategy, chunking_specs,
                                           column, mathematical_ids, drop_text)
            return self._split_df(data=data, mode=mode, **specs)
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
               drop_empty: bool,
               as_tuples: bool,
               include_span: bool,
               paragraph_specs: dict,
               sentence_specs: dict,
               chunking_strategy: str,
               chunking_specs: dict
               ) -> list:
        if mode == "sentences":
            return make_sentences_from_text(text=data,
                                            sentence_specs=sentence_specs,
                                            drop_empty=drop_empty,
                                            as_tuples=as_tuples,
                                            include_span=include_span
                                            )
        elif mode == "paragraphs":
            return make_paragraphs_from_text(text=data,
                                             drop_empty=drop_empty,
                                             as_tuples=as_tuples,
                                             include_span=include_span,
                                             paragraph_specs=paragraph_specs
                                             )
        elif mode == "chunks":
            return chunk_text(text=data,
                              as_tuples=as_tuples,
                              include_span=include_span,
                              strategy=chunking_strategy,
                              specs=chunking_specs,
                              sent_specs=sentence_specs
                              )

    def _split_multiple(self,
                        data: list,
                        mode: str,
                        drop_empty: bool,
                        as_tuples: bool,
                        include_span: bool,
                        paragraph_specs: dict,
                        sentence_specs: dict,
                        chunking_strategy: str,
                        chunking_specs: dict
                        ) -> list:
        if mode == "sentences":
            return [make_sentences_from_text(text=text,
                                             sentence_specs=sentence_specs,
                                             drop_empty=drop_empty,
                                             as_tuples=as_tuples,
                                             include_span=include_span
                                             )
                    for text in data]
        elif mode == "paragraphs":
            return [make_paragraphs_from_text(text=text,
                                              drop_empty=drop_empty,
                                              as_tuples=as_tuples,
                                              include_span=include_span,
                                              paragraph_specs=paragraph_specs
                                              )
                    for text in data]
        elif mode == "chunks":
            return chunk_multiple_texts(texts=data,
                                        as_tuples=as_tuples,
                                        include_span=include_span,
                                        strategy=chunking_strategy,
                                        specs=chunking_specs,
                                        sent_specs=sentence_specs
                                        )

    def _split_series(self,
                      data: pd.Series,
                      mode: str,
                      drop_empty: bool,
                      as_tuples: bool,
                      include_span: bool,
                      paragraph_specs: dict,
                      sentence_specs: dict,
                      chunking_strategy: str,
                      chunking_specs: dict
                      ) -> pd.Series:
        if mode in ["sentences", "paragraphs"]:
            return data.progress_map(
                lambda text: self._split(data=text,
                                         mode=mode,
                                         drop_empty=drop_empty,
                                         as_tuples=as_tuples,
                                         include_span=include_span,
                                         paragraph_specs=paragraph_specs,
                                         sentence_specs=sentence_specs,
                                         chunking_strategy=chunking_strategy,
                                         chunking_specs=chunking_specs
                                         )
            )
        elif mode == "chunks":
            return chunk_text_series(texts=data,
                                     strategy=chunking_strategy,
                                     specs=chunking_specs,
                                     as_tuples=as_tuples,
                                     include_span=include_span,
                                     drop_empty=drop_empty,
                                     mathematical_ids=False,
                                     sent_specs=sentence_specs
                                     )

    def _split_df(self,
                  data: pd.DataFrame,
                  column: str,
                  mode: str,
                  include_span: bool,
                  drop_empty: bool,
                  paragraph_specs: dict,
                  sentence_specs: dict,
                  chunking_strategy: str,
                  chunking_specs: dict,
                  drop_text: bool,
                  mathematical_ids: bool
                  ) -> pd.DataFrame:

        if mode == "sentences":
            df= split_sentences(input_df=data,
                                sentence_specs=sentence_specs,
                                column=column,
                                include_span=include_span,
                                drop_text=drop_text,
                                mathematical_ids=mathematical_ids,
                                drop_empty=drop_empty
                                )
        elif mode == "paragraphs":
            df = split_paragraphs(input_df=data,
                                  column=column,
                                  include_span=include_span,
                                  drop_text=drop_text,
                                  mathematical_ids=mathematical_ids,
                                  drop_empty=drop_empty,
                                  paragraph_specs=paragraph_specs
                                  )
        elif mode == "chunks":
            df = split_chunks(input_df=data,
                              column=column,
                              include_span=include_span,
                              drop_text=drop_text,
                              mathematical_ids=mathematical_ids,
                              drop_empty=drop_empty,
                              strategy=chunking_strategy,
                              specs=chunking_specs
                              )
        df = df.reset_index(drop=True)

        return df

    def _load_model(
            self,
            model: Union[str, EmbeddingModel, SentenceTransformer]
            ) -> Union[EmbeddingModel, SentenceTransformer, str]:
        if isinstance(model, str):
            return EmbeddingModel(model)
        elif (isinstance(model, EmbeddingModel) or
              isinstance(model, SentenceTransformer)):
            return model
        else:
            raise ValueError("Model must be a string or an instance of "
                             "EmbeddingModel or SentenceTransformer.")

    def _compile_specs(self,
                       drop_empty: bool,
                       as_tuples: bool,
                       include_span: bool,
                       paragraph_specs: dict,
                       sentence_specs: dict,
                       chunking_strategy: str,
                       chunking_specs: dict,
                       ) -> dict:

        return {
            "drop_empty": (drop_empty if drop_empty is not None
                           else self.drop_empty),
            "as_tuples": (as_tuples if as_tuples is not None
                          else self.as_tuples),
            "include_span": (include_span if include_span is not None
                             else self.include_span),
            "paragraph_specs": (paragraph_specs if paragraph_specs is not None
                                else self.paragraph_specs),
            "sentence_specs": (sentence_specs if sentence_specs is not None
                               else self.sentence_specs),
            "chunking_strategy": (chunking_strategy
                                  if chunking_strategy is not None
                                  else self.chunking_strategy),
            "chunking_specs": self._compile_chunking_specs(chunking_specs)
        }

    def _compile_df_specs(self,
                          drop_empty: bool,
                          as_tuples: bool,
                          include_span: bool,
                          paragraph_specs: dict,
                          sentence_specs: dict,
                          chunking_strategy: str,
                          chunking_specs: dict,
                          column: str,
                          mathematical_ids: bool,
                          drop_text: bool
                          ) -> dict:
        base_specs = {
            k: v for k, v in self._compile_specs(drop_empty, as_tuples,
                                                 include_span,
                                                 paragraph_specs,
                                                 sentence_specs,
                                                 chunking_strategy,
                                                 chunking_specs).items()
            if k != "as_tuples"
        }
        df_specs = {
            "column": (column if column is not None else TEXT_COL),
            "mathematical_ids": (mathematical_ids
                                 if mathematical_ids is not None
                                 else self.mathematical_ids),
            "drop_text": (drop_text if drop_text is not None
                          else self.drop_text)
        }

        return {**base_specs, **df_specs}

    def _compile_chunking_specs(self, specs):
        specs = specs or {}
        if 'model' in specs:
            specs['model'] = self._load_model(specs['model'])
        if self.chunking_specs is not None:
            for k, v in self.chunking_specs.items():
                if k not in specs:
                    specs[k] = v
        return specs

if __name__ == "__main__":
    filename = "/Users/borisbachmann/sciebo/Forschung_cloud/5_Projekte/3_VW/3_WP 3/Quellen sortiert/2_Korpus/2020-11-30_707.txt"
    filename_2 = "/Users/borisbachmann/sciebo/Forschung_cloud/5_Projekte/3_VW/3_WP 3/Quellen sortiert/2_Korpus/2021-MM-DD_810.txt"

    with open(filename, "r") as file:
        text_1 = file.read()
    with open(filename_2, "r") as file:
        text_2 = file.read()

    nlp = spacy.load("de_core_news_sm")

    splitter_specs = {"drop_empty": True,
                      "as_tuples": True,
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
