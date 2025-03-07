from typing import Dict, Union, Tuple, Callable

import pandas as pd
from tqdm.auto import tqdm

from ..constants import (TEXT_COL, SENT_COL, SENTS_COL, SENT_N_COL, SENT_ID_COL,
                         SENT_SPAN_COL)
from ..paragraphs.para_handling import ParagraphSegmenter
from .sentencizer import Sentencizer
from ..paragraphs.paragrapher import Paragrapher
from ..utils import column_list, increment_ids, add_id, find_substring_indices

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()


class SentenceSegmenter:
    def __init__(self, sent_specs, para_specs):
        if sent_specs is None:
            sent_specs = {}
        self.splitter = initiate_sentencizer(sent_specs.get("sentencizer",
                                                            ("pysbd", "de"))
                                             )

        self.paragrapher = ParagraphSegmenter(para_specs)

    def split(self,
              text: str,
              as_tuples: bool = False,
              include_span: bool = False,
              **kwargs
              ) -> list:
        """Split a string containing natural language data into sentences. Returns
        a list of sentences as strings. Optionally, return a list of tuples with
        sentence index and sentence as strings. Uses spacy for sentence splitting
        and requires a spacy language model as input."""
        drop_placeholders = kwargs.pop("drop_placeholders", [])

        # split into paragraphs first
        paragraphs = self.paragrapher.split(text,
                                            drop_placeholders=drop_placeholders)
        # process paragraphs individually into sentences
        sentences = self.splitter.split(paragraphs,
                                        **kwargs
                                        )
        # flatten sentence lists for paragraphs into one list for whole text
        sentences = [sentence for paragraph in sentences
                     for sentence in paragraph]

        if include_span:
            indices = find_substring_indices(text, sentences)
            sentences = list(zip(indices, sentences))

        if as_tuples:
            sentences = add_id(sentences)

        return sentences

    def split_list(self,
                   texts: list,
                   as_tuples: bool = False,
                   include_span: bool = False,
                   **kwargs
                   ) -> list:
        """
        Split a list of strings containing natural language data into sentences.
        Returns a list of sentences per text as lists of strings. Optionally,
        returns lists of tuples also including sentences index and/or start and
        end indices of sentences in the original text.

        Args:
            texts: list: List of strings to split into sentences.
            as_tuples: bool: Return sentences as tuples with index and sentence.
            include_span: bool: Return sentences as tuples with index, start and
                end indices of sentence in original text.

        Returns:
            list: List of lists of sentences as strings or tuples.
        """
        drop_placeholders = kwargs.pop("drop_placeholders", [])
        show_progress = kwargs.pop("show_progress", False)

        # split into paragraphs first
        paragraphs = self.paragrapher.split_list(
            texts, drop_placeholders=drop_placeholders)

        # process paragraphs individually into sentences
        if show_progress:
            iterator = tqdm(paragraphs, desc="Splitting sentences")
        else:
            iterator = paragraphs

        sentences = [self.splitter.split(data=para_list,
                                         as_tuples=as_tuples,
                                         include_span=include_span,
                                         **kwargs
                                         )
                     for para_list in iterator
                     ]

        # flatten sentence lists for paragraphs into one list for whole text
        sentences = [[sentence for paragraph in para_list for sentence in paragraph]
                     for para_list in sentences]

        if include_span:
            if show_progress:
                iterator = tqdm(zip(texts, sentences),
                                desc="Adding span indices",
                                total=len(texts))
            else:
                iterator = zip(texts, sentences)

            sentences = [list(zip(find_substring_indices(text, sents), sents))
                            for text, sents in iterator]

        if as_tuples:
            sentences = [add_id(sents) for sents in sentences]

        return sentences

    def split_df(self,
                 input_df: pd.DataFrame,
                 column: str = TEXT_COL,
                 drop_text: bool = True,
                 mathematical_ids: bool = False,
                 include_span: bool = False,
                 **kwargs
                 ) -> pd.DataFrame:
        """
        In a pandas dataframe containing a column with data data, insert three
        new columns with individual embeddings derived from data data, number of
        embeddings per data, and embeddings IDs. DataFrame is exploded to one row
        per sentence, keeping sentence together with original data data.
        Optionally, drop the original data column.

        Sentences are split with the help of spaCy NLP, therefore an NLP object
        is required as input. Before processing, data are split into paraggraphs
        by regex before processing. The default pattern is simple, splitting at
        newlines, optionally a more complex pattern tailored to German data is
        employed.
        """

        df = input_df.copy()

        df[SENTS_COL] = pd.Series(self.split_list(df[column].tolist(),
                                                  as_tuples=True,
                                                  include_span=include_span,
                                                  **kwargs
                                                  )
                                  )

        df = df.explode(SENTS_COL).reset_index(drop=True)

        # unpack sentence data into separate columns
        sents_df = pd.DataFrame(df[SENTS_COL].tolist())
        if include_span:
            df[[SENT_ID_COL, SENT_SPAN_COL, SENT_COL]] = sents_df
        else:
            df[[SENT_ID_COL, SENT_COL]] = sents_df

        # count sentences per text
        df[SENT_N_COL] = df.groupby(column)[column].transform("size")

        if mathematical_ids:
            df[SENT_ID_COL] = df[SENT_ID_COL].map(lambda x: x + 1)

        # keep only desired columns for output dataframe
        columns = [c for c in column_list(SENT_COL, column) if c in df.columns]

        if drop_text:
            columns.remove(column)

        return df[columns]


def split_sentences(
        input_df: pd.DataFrame,
        column: str = TEXT_COL,
        sentence_specs: Dict[str, Union[Tuple[str, str], Callable, bool]] = None,
        include_span: bool = False,
        drop_text: bool = True,
        mathematical_ids: bool = False,
        ) -> pd.DataFrame:
    """
    In a pandas dataframe containing a column with data data, insert three
    new columns with individual embeddings derived from data data, number of
    embeddings per data, and embeddings IDs. DataFrame is exploded to one row
    per sentence, keeping sentence together with original data data.
    Optionally, drop the original data column.

    Sentences are split with the help of spaCy NLP, therefore an NLP object
    is required as input. Before processing, data are split into paraggraphs
    by regex before processing. The default pattern is simple, splitting at
    newlines, optionally a more complex pattern tailored to German data is
    employed.
    """

    df = input_df.copy()
    df[[SENTS_COL, SENT_N_COL]] = find_sentences(dataframe = df,
                                                 sentence_specs=sentence_specs,
                                                 column=column,
                                                 mathematical_ids=mathematical_ids,
                                                 include_span=include_span,
                                                 )
    df = df.explode(SENTS_COL)
    df = df.reset_index(drop=True)

    # unpack sentence data into separate columns
    sents_df = pd.DataFrame(df[SENTS_COL].tolist())
    if include_span:
        df[[SENT_ID_COL, SENT_SPAN_COL, SENT_COL]] = sents_df
    else:
        df[[SENT_ID_COL, SENT_COL]] = sents_df

    # keep only desired columns for output dataframe
    columns = [c for c in column_list(SENT_COL, column) if c in df.columns]

    if drop_text:
        columns.remove(column)

    return df[columns]


def find_sentences(
        dataframe: pd.DataFrame,
        column: str = TEXT_COL,
        sentence_specs: Dict[str, Union[Tuple[str, str], Callable, bool]] = None,
        mathematical_ids: bool = False,
        include_span: bool = False,
        ) -> pd.DataFrame:
    """Create new columns with data split into embeddings and no of embeddings
    Split data into embeddings with the help of spacy."""

    sentences = dataframe[column].progress_map(
        lambda text: make_sentences_from_text(text=text,
                                              sentence_specs=sentence_specs,
                                              as_tuples=True,
                                              include_span=include_span,
                                              )
    )

    if mathematical_ids:
        sentences = sentences.map(lambda cell: increment_ids(cell, 1))

    dataframe[SENTS_COL] = sentences
    dataframe[SENT_N_COL] = dataframe[SENTS_COL].map(len)

    return dataframe[[SENTS_COL, SENT_N_COL]]


def make_sentences_from_text(
        text: str,
        sentence_specs: Dict[str, Union[Tuple[str, str], Callable, bool]] = None,
        as_tuples: bool = False,
        include_span: bool = False,
        ) -> list:
    """Split a string containing natural language data into sentences. Returns
    a list of sentences as strings. Optionally, return a list of tuples with
    sentence index and sentence as strings. Uses spacy for sentence splitting
    and requires a spacy language model as input."""
    if sentence_specs is None:
        sentence_specs = {}

    sentencizer_specs = sentence_specs.get("sentencizer", ("pysbd", "de"))
    show_progress = sentence_specs.get("show_progress", False)
    paragrapher_specs = sentence_specs.get("paragrapher", "clean")
    drop_placeholders = sentence_specs.get("drop_placeholders", [])


    # initiate sentencizer
    if isinstance(sentencizer_specs, tuple):
        sentencizer = Sentencizer(*sentencizer_specs)
    elif callable(sentencizer_specs):
        sentencizer = Sentencizer(sentencizer_specs)
    else:
        raise ValueError("Sentencizer must be a tuple of strings specifying a "
                         "built-in type with language/model or a custom "
                         "callable.")

    # initiate paragrapher
    if isinstance(paragrapher_specs, str):
        paragrapher = Paragrapher(paragrapher_specs)
    elif isinstance(paragrapher_specs, tuple):
        paragrapher = Paragrapher(*paragrapher_specs)
    elif callable(paragrapher_specs):
        paragrapher = Paragrapher(paragrapher_specs)
    else:
        raise ValueError("Paragrapher must be either a built-in type (specified "
                         "by string or tuple of string and specs) or a custom "
                         "callable.")

    # split into paragraphs first
    paragraphs = paragrapher.split(text)
    # process paragraphs individually into sentences
    sentences = sentencizer.split(paragraphs, show_progress=show_progress)
    # flatten sentence lists for paragraphs into one list for whole text
    sentences = [sentence for paragraph in sentences for sentence in paragraph]

    if include_span:
        indices = find_substring_indices(text, sentences)
        sentences = list(zip(indices, sentences))

    if as_tuples:
        sentences = add_id(sentences)

    return sentences


def initiate_sentencizer(specs):
    if isinstance(specs, tuple):
        return Sentencizer(*specs)
    elif callable(specs):
        return Sentencizer(specs)
    else:
        raise ValueError("Sentencizer must be a tuple of strings specifying a "
                         "built-in type with language/model or a custom "
                         "callable.")
