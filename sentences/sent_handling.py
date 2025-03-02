from typing import Dict, Union, Tuple, Callable

import pandas as pd
from tqdm.auto import tqdm

from constants import TEXT_COL, SENT_COL, SENTS_COL, SENT_N_COL, SENT_ID_COL
from ..paragraphs.para_handling import make_paragraphs_from_text
from .sentencizer import Sentencizer
from utils import column_list, increment_ids, add_id

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()

def split_sentences(
        input_df: pd.DataFrame,
        column: str = TEXT_COL,
        sentence_specs: Dict[str, Union[Tuple[str, str], Callable, bool]] = None,
        drop_text: bool = True,
        mathematical_ids: bool = False,
        drop_empty: bool = True,
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
                                                 drop_empty=drop_empty
                                                 )
    df = df.explode(SENTS_COL)
    df[[SENT_ID_COL, SENT_COL]] = df[SENTS_COL].tolist()
    columns = [c for c in column_list(SENT_COL, column) if c in df.columns]

    if drop_text:
        columns.remove(column)

    return df[columns]


def find_sentences(
        dataframe: pd.DataFrame,
        column: str = TEXT_COL,
        sentence_specs: Dict[str, Union[Tuple[str, str], Callable, bool]] = None,
        mathematical_ids: bool = False,
        drop_empty: bool = True,
        ) -> pd.DataFrame:
    """Create new columns with data split into embeddings and no of embeddings
    Split data into embeddings with the help of spacy."""

    sentences = dataframe[column].progress_map(
        lambda text: make_sentences_from_text(text=text,
                                              sentence_specs=sentence_specs,
                                              drop_empty=drop_empty,
                                              as_tuples=True,
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
        drop_empty: bool = True,
        as_tuples: bool = False,
        ) -> list:
    """Split a string containing natural language data into sentences. Returns
    a list of sentences as strings. Optionally, return a list of tuples with
    sentence index and sentence as strings. Uses spacy for sentence splitting
    and requires a spacy language model as input."""
    if sentence_specs is None:
        sentence_specs = {}

    sentencizer_specs = sentence_specs.get("sentencizer", ("pysbd", "de"))
    show_progress = sentence_specs.get("show_progress", False)
    paragraph_function = sentence_specs.get("paragraph_function", None)

    if isinstance(sentencizer_specs, tuple):
        sentencizer = Sentencizer(*sentencizer_specs)
    elif callable(sentencizer_specs):
        sentencizer = Sentencizer(sentencizer_specs)
    else:
        raise ValueError("Sentencizer must be a tuple of strings specifying a "
                         "built-in type with language/model or a custom "
                         "callable.")

    # split into paragraphs first
    paragraphs = make_paragraphs_from_text(text = text,
                                           drop_empty=drop_empty,
                                           as_tuples=False,
                                           function=paragraph_function)
    # process paragraphs individually into sentences
    sentences = sentencizer.split(paragraphs, show_progress=show_progress)
    # flatten sentence lists for paragraphs into one list for whole text
    sentences = [sentence for paragraph in sentences for sentence in paragraph]

    if as_tuples:
        sentences = add_id(sentences)

    return sentences
