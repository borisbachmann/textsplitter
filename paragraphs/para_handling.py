from typing import Dict, Union, Tuple, Callable, Any

import pandas as pd
from tqdm.auto import tqdm

from .paragrapher import Paragrapher
from ..constants import (TEXT_COL, PARA_COL, PARAS_COL, PARA_N_COL, PARA_ID_COL,
                         BULLETS, PARA_SPAN_COL)
from .para_splitting import split_clean_paragraphs
from ..patterns import PARAGRAPH_PATTERN_SIMPLE, ENUM_PATTERN_NO_DATE
from ..utils import column_list, increment_ids, add_id, clean_placeholders, find_substring_indices

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()


def split_paragraphs(
        input_df: pd.DataFrame,
        column: str = TEXT_COL,
        function: callable = None,
        drop_text: bool = True,
        mathematical_ids: bool = False,
        drop_placeholders: list = None,
        drop_empty: bool = True,
        include_span: bool = False
        ) -> pd.DataFrame:
    """
    In a pandas dataframe containing a column with data data, insert three
    new columns with individual paragraphs derived from data data, number of
    paragraphs per data, and paragraph IDs. DataFrame is exploded to one row
    per paragraph, keeping paragraphs together with original data data.
    Optionally, drop the original data column.

    Paragraphs are split based on a function passed as an argument. If no
    function is provided, a standard function will be selected under the hood.
    """
    df = input_df.copy()
    df[[PARAS_COL, PARA_N_COL]] = find_paragraphs(dataframe=df,
                                                  column=column,
                                                  function=function,
                                                  drop_placeholders=drop_placeholders,
                                                  mathematical_ids=mathematical_ids,
                                                  drop_empty=drop_empty,
                                                  include_span=include_span
                                                  )
    df = df.explode(PARAS_COL)
    df = df.reset_index(drop=True)

    # unpack sentence data into separate columns
    paras_df = pd.DataFrame(df[PARAS_COL].tolist())
    if include_span:
        df[[PARA_ID_COL, PARA_SPAN_COL, PARA_COL]] = paras_df
    else:
        df[[PARA_ID_COL, PARA_COL]] = paras_df

    # keep only desired columns for output dataframe
    columns = [c for c in column_list(PARA_COL, column) if c in df.columns]

    if drop_text:
        columns.remove(TEXT_COL)

    return df[columns]

def find_paragraphs(
        dataframe: pd.DataFrame,
        column: str = TEXT_COL,
        function: callable = None,
        mathematical_ids: bool = False,
        drop_placeholders: list = None,
        drop_empty: bool = True,
        include_span: bool = False
        ) -> pd.DataFrame:
    """
    Create new columns with data split into paragraphs and no of paragraphs
    Split data into paragraphs based on simple heuristic ( ?, !, ., ), " and “
    at end of paragraph.

    Paragraphs are split based on a function passed as an argument. If no
    function is provided, a standard function will be selected under the hood.
    """
    paragraphs = dataframe[column].progress_map(
       lambda text: make_paragraphs_from_text(text=text,
                                              function=function,
                                              drop_placeholders=drop_placeholders,
                                              drop_empty=drop_empty,
                                              as_tuples=True,
                                              include_span=include_span
                                              )
    )

    if mathematical_ids:
        paragraphs = paragraphs.map(lambda cell: increment_ids(cell, 1))

    dataframe[PARAS_COL] = paragraphs
    dataframe[PARA_N_COL] = dataframe[PARAS_COL].map(len)

    return dataframe[[PARAS_COL, PARA_N_COL]]

# Main paragraph splitting function

def make_paragraphs_from_text(
        text: str,
        paragraph_specs: (Dict[str, Any]) = None,
        drop_empty: bool = True,
        as_tuples: bool = False,
        include_span: bool = False,
        *args,
        **kwargs
        ) -> list:
    """Split a string containing natural language data into paragraphs. Returns
    a list of paragraphs. Optionally, return a list of tuples with paragraph
    ids and data.

    Paragraphs are split based on a function passed as an argument. If no
    function is provided, a standard function will be selected under the hood.
    """
    if paragraph_specs is None:
        paragraph_specs = {}

    paragrapher_specs = paragraph_specs.get("paragrapher", ("clean", None))
    drop_placeholders = paragraph_specs.get("drop_placeholders", [])

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

    paragraphs = paragrapher.split(text, *args, **kwargs)

    if drop_placeholders:
        paragraphs = clean_placeholders(paragraphs,
                                        placeholders=drop_placeholders)

    if drop_empty:
        paragraphs = [p for p in paragraphs if len(p) > 0]

    if include_span:
        indices = find_substring_indices(text, paragraphs)
        paragraphs = list(zip(indices, paragraphs))

    if as_tuples:
        paragraphs = add_id(paragraphs)

    return paragraphs
