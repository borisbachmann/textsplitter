import pandas as pd
from tqdm.auto import tqdm

from constants import (TEXT_COL, PARA_COL, PARAS_COL, PARA_N_COL, PARA_ID_COL,
                       BULLETS)
from .para_splitting import split_clean_paragraphs
from regex_patterns import PARAGRAPH_PATTERN_SIMPLE, ENUM_PATTERN_NO_DATE
from utils import column_list, increment_ids, add_id

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()


def split_paragraphs(
        input_df: pd.DataFrame,
        column: str = TEXT_COL,
        function: callable = None,
        drop_text: bool = True,
        mathematical_ids: bool = False,
        drop_empty: bool = True
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
                                                  mathematical_ids=mathematical_ids,
                                                  drop_empty=drop_empty
                                                  )
    df = df.explode(PARAS_COL)
    df[[PARA_ID_COL, PARA_COL]] = df[PARAS_COL].tolist()
    columns = [c for c in column_list(PARA_COL, column) if c in df.columns]

    if drop_text:
        columns.remove(TEXT_COL)

    return df[columns]

def find_paragraphs(
        dataframe: pd.DataFrame,
        column: str = TEXT_COL,
        function: callable = None,
        mathematical_ids: bool = False,
        drop_empty: bool = True
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
                                              drop_empty=drop_empty,
                                              as_tuples=True
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
        function: callable = None,
        drop_empty: bool = True,
        as_tuples: bool = False,
        ) -> list:
    """Split a string containing natural language data into paragraphs. Returns
    a list of paragraphs. Optionally, return a list of tuples with paragraph
    ids and data.

    Paragraphs are split based on a function passed as an argument. If no
    function is provided, a standard function will be selected under the hood.
    """
    if function is not None:
        paragraphs = function(text)
    else:
        paragraphs = split_clean_paragraphs(
            text,
            merge_bullets=True,
            paragraph_pattern=PARAGRAPH_PATTERN_SIMPLE,
            enum_pattern=ENUM_PATTERN_NO_DATE,
            bullets=BULLETS
        )

    if drop_empty:
        paragraphs = [p for p in paragraphs if len(p) > 0]

    if as_tuples:
        paragraphs = add_id(paragraphs)

    return paragraphs
