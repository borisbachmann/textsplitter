from typing import List, Union, Tuple

import pandas as pd

from .columns import (TEXT_COL, id_pattern, span_pattern, n_pattern,
                      multi_pattern)


def column_list(base_col, text_col):
    """Make column list for dataframe based upon base column name."""
    return [
        "file",
        "ID",
        text_col,
        id_pattern(base_col),
        base_col,
        span_pattern(base_col),
        n_pattern(base_col),
        "meta" # remove later
    ]


def cast_to_df(
    input_df: pd.DataFrame,
    segments: List[Union[str, Tuple[Tuple[int, int], str]]],
    base_column: str,
    text_column: str = TEXT_COL,
    include_span: bool = False,
    include_metadata: bool = False,
    drop_text: bool = True,
    mathematical_ids: bool = False
    ) -> pd.DataFrame:
    """
    From a list of segments and a dataframe with original text data, create a
    new segment dataframe with one row per segment.

    Args:
        input_df (pd.DataFrame): DataFrame with original text data
        segments (List[Union[str, Tuple[Tuple[int, int], str]]]): List of
            segments as strings or tuples including span information
        base_column (str): Base column name for segment data
        text_column (str): Column name with original text data
        include_span (bool): Whether to include span information in output
        drop_text (bool): Whether to drop the original text column
        mathematical_ids (bool): Whether to increment segment IDs by 1 to
            avoid 0

    Returns:
        pd.DataFrame: DataFrame with segment data as rows
    """
    multi_column = multi_pattern(base_column)
    id_column = id_pattern(base_column)
    span_column = span_pattern(base_column)
    n_column = n_pattern(base_column)

    # make a clean copy with reset index to avoid problems with slices
    df = input_df.copy().reset_index(drop=True)

    df[multi_column] = pd.Series(segments)
    df = df.explode(multi_column).reset_index(drop=True)

    # unpack segment data into separate columns
    segment_df = pd.DataFrame(df[multi_column].tolist())
    split_columns = [id_column, base_column]
    if include_span:
        split_columns.insert(1, span_column)
    if include_metadata:
        split_columns.append("meta")
    df[split_columns] = segment_df

    # handle metadta
    if include_metadata:
        meta_df = pd.json_normalize(df.pop("meta"))
        df[meta_df.columns] = meta_df

    # count segments per text
    df[n_column] = df.groupby(text_column)[text_column].transform("size")

    if mathematical_ids:
        df[id_column] = df[id_column].map(lambda x: x + 1)

    # keep only desired columns for output dataframe
    columns = [c for c in column_list(base_column, text_column)
               if c in df.columns]

    # add metadata columns if present
    if include_metadata:
        columns.extend(meta_df.columns.tolist())

    if drop_text:
        columns.remove(text_column)

    return df[columns]
