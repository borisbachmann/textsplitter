from typing import Dict, Any, Union, List, Tuple

import pandas as pd
from tqdm.auto import tqdm

from .utils import make_indices_from_paragraph
from .paragrapher import Paragrapher
from ..dataframes import columns
from ..utils import add_id, clean_placeholders
from ..dataframes.functions import cast_to_df

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()


class ParagraphHandler:
    def __init__(self, specs: Dict[str, Any]):
        if specs is None:
            specs = {}
        self.splitter = self._initialize_splitter(
            specs.get("paragrapher", "clean"))
        self.drop_placeholders = specs.get("drop_placeholders", [])

    def _initialize_splitter(self, specs):
        if (isinstance(specs, str)
                or callable(specs)
        ):
            return Paragrapher(specs)
        elif isinstance(specs, tuple):
            return Paragrapher(*specs)
        else:
            raise ValueError("Paragrapher must be either a built-in type "
                             "(specified by string or tuple of string and "
                             "specs) or a custom callable.")

    def split(
            self,
            text: str,
            as_tuples: bool = False,
            include_span: bool = False,
            **kwargs
            ) -> Union[List[str], List[Tuple[int, str]],
                       List[Tuple[Tuple[int, int], str]]
                      ]:
        """Split a string containing natural language data into paragraphs.
        Returns a list of paragraphs. Optionally, return a list of tuples with
        paragraph ids and data.

        Args:
            text (str): Text to split into paragraphs
            as_tuples (bool): Return paragraphs as tuples idx, paragraph if True.
            include_span (bool): Include span information in output if True.
            **kwargs: Additional keyword arguments to be passed to the splitter.

        Returns:
            Union[List[str], List[Tuple[int, str]], List[Tuple[Tuple[int, int],
            str]]: List of paragraphs as strings, or list of tuples including
            ids and/or span information.
        """
        drop_placeholders = kwargs.pop("drop_placeholders", [])

        paragraphs = self.splitter.split(text, **kwargs)

        if drop_placeholders:
            paragraphs = clean_placeholders(
                paragraphs, placeholders=drop_placeholders
                )

        if include_span:
            indices = [make_indices_from_paragraph(p, text)
                       for p in paragraphs]
            paragraphs = list(zip(indices, paragraphs))

        if as_tuples:
            paragraphs = add_id(paragraphs)

        return paragraphs

    def split_list(
            self,
            texts: list,
            as_tuples: bool = False,
            include_span: bool = False,
            **kwargs
            ) -> Union[List[List[str]], List[List[Tuple[int, str]]],
                       List[List[Tuple[Tuple[int, int], str]]]]:
        """
        Split a list of strings containing natural language data into
        paragraphs. Returns a list of ppragraphs per text as lists of strings.
        Optionally, returns lists of tuples also including paragraph index
        and/or start and end indices of paragraphs in the original text.

        Args:
            texts: list: List of strings to split into paragraphs.
            as_tuples: bool: Return paragraphs as tuples if True.
            include_span: bool: Include span information in output if True.

        Returns:
            Union[List[List[str]], List[List[Tuple[int, str]]],
                List[List[Tuple[Tuple[int, int], str]]]]: List of paragraphs
                per text as list of strings or tuples including paragraph ids
                and span indices.
        """

        drop_placeholders = kwargs.pop("drop_placeholders", [])
        show_progress = kwargs.get("show_progress", False)

        paragraphs = self.splitter.split(texts, **kwargs)

        if drop_placeholders:
            paragraphs = [
                clean_placeholders(para_list,
                                   placeholders=drop_placeholders)
                 for para_list in paragraphs
                ]

        if include_span:
            if show_progress:
                iterator = tqdm(zip(paragraphs, texts),
                                desc="Adding span indices",
                                total=len(texts))
            else:
                iterator = zip(paragraphs, texts)

            paragraphs = [list(zip([make_indices_from_paragraph(p, text)
                                    for p in para_list], para_list))
                          for para_list, text in iterator]

        if as_tuples:
            paragraphs = [add_id(para_list) for para_list in paragraphs]

        return paragraphs

    def split_df(
        self,
        input_df: pd.DataFrame,
        text_column: str = columns.TEXT_COL,
        drop_text: bool = True,
        mathematical_ids: bool = False,
        include_span: bool = False,
        **kwargs
        ) -> pd.DataFrame:
        """
        In a pandas dataframe containing a column with text data, insert three
        new columns with individual paragraphs derived from text data, number of
        paragraphs per data, and paragraph IDs. DataFrame is exploded to one row
        per paragraph, keeping paragraphs together with original text data.
        Optionally, drop the original data column.

        Args:
            input_df: pd.DataFrame: DataFrame containing text data.
            text_column: str: Column containing text data.
            drop_text: bool: Drop the original text column if True.
            mathematical_ids: bool: Include mathematical IDs in output if True.
            include_span: bool: Include span information in output if True.




        """
        texts = input_df[text_column].tolist()
        paragraphs = self.split_list(texts,
                                     as_tuples=True,
                                     include_span=include_span,
                                     **kwargs
                                     )

        return cast_to_df(
            input_df=input_df,
            segments=paragraphs,
            base_column=columns.PARA_COL,
            text_column=text_column,
            drop_text=drop_text,
            mathematical_ids=mathematical_ids,
            include_span=include_span
        )
