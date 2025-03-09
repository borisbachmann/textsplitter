import pandas as pd
from tqdm.auto import tqdm

from .utils import make_indices_from_paragraph
from .paragrapher import Paragrapher
from ..dataframes.columns import TEXT_COL, PARA_COL
from ..utils import add_id, clean_placeholders
from text_splitter.dataframes.functions import cast_to_df

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()


class ParagraphHandler:
    def __init__(self, specs):
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
            ) -> list:
        """Split a string containing natural language data into paragraphs.
        Returns a list of paragraphs. Optionally, return a list of tuples with
        paragraph ids and data.

        Paragraphs are split based on a function passed as an argument. If no
        function is provided, a standard function will be selected under the
        hood.
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
            ) -> list:
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
            list: List of paragraphs per text as list of strings or tuples
            with paragraph ids and data.
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
        column: str = TEXT_COL,
        drop_text: bool = True,
        mathematical_ids: bool = False,
        include_span: bool = False,
        **kwargs
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
        texts = input_df[column].tolist()
        paragraphs = self.split_list(texts,
                                     as_tuples=True,
                                     include_span=include_span,
                                     **kwargs
                                     )

        return cast_to_df(
            input_df=input_df,
            segments=paragraphs,
            base_column=PARA_COL,
            text_column=column,
            drop_text=drop_text,
            mathematical_ids=mathematical_ids,
            include_span=include_span
        )
