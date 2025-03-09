from typing import Union, List, Tuple

import pandas as pd
from tqdm.auto import tqdm

from ..dataframes import columns
from ..paragraphs.handling import ParagraphHandler
from .sentencizer import Sentencizer
from .backends import SentSegmenterProtocol
from ..utils import add_id, find_substring_indices
from ..dataframes.functions import cast_to_df

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()


class SentenceHandler:
    """
    Class for handling sentence splitting of text data. Aims to split text data
    into sentences as perceived by human readers. Texts are first split into
    paragraphs before being processed into sentences. Paragraph and sentence
    splitting is delegated to dedicated classes under the hood. Can handle
    single strings, lists of strings, and pandas dataframes with a text data
    column and return output of an according, more fine-grained type: Single
    lists of strings, lists of lists of strings, or dataframes with one row per
    sentence.

    Args:
        sent_specs (dict): Specifications for sentence splitting.
        para_specs (dict): Specifications for paragraph splitting.
    """
    def __init__(self, sent_specs, para_specs):
        if sent_specs is None:
            sent_specs = {}
        self.sentencizer = self._initialize_splitter(
            sent_specs.get("sentencizer", ("pysbd", "de")))

        self.paragrapher = ParagraphHandler(para_specs)

    def _initialize_splitter(
            self,
            sentencizer: Union[tuple, SentSegmenterProtocol]
            ) -> Sentencizer:
        """
        Load internal Sentencizer based upon the sentencizer specifications #
        passed as argument.

        Args:
            sentencizer (Union[tuple, SentSegmenterProtocol]): Sentencizer
                specifications. Can be a tuple of strings specifying a built-in
                type with a corresponding language or model, or a custom
                callable implementing the SentSegmenterProtocol.

        Returns:
            Sentencizer: Sentencizer instance
        """
        if isinstance(sentencizer, tuple):
            return Sentencizer(*sentencizer)
        elif callable(sentencizer):
            return Sentencizer(sentencizer)
        else:
            raise ValueError("Sentencizer must be a tuple of strings "
                             "specifying a built-in type with language/model "
                             "or a custom callable implementing the "
                             "SentSegmenterProtocol.")

    def split(self,
              text: str,
              as_tuples: bool = False,
              include_span: bool = False,
              **kwargs
              ) -> Union[List[str], List[Tuple[int, str]],
                         List[Tuple[Tuple[int, int], str]]
                        ]:
        """
        Split a string containing natural language data into sentences. Returns
        a list of sentences as strings. Optionally, return a list of tuples
        with sentence index and sentence as strings and/or start and end indices
        of sentences in the original text.

        Args:
            text (str): Text to split into sentences.
            as_tuples: (bool): Return sentences as tuples with index and sentence.
            include_span: (bool): Return sentences as tuples with index, start and
                end indices of sentence in original text.
            kwargs: Additional arguments for sentencizer

        Returns:
            Union[List[str], List[Tuple[int, str]], List[Tuple[Tuple[int, int],
                str]]: List of sentences as strings, or list of tuples including
                ids and/or span information.
        """
        drop_placeholders = kwargs.pop("drop_placeholders", [])

        # split into paragraphs first
        paragraphs = self.paragrapher.split(
            text, drop_placeholders=drop_placeholders)

        # process paragraphs individually into sentences
        sentences = self.sentencizer.split(paragraphs,
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
                   ) -> Union[List[List[str]], List[List[Tuple[int, str]]],
                              List[List[Tuple[Tuple[int, int], str]]]
                              ]:
        """
        Split a list of strings containing natural language data into sentences.
        Returns a list of sentences per text as lists of strings. Optionally,
        returns lists of tuples also including sentences index and/or start and
        end indices of sentences in the original text.

        Args:
            texts (list): List of strings to split into sentences.
            as_tuples (bool): Return sentences as tuples with index and sentence.
            include_span (bool): Return sentences as tuples with index, start and
                end indices of sentence in original text.
            kwargs: Additional arguments for sentencizer

        Returns:
            Union[List[List[str]], List[List[Tuple[int, str]]],
                List[List[Tuple[Tuple[int, int], str]]]]: List of lists with
                one list for each original text, including sentences as strings
                or tuples including ids and/or span information.
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

        sentences = [self.sentencizer.split(data=para_list,
                                            as_tuples=as_tuples,
                                            include_span=include_span,
                                            **kwargs
                                            )
                     for para_list in iterator
                     ]

        # flatten sentence lists for paragraphs into one list for whole text
        sentences = [[sentence for paragraph in para_list
                      for sentence in paragraph]
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
                 text_column: str = columns.TEXT_COL,
                 drop_text: bool = True,
                 mathematical_ids: bool = False,
                 include_span: bool = False,
                 **kwargs
                 ) -> pd.DataFrame:
        """
        In a pandas dataframe containing a column with text data, insert three
        new columns with individual embeddings derived from text data, number of
        embeddings per data, and embeddings IDs. DataFrame is exploded to one
        row per sentence, keeping sentence together with original text data.
        Optionally, drop the original data column.

        Args:
            input_df (pd.DataFrame): DataFrame with original text data
            text_column (str): Column name with text data
            drop_text (bool): Whether to drop the original text column
            mathematical_ids (bool): Whether to increment data IDs by 1 to
                avoid 0
            include_span (bool): Whether to include span information in output
            kwargs: Additional arguments for sentencizer

        Returns:
            pd.DataFrame: DataFrame with sentence data as rows
        """
        texts = input_df[text_column].tolist()
        sentences = self.split_list(texts,
                                    as_tuples=True,
                                    include_span=include_span,
                                    **kwargs
                                    )

        return cast_to_df(
            input_df=input_df,
            segments=sentences,
            base_column=columns.SENT_COL,
            text_column=text_column,
            drop_text=drop_text,
            mathematical_ids=mathematical_ids,
            include_span=include_span,
        )
