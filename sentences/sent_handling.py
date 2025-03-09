import pandas as pd
from tqdm.auto import tqdm

from ..dataframes.columns import TEXT_COL, SENT_COL
from ..paragraphs.para_handling import ParagraphHandler
from .sentencizer import Sentencizer
from ..utils import add_id, find_substring_indices
from text_splitter.dataframes.functions import cast_to_df

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()


class SentenceHandler:
    def __init__(self, sent_specs, para_specs):
        if sent_specs is None:
            sent_specs = {}
        self.splitter = self._initialize_splitter(
            sent_specs.get("sentencizer", ("pysbd", "de")))

        self.paragrapher = ParagraphHandler(para_specs)

    def _initialize_splitter(self, specs):
        if isinstance(specs, tuple):
            return Sentencizer(*specs)
        elif callable(specs):
            return Sentencizer(specs)
        else:
            raise ValueError("Sentencizer must be a tuple of strings "
                             "specifying a  built-in type with language/model "
                             "or a custom callable.")

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
        paragraphs = self.paragrapher.split(
            text, drop_placeholders=drop_placeholders)

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
        texts = input_df[column].tolist()
        sentences = self.split_list(texts,
                                    as_tuples=True,
                                    include_span=include_span,
                                    **kwargs
                                    )

        return cast_to_df(
            input_df=input_df,
            segments=sentences,
            base_column=SENT_COL,
            text_column=column,
            drop_text=drop_text,
            mathematical_ids=mathematical_ids,
            include_span=include_span,
        )
