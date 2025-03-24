from typing import Optional, Dict, Any, List, Union, Tuple

import pandas as pd
from tqdm.auto import tqdm

from textsplitter.dataframes.functions import cast_to_df
from textsplitter.dataframes import columns
from textsplitter.utils import find_substring_indices
from .tokenizer import Tokenizer

# As TokenHandler outputs can get quite complex, define possible output
# structures here
TokenStr = str
TokenID = int
TokenSpan = Tuple[int, int]
TokenMetadata = Dict[str, Union[str, int]]

# Possible token output types
TokenFormats = Union[
    TokenStr,
    Tuple[TokenStr],
    Tuple[TokenID, TokenStr],
    Tuple[TokenSpan, TokenStr],
    Tuple[TokenStr, TokenMetadata],
    Tuple[TokenID, TokenSpan, TokenStr],
    Tuple[TokenID, TokenStr, TokenMetadata],
    Tuple[TokenID, TokenSpan, TokenStr, TokenMetadata]
]


class TokenHandler:
    def __init__(self,
                 token_specs: Optional[Dict[str, Any]] = None
                 ):
        token_specs = token_specs or {}
        self.tokenizer = self._initialize_splitter(token_specs)

    def _initialize_splitter(self,
                             token_specs: Optional[Dict[str, Any]] = None
                             ) -> Tokenizer:
        tokenizer = token_specs.pop("tokenizer", "split")

        return Tokenizer(tokenizer, specs=token_specs)

    def split(self,
              text: str,
              include_span: bool = False,
              as_tuples: bool = False,
              include_metadata: bool = False,
              **kwargs
              ) -> Union[List[str], List[tuple]]:
        tokens = self.tokenizer.split(text, show_progress=False, **kwargs)
        formatted_tokens = format_tokens(tokens=tokens,
                                         text=text,
                                         as_tuples=as_tuples,
                                         include_span=include_span,
                                         include_metadata=include_metadata
                                         )

        return formatted_tokens


    def split_list(self,
                   texts: List[str],
                   include_span: bool = False,
                   as_tuples: bool = False,
                   include_metadata: bool = False,
                   **kwargs
                   ) -> List[List[TokenFormats]]:
        show_progress = kwargs.pop("show_progress", False)
        token_lists = self.tokenizer.split(texts,
                                           show_progress=show_progress,
                                           **kwargs)
        if show_progress:
            iterator = tqdm(zip(texts, token_lists),
                            desc="Formatting tokens",
                            total=len(texts))
        else:
            iterator = zip(texts, token_lists)
        formatted_tokens = [format_tokens(tokens=token_list,
                                          text=text,
                                          as_tuples=as_tuples,
                                          include_span=include_span,
                                          include_metadata=include_metadata
                                          )
                            for text, token_list in iterator]
        return formatted_tokens

    def split_df(self,
                 input_df: pd.DataFrame,
                 text_column: str = columns.TEXT_COL,
                 drop_text: bool = True,
                 mathematical_ids: bool = False,
                 include_span: bool = False,
                 include_metadata: bool = False,
                 **kwargs
                 ) -> pd.DataFrame:
        texts = input_df[text_column].tolist()
        tokens = self.split_list(texts,
                                 as_tuples=True,
                                 include_span=include_span,
                                 include_metadata=include_metadata,
                                 **kwargs
                                 )

        return cast_to_df(
            input_df=input_df,
            segments=tokens,
            base_column=columns.TOKEN_COL,
            text_column=text_column,
            drop_text=drop_text,
            mathematical_ids=mathematical_ids,
            include_span=include_span,
            include_metadata=include_metadata,
        )


def format_tokens(tokens: List[Dict],
                  text: str,
                  as_tuples: bool = False,
                  include_span: bool = False,
                  include_metadata: bool = False
                  ) -> List[TokenFormats]:
    if not tokens:
        return []

    # Compute span indices if not provided
    need_span_fallback = not all("start_idx" in token and "end_idx" in token for token in tokens)
    indices = find_substring_indices(text, [token["token"] for token in tokens]) if need_span_fallback else None

    formatted_tokens = []

    for idx, token in enumerate(tokens):
        token_id = token.get("token_id", idx)
        token_text = token["token"]
        start_end = (token["start_idx"], token["end_idx"]) if "start_idx" in token and "end_idx" in token else indices[
            idx]
        metadata = {k: v for k, v in token.items() if k not in {"token_id", "token", "start_idx", "end_idx"}}

        if any(e for e in [as_tuples, include_span, include_metadata]):
            entry = (token_text,)
        else:
            entry = token_text

        if include_span:
            entry = (start_end, *entry)

        if as_tuples:
            entry = (token_id, *entry)

        if include_metadata:
            entry = (*entry, metadata)

        formatted_tokens.append(entry)

    return formatted_tokens
