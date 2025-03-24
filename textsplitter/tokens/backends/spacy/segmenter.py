from typing import List, Dict

import spacy
from spacy.tokens import Token, Doc
from tqdm.auto import tqdm

from . import constants as spacy_const


class SpacyTokenSegmenter:
    """
    Token segmenter that uses spaCy to split texts into tokens.

    Args:
        model: str: SpaCy model to use for tokenization.
    """

    def __init__(self, model: str):
        Token.set_extension(spacy_const.SENT_I, default=None, force=True)
        self._nlp = spacy.load(model)

    def __call__(self,
                 texts: List[str],
                 show_progress: bool = False,
                 **kwargs
                 ) -> List[List[Dict]]:
        """
        Split texts into tokens using spaCy.

        Args:
            texts: List[str]: List of texts to split into tokens.
            show_progress: bool: Show progress bar.
            kwargs: Additional arguments for tokenization (not used).

        Returns:
            List[List[Dict]]: List of lists of tokens as dictionaries with one
                list of tokens for each input text and token information in dicts.
                Dicts include the following keys:
                - "token_id": Token ID
                - "token": Token text
                - additional token annotation columns from spaCy,
                    see spacy.constants.py
        """
        if show_progress:
            iterator = tqdm(texts, desc="Processing docs")
        else:
            iterator = texts
        docs = list(self._nlp.pipe(iterator))
        tokens = self._extract_tokens(docs, show_progress)
        token_dicts = self._compile_token_dicts(tokens, show_progress)
        return token_dicts

    def _extract_tokens(
            self,
            docs: List[Doc],
            show_progress: bool = False
            ) -> List[List[Token]]:
        if show_progress:
            iterator = tqdm(docs, desc="Extracting tokens")
        else:
            iterator = docs

        tokens = []
        for doc in iterator:
            for idx, sent in enumerate(doc.sents):
                for token in sent:
                    setattr(token._, spacy_const.SENT_I, idx)
            tokens.append([token for token in doc])


        #tokens = [[token for token in doc] for doc in docs]
        return tokens

    def _compile_token_dicts(
            self,
            tokens: List[List[Token]],
            show_progress: bool = False
            ) -> List[List[Dict]]:
        def normalize_key(key):
            key_mapping = (
                    spacy_const.MANDATORY_ATTRS |
                    spacy_const.OPTIONAL_ATTRS |
                    spacy_const.DERIVED_ATTRS
            )
            return key_mapping.get(key, key)

        def info(token):
            mandatory = {
                normalize_key(attr): getattr(token, attr)
                for attr in spacy_const.MANDATORY_ATTRS
            }
            sent_i = {
                normalize_key(spacy_const.SENT_I):
                    getattr(token._, spacy_const.SENT_I)
            }
            optional = {
                normalize_key(attr): getattr(token, attr)
                for attr in spacy_const.OPTIONAL_ATTRS}
            end_i = {
                normalize_key(spacy_const.END_I):
                    token.idx + len(token.text)
            }
            all = sent_i | mandatory | optional | end_i
            return all

        if show_progress:
            iterator = tqdm(tokens, desc="Compiling token information")
        else:
            iterator = tokens
        token_dicts = [
            [info(token) for token in token_list] for token_list in iterator
        ]
        return token_dicts
