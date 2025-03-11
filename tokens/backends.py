from typing import List, Dict, Protocol

import spacy
from spacy.tokens import Token, Doc
from tqdm.auto import tqdm

from .constants import MANDATORY_ATTRS, OPTIONAL_ATTRS, END_I, SENT_I, DERIVED_ATTRS


class TokenSegmenterProtocol(Protocol):
    """
    Protocol for custom token segmenters to implement.

    Args:
        texts: List[str]: List of texts to split into tokens.

    Returns:
        List[List[Dict]]: List of lists of tokens as dictionaries with one
            list of tokens for each input text and token information in dicts.
            Dicts have to include at least the following keys:
            - "token_id": Token ID
            - "token": Token text
    """
    def __call__(self,
                 texts: List[str],
                 show_progress: bool = False,
                 **kwargs
                 ) -> List[List[Dict]]:
        ...


class SplitTokenSegmenter:
    def __init__(self, **kwargs):
        pass

    def __call__(self,
                 texts: List[str],
                 show_progress: bool = False,
                 **kwargs
                 ) -> List[List[Dict]]:
        if show_progress:
            iterator = tqdm(texts, desc="Tokenizing texts")
        else:
            iterator = texts
        token_dicts = self._extract_tokens(iterator)
        return token_dicts

    def _extract_tokens(self,
                        texts: List[str]
                        ) -> List[List[Dict]]:
        tokens = [text.split() for text in texts]
        token_dicts = [
            [{"token_id": idx, "token": token}
             for idx, token in enumerate(token_list)]
            for token_list in tokens
        ]
        return token_dicts


class SpacyTokenSegmenter:
    def __init__(self, model: str):
        Token.set_extension("sent_i", default=None, force=True)
        self._nlp = spacy.load(model)

    def __call__(self,
                 texts: List[str],
                 show_progress: bool = False,
                 **kwargs
                 ) -> List[List[Dict]]:
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
                    setattr(token._, SENT_I, idx)
            tokens.append([token for token in doc])


        #tokens = [[token for token in doc] for doc in docs]
        return tokens

    def _compile_token_dicts(
            self,
            tokens: List[List[Token]],
            show_progress: bool = False
            ) -> List[List[Dict]]:
        def normalize_key(key):
            key_mapping = MANDATORY_ATTRS | OPTIONAL_ATTRS | DERIVED_ATTRS
            return key_mapping.get(key, key)

        def info(token):
            mandatory = {normalize_key(attr): getattr(token, attr)
                         for attr in MANDATORY_ATTRS}
            sent_i = {normalize_key(SENT_I): getattr(token._, SENT_I)}
            optional = {normalize_key(attr): getattr(token, attr)
                        for attr in OPTIONAL_ATTRS}
            end_i = {normalize_key(END_I): token.idx + len(token.text)}
            all = sent_i | mandatory | optional | end_i
            return all

        if show_progress:
            iterator = tqdm(tokens, desc="Compiling token information")
        else:
            iterator = tokens
        token_dicts = [[info(token) for token in token_list]
                       for token_list in iterator]
        return token_dicts

# Mapping of segmenter names to built-in segmenter classes
TOKEN_SEGMENTER_MAP = {
    "spacy": SpacyTokenSegmenter,
    "split": SplitTokenSegmenter
}
