from typing import List, Dict

import spacy
from spacy.tokens import Token, Doc
from tqdm.asyncio import tqdm

from text_splitter.tokens.constants import SENT_I, MANDATORY_ATTRS, OPTIONAL_ATTRS, DERIVED_ATTRS, END_I


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
