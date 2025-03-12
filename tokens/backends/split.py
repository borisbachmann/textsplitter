from typing import List, Dict

from tqdm.auto import tqdm


class SplitTokenSegmenter:
    """
    Token segmenter that splits texts into tokens based on whitespace.

    Functions as a baseline token segmenter.
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self,
                 texts: List[str],
                 show_progress: bool = False,
                 **kwargs
                 ) -> List[List[Dict]]:
        """
        Split texts into tokens based on whitespace.

        Args:
            texts (List[str]): List of texts to split into tokens.
            show_progress (bool): Show progress bar.
            kwargs: Additional arguments for tokenization (not used).

        Returns:
            List[List[Dict]]: List of lists of tokens as dictionaries with one
                list of tokens for each input text and token information in dicts.
                Dicts include the following keys:
                - "token_id": Token ID
                - "token": Token text
        """
        if show_progress:
            iterator = tqdm(texts, desc="Tokenizing texts")
        else:
            iterator = texts
        token_dicts = self._extract_tokens(iterator)
        return token_dicts

    def _extract_tokens(self,
                        texts: List[str]
                        ) -> List[List[Dict]]:
        """
        Split texts into tokens based on whitespace.

        Args:
            texts: List[str]: List of texts to split into tokens.

        Returns:
            List[List[Dict]]: List of lists of tokens as dictionaries with one
                list of tokens for each input text and token information in dicts.
                Dicts include the following keys:
                - "token_id": Token ID
                - "token": Token text
        """
        tokens = [text.split() for text in texts]
        token_dicts = [
            [{"token_id": idx, "token": token}
             for idx, token in enumerate(token_list)]
            for token_list in tokens
        ]
        return token_dicts
