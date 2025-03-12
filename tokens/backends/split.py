from typing import List, Dict

from tqdm.asyncio import tqdm


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
