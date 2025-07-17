from typing import Dict, Any, Union, List, Optional

from .backends import TokenSegmenterProtocol


class Tokenizer(object):
    def __init__(self,
                 segmenter: TokenSegmenterProtocol,
                 # specs: Optional[Dict[str, Any]] = None, ## CORRECT THIS LATER
                 ):
        self._backend = segmenter

    def split(self,
              data: Union[str, List[str]],
              show_progress: bool = False,
              **kwargs
              ) -> Union[List[Dict], List[List[Dict]]]:
        if isinstance(data, str):
            # wrap to ensure that the segmenter receives a list
            tokens = self._backend([data], **kwargs)
            # unwrap to return a single list
            return tokens[0]
        if isinstance(data, list):
            if not data:
                return []
            if all([isinstance(e, str) for e in data]):
                tokens = self._backend(data,
                                       show_progress=show_progress,
                                       **kwargs
                                       )
                return tokens
        raise ValueError("Data must be either string or list of strings only.")
