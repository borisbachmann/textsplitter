from typing import Dict, Any, Union, List, Optional

from .backends import TokenSegmenterProtocol, TOKEN_SEGMENTER_MAP


class Tokenizer(object):
    def __init__(self,
                 segmenter: Union[str, TokenSegmenterProtocol],
                 specs: Optional[Dict[str, Any]] = None,
                 ):
        specs = specs or {}
        if isinstance(segmenter, str):
            built_in_segmenter = TOKEN_SEGMENTER_MAP.get(segmenter, None)
            if not built_in_segmenter:
                raise ValueError(f"Invalid segmenter '{segmenter}'. "
                                 f"Must be in "
                                 f"{list(TOKEN_SEGMENTER_MAP.keys())}.")
            self._segmenter = built_in_segmenter(**specs)
        elif callable(segmenter):
            if not specs:
                self._segmenter = segmenter
            else:
                self._segmenter = segmenter(**specs)
        else:
            raise ValueError("Segmenter must be a string or callable. Custom "
                             "callables must implement the "
                             "TokenSegmenterProtocol.")

    def split(self,
              data: Union[str, List[str]],
              show_progress: bool = False,
              **kwargs
              ) -> Union[List[Dict], List[List[Dict]]]:
        if isinstance(data, str):
            # wrap to ensure that the segmenter receives a list
            tokens = self._segmenter([data], **kwargs)
            # unwrap to return a single list
            return tokens[0]
        if isinstance(data, list):
            if not data:
                return []
            if all([isinstance(e, str) for e in data]):
                tokens = self._segmenter(data,
                                         show_progress=show_progress,
                                         **kwargs
                                         )
                return tokens
        raise ValueError("Data must be either string or list of strings only.")
