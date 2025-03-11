from typing import Union, List, Optional

from .backends import SENT_SEGMENTER_MAP, SentSegmenterProtocol


class Sentencizer:
    """
    Sentence splitter class that wraps around different sentence segmenters.
    Takes a text or list of texts and returns a list of sentences for each
    text. Initialized with the name of a built-in segmenter and a corresponding
    language or model name. Alternatively, a custom segmenter callable that
    implements the SegmenterProtocol can be passed.

    Args:
        segmenter: Union[str, SegmenterProtocol]: Name of a built-in segmenter
            or a custom segmenter callable implementing the SegmenterProtocol.
        language_or_model: Optional[str]: Language or model name for built-in
            segmenters.
    """
    def __init__(self,
                 segmenter: Union[str, SentSegmenterProtocol],
                 language_or_model: Optional[str] = None
                 ):
        if isinstance(segmenter, str):
            if segmenter not in SENT_SEGMENTER_MAP:
                raise ValueError(f"Invalid segmenter '{segmenter}'. "
                                 f"Must be in "
                                 f"{list(SENT_SEGMENTER_MAP.keys())}.")
            if language_or_model is None:
                raise ValueError("Language or model must be provided for "
                                 "built-in segmenters.")
            self._segmenter = SENT_SEGMENTER_MAP[segmenter](language_or_model)
        elif callable(segmenter):
            self._segmenter = segmenter
        else:
            raise ValueError("Segmenter must be a string or callable. Custom "
                             "callables must implement the "
                             "SentSegmenterProtocol.")

    def split(self,
              data: Union[str, List[str]],
              **kwargs
              ) -> Union[List[str], List[List[str]]]:
        """
        Split text into sentences. If data is a string, return a list of
        strings, if data is a list of strings, return a list of lists of
        strings. Returned sentences are stripped of leading and trailing
        whitespace and empty strings are removed.

        Args:
        data: Union[str, List[str]]: Text to split into sentences. If a list of
            strings, each string is split into sentences separately.
        show_progress: bool: Show progress bar if True.

        Returns:
        Union[List[str], List[List[str]]]: List of sentences as strings (if
            input is a string) or list of lists of sentences as strings (if
            input is a list of strings).
        """
        if isinstance(data, str):
            # supress progress bar for single texts
            if "show_progress" in kwargs:
                kwargs.pop("show_progress")
            # wrap to ensure that the segmenter receives a list
            sentences = self._segmenter([data], **kwargs)
            sentences = self._postprocess(sentences)
            # unwrap to return a single list
            return sentences[0]
        if isinstance(data, list):
            if not data:
                return []
            if all([isinstance(e, str) for e in data]):
                sentences = self._segmenter(data, **kwargs)
                sentences = self._postprocess(sentences)
                return sentences
        raise ValueError("Data must be either string or list of strings only.")

    def _postprocess(self,
                     sentences: List[List[str]]
                    ) -> List[List[str]]:
        """
        Clear away irregularities in the sentence lists produced by different
        sentence segmenters. Removes leading and trailing whitespace and empty
        strings.
        """
        return [[s.strip() for s in sent_list if s.strip()]
                for sent_list in sentences]
