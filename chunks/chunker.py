from typing import Union, List, Protocol, Optional

from .chunk_backends import CHUNK_SEGMENTER_MAP


class ChunkSegmenterProtocoll(Protocol):
    """
    Protocol for custom sentence segmenters to implement.

    Args:
        data: List[str]: List of strings to split into sentences.
        show_progress: bool: Show progress bar if True.

    Returns:
        List[List[str]]: List of lists of sentences as strings with one
            list of sentences for each input string.
    """
    def __call__(self,
                 data: List[str],
                 show_progress: bool=False
                 ) -> List[List[str]]:
        ...

class Chunker:
    """
    Sentence splitter class that wraps around different chunk segmenters.
    Takes a text or list of texts and returns a list of chunks for each
    text. Initialized with the name of a built-in chunker and a corresponding
    transformer model name. Alternatively, a custom segmenter callable that
    implements the ChunkSegmenterProtocoll can be passed.

    Args:
        segmenter: Union[str, SegmenterProtocol]: Name of a built-in segmenter
            or a custom segmenter callable implementing the SegmenterProtocol.
        language_or_model: Optional[str]: Language or model name for built-in
            segmenters.
    """
    def __init__(self,
                 segmenter: Union[str, ChunkSegmenterProtocoll],
                 model: Optional[str] = None,
                 specs: Optional[dict] = None,
                 ):
        if isinstance(segmenter, str):
            if segmenter not in CHUNK_SEGMENTER_MAP:
                raise ValueError(f"Invalid segmenter '{segmenter}'. "
                                 f"Must be in {CHUNK_SEGMENTER_MAP.keys()}.")
            if model is None:
                raise ValueError("A transfomer model must be provided for "
                                 "built-in chunk segmenters.")
            if specs is None:
                specs = {}
            self._segmenter = CHUNK_SEGMENTER_MAP[segmenter](model, **specs)
        elif callable(segmenter):
            self._segmenter = segmenter
        else:
            raise ValueError("Segmenter must be a string or callable. Custom "
                             "callables must implement the SegmenterProtocol.")

    def split(self,
              data: Union[str, List[str]],
              show_progress: bool = False
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
            # wrap to ensure that the segmenter receives a list
            sentences = self._segmenter([data])
            sentences = self._postprocess(sentences)
            # unwrap to return a single list
            return sentences[0]
        if isinstance(data, list):
            if not data:
                return []
            if all([isinstance(e, str) for e in data]):
                sentences = self._segmenter(data, show_progress)
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
