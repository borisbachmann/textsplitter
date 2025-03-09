from typing import Union, Optional, List, Protocol, Any

from .backends import PARA_SEGMENTER_MAP


class ParaSegmenterProtocol(Protocol):
    """
    Protocol for custom paragraph segmenters to implement.

    Args:
        data (List[str]): List of strings to split into sentences.

    Returns:
        List[List[str]]: List of lists of sentences as strings with one
            list of sentences for each input string.
    """
    def __call__(self,
                 data: List[str],
                 *args,
                 **kwargs
                 ) -> List[List[str]]:
        ...

class Paragrapher:
    """
    Paragraph splitter class that wraps around different paragraph segmenters.
    Takes a text or list of texts and returns a list of paragraphs for each
    text. Initialized with the name of a built-in segmenter and a corresponding
    specs. Alternatively, a custom segmenter callable that
    implements the SegmenterProtocol can be passed.

    Args:
        segmenter (Union[str, SegmenterProtocol]): Name of a built-in segmenter
            or a custom segmenter callable implementing the SegmenterProtocol.
        language_or_model (Optional[str]): specs for built-in segmenters.
    """
    def __init__(self,
                 segmenter: Union[str, ParaSegmenterProtocol],
                 specs: Optional[Any] = None
                 ):
        if isinstance(segmenter, str):
            if segmenter not in PARA_SEGMENTER_MAP:
                raise ValueError(f"Invalid segmenter '{segmenter}'. "
                                 f"Must be in "
                                 f"{list(PARA_SEGMENTER_MAP.keys())}.")
            if specs is not None:
                self._segmenter = PARA_SEGMENTER_MAP[segmenter](specs)
            else:
                self._segmenter = PARA_SEGMENTER_MAP[segmenter]()
        elif callable(segmenter):
            self._segmenter = segmenter
        else:
            raise ValueError("Segmenter must be a string or callable. Custom "
                             "callables must implement the SegmenterProtocol.")

    def split(self,
              data: Union[str, List[str]],
              *args, **kwargs
              ) -> Union[List[str], List[List[str]]]:
        """
        Split text into paragraphs. If data is a string, return a list of
        strings, if data is a list of strings, return a list of lists of
        strings. Returned paragraphs are stripped of leading and trailing
        whitespace and empty strings are removed.

        Args:
        data (Union[str, List[str]]): Text to split into paragraphs. If a list of
            strings, each string is split into paragraphs separately.
        show_progress (bool): Show progress bar if True.

        Returns:
        Union[List[str], List[List[str]]]: List of paragraphs as strings (if
            input is a string) or list of lists of paragraphs as strings (if
            input is a list of strings).
        """
        if isinstance(data, str):
            # supress progress bar for single texts
            if "show_progress" in kwargs:
                kwargs.pop("show_progress")
            # wrap to ensure that the segmenter receives a list
            paragraphs = self._segmenter([data], *args, **kwargs)
            paragraphs = self._postprocess(paragraphs)
            # unwrap to return a single list
            return paragraphs[0]
        if isinstance(data, list):
            if not data:
                return []
            if all([isinstance(e, str) for e in data]):
                paragraphs = self._segmenter(data, *args, **kwargs)
                paragraphs = self._postprocess(paragraphs)
                return paragraphs
        raise ValueError("Data must be either string or list of strings only.")

    def _postprocess(self,
                     paragraphs: List[List[str]]
                    ) -> List[List[str]]:
        """
        Clear away irregularities in the paragraphs lists produced by different
        paragraphs segmenters. Removes leading and trailing whitespace and empty
        strings.

        Args:
            paragraphs (List[List[str]]): List of lists of paragraphs as
                strings.

        Returns:
            List[List[str]]: List of lists of paragraphs as strings with leading
                and trailing whitespace removed and empty strings removed.
        """
        return [[s.strip() for s in para_list if s.strip()]
                for para_list in paragraphs]
