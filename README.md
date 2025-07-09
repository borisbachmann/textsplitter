# textsplitter

textsplitter provides a common interface for different text segmentation algorithms. It is designed to be easily 
extensible. Under the hood, it provides access to some built-in techniques from popular packages like spaCy, wtpsplit's 
SaT models or pySBD, rules-based approaches as well as embedding-based chunking approaches. In some rules-based 
techniques, It is specialized for GERMAN language, but can be adjusted for other languages as well.

Currently, textsplitter supports segmentation into the following types: (1) tokens, (2) sentences, (3) paragraphs 
and (4) chunks. The first three are considered "natural" segmentation types representing the most common ways to 
think about smaller textual units from the perspective of a human reader. The last type, chunks, are segments 
constructed out of single consecutive sentences within a larger text.

The common interface is designed to be extended on two levels: Individual segmentation algorithms for each of 
the types, as well as new types of segmentation.

The package is work in progress, with missing documentation and tests. API may change in the future.

## Key Features
- Common interface for multiple segmentation algorithms
- Works with strings, lists, pandas DataFrames
- Four segmentation types: tokens, sentences, paragraphs, chunks
- Extensible architecture for custom algorithms
- some built-in segmentation algorithms specialized for German language

## Installation

You have to pull the package directly from the repository and install it via pip.

## Usage

### Basic

For all segmentation tasks, textsplitter' basic functionality is called via the `tokens`, `sentences`, 
`paragraphs`and `chunks` methods like so:

```Python
from textsplitter import TextSplitter

splitter = TextSplitter()
example_text = "Sag' wo die Soldaten sind. Wo sind sie geblieben? Sag' wo die Soldaten sind, was ist geschehen?"
example_sents = splitter.sentences(example_text)

print(example_sents)
```

textsplitter works on single texts strings, lists of strings or pandas Series and DataFrames and returns output in a 
corresponding format: Lists of segmented units, lists of lists of segmented units (for Lists and Series) or DataFrames 
with segmented units as one row per unit. Basic units within lists or DataFrames are strings, but additional 
information like token metadata, spans and indices can be included in the output.

```Python
from textsplitter import TextSplitter

splitter = TextSplitter()
example_text = "Sag' wo die Soldaten sind. Wo sind sie geblieben? Sag' wo die Soldaten sind, was ist geschehen?"
example_sents = splitter.sentences(example_text, include_span=True)

print(example_sents)
```

### Segmentation backends

Different segmentation backends can be used upon initialization by passing according specs as a dict:

```Python
from textsplitter import TextSplitter

# for spacy, language models have to be pre-installed in your environment
splitter = TextSplitter(token_specs={"tokenizer": "spacy", "model": "de_core_news_sm"})
example_text = "Sag' wo die Soldaten sind. Wo sind sie geblieben? Sag' wo die Soldaten sind, was ist geschehen?"
example_tokens = splitter.tokens(example_text)

print(example_tokens)
```

Built-in types are not yet documented, but you can pass any custom callable that implements the protocols found in 
`chunks.backends.segmenters.py` (same for `paragraphs`, `sentences` and `tokens` respectively), e.g.:

```Python
# define or import your custom callable somewhere
custom_sentencizer = SomeCallableDefinedBefore()
splitter = TextSplitter(sentence_specs={"sentencizer": custom_sentencizer})
```

**Note on sentences:** TextSplitter splits sentences as a sub-unit of paragraphs, so if you set a sentencizer, you 
might want to set a paragrapher as well, e.g. If you want some raw splitting functionality, you might have to define 
a custom paragrapher returning the whole text as a single paragraph. Otherwise, the default paragrapher is used 
under the hood.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
