# textsplitter

textsplitter provides a common interface for different text segmentation algorithms. It developed as a tool to handle 
segmentation for different NLP and machine learning tasks in the University of Wuppertal's Narrative Futures as well 
as some personal projects and is designed to be easily extensible. Under the hood, it provides access to some 
built-in techniques from popular packages like spaCy, wtsplit's SaT models or pySBD, rules-based 
approaches as well as embedding-based chunking approaches. In some rules-based techniques, It is specialized for GERMAN 
language, but can be adjusted for other languages as well.

Currently, textsplitter supports segmentation into the following types: (1) tokens, (2) sentences, (3) paragraphs 
and (4) chunks. The first three are considered "natural" segmentation types representing the most common ways to 
think about smaller textual units from the perspective of a human reader. The last type, chunks, are segments 
constructed out of single consecutive sentences within a larger text.

The common interface can is designed to be extended on two levels: Individual segmentation algorithms for each of 
the types, as well as new types of segmentation.

The package is still work in progress, with some interface irregularities, missing documentation and tests.

## Installation

...

## Usage

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

example_sents = splitter.sentences(example_text, inlcude_span=True)

print(example_sents)
```

Different segmentation algorithms can be implemented upon initialization by passing accoridng specs as a dict:

```Python
from textsplitter import TextSplitter

# for spacy, language models have to be pre-installed in your environment
splitter = TextSplitter(token_specs={"tokenizer": "spacy", "model": "de_core_news_sm"})

example_text = "Sag' wo die Soldaten sind. Wo sind sie geblieben? Sag' wo die Soldaten sind, was ist geschehen?"

example_tokens = splitter.tokens(example_text)

print(example_tokens)
```

For a more detailed overview of the package's functionality, please refer to the `usage` notebook in the `notebooks`
