from copy import deepcopy
import inspect
from typing import Optional, Union, List

import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from tqdm.auto import tqdm

from .chunker import Chunker
#from text_splitter.chunks.chunkers.graph_chunker import graph_chunking
#from text_splitter.chunks.chunkers.linear_chunker import linear_chunking
from .chunk_utils import TokenCounter
from ..constants import (TEXT_COL, CHUNK_COL, CHUNKS_COL, CHUNK_N_COL,
                         CHUNK_ID_COL, DEFAULT_SCOPE, DEFAULT_STRATEGY,
                         DEFAULT_RESOLUTION, CHUNK_SPAN_COL)
#from ..embeddings import create_embeddings
#from ..sentences.sent_handling import make_sentences_from_text
from ..utils import column_list, increment_ids, add_id, find_substring_indices

# Enable progress bars for dataframe .map and .apply methods
tqdm.pandas()


class DummyChunkModule:
    def __init__(self):
        pass

    def chunk(self,
              text: str,
              as_tuples: bool = False,
              include_span: bool = False
              ) -> list:
        print("Chunking not available.")
        chunks = [text]

        if include_span:
            indices = [(0, len(text))]
            chunks = list(zip(indices, chunks))

        if as_tuples:
            chunks = add_id(chunks)

        return chunks

    def chunk_df(self,
                 input_df: pd.DataFrame,
                 column: str = TEXT_COL,
                 drop_text: bool = True,
                 mathematical_ids: bool = False,
                 include_span: bool = False
                 ) -> pd.DataFrame:
        print("Chunking not available.")
        df = input_df.copy()
        df[CHUNK_ID_COL] = 0
        df[CHUNK_COL] = df[column]
        if include_span:
            df[CHUNK_SPAN_COL] = (0, len(df[column]))
        df[CHUNK_N_COL] = 1

        if mathematical_ids:
            df[CHUNK_ID_COL] = df[CHUNK_ID_COL].map(lambda x: x + 1)

        columns = [c for c in column_list(CHUNK_COL, column) if c in df.columns]

        if drop_text:
            columns.remove(column)

        return df[columns]



class ChunkModule:
    def __init__(self, chunk_specs, para_specs, sent_specs):
        model = chunk_specs.get("model")
        specs = self._compile_specs(chunk_specs, para_specs, sent_specs)
        self.chunker = Chunker(model, specs)

    def _compile_specs(self, chunk_specs, para_specs, sent_specs):
        specs = {}

        # resolve chunking specs
        specs["chunker"] = chunk_specs.get("chunker", "linear")
        chunk_specs = {k: v for k, v in chunk_specs.items()
                       if k not in ["chunker", "model"]}
        specs["chunk_specs"] = chunk_specs

        # resolve paragraphing specs
        for k, v in para_specs.items():
            specs[k] = v

        # resolve sentencizing specs
        for k, v in sent_specs.items():
            specs[k] = v

        return specs

    def chunk(self,
              text: str,
              as_tuples: bool = False,
              include_span: bool = False,
              **chunker_kwargs
              ) -> list:
        """Split a string containing natural language data into chunks. Returns
        a list of chunks as strings. Optionally, return a list of tuples also
        including chunk index and/or start and end indices of chunks in the
        original text.

        Args:
            text: str: Text to split into chunks
            as_tuples: bool: Return chunks as tuples with index and text
            include_span: bool: Include start and end indices of chunks
            chunker_kwargs: dict: Additional arguments for chunker

        Returns:
            list: List of chunks as strings or tuples
        """
        if include_span:
            ensure_separators = chunker_kwargs.pop("ensure_separators", False)
            chunks = self.chunker.split(text,
                                        compile=False,
                                        postprocess=False,
                                        **chunker_kwargs)
            indices = [make_indices_from_chunk(c, text) for c in chunks]
            chunks = self.chunker._compile_chunks(chunks, ensure_separators)
            chunks = self.chunker._postprocess(chunks)
            chunks = list(zip(indices, chunks))
        else:
            chunks = self.chunker.split(text,
                                        compile=True,
                                        postprocess=True,
                                        **chunker_kwargs)

        if as_tuples:
            chunks = add_id(chunks)

        return chunks

    def chunk_list(self,
                   texts: list,
                   as_tuples: bool = False,
                   include_span: bool = False,
                   **chunker_kwargs
                   ) -> list:
        """Split a list of strings containing natural language data into chunks.
        Returns a list of chunks per text as lists of strings. Optionally,
        returns lists of tuples also including chunk index and/or start and end
        indices of chunks in the original text.

        Args:
            texts: list: List of texts to split into chunks
            as_tuples: bool: Return chunks as tuples with index and text
            include_span: bool: Include start and end indices of chunks
            chunker_kwargs: dict: Additional arguments for chunker

        Returns:
            list: List of chunks as lists strings or tuples
        """
        if include_span:
            ensure_separators = chunker_kwargs.pop("ensure_separators", False)
            chunks = self.chunker.split(texts,
                                        show_progress=True,
                                        compile=False,
                                        postprocess=False,
                                        **chunker_kwargs)
            indices = [[make_indices_from_chunk(c, text) for c in chunk_list]
                       for text, chunk_list in zip(texts, chunks)]
            chunks = self.chunker._compile_chunks(chunks, ensure_separators)
            chunks = self.chunker._postprocess(chunks)
            chunks = [list(zip(index_list, chunk_list))
                      for index_list, chunk_list in zip(indices, chunks)]
        else:
            chunks = self.chunker.split(texts,
                                        show_progress=True,
                                        compile=True,
                                        postprocess=True,
                                        **chunker_kwargs)

        if as_tuples:
            chunks = [add_id(chunk_list) for chunk_list in chunks]

        return chunks

    def chunk_df(self,
                input_df: pd.DataFrame,
                column: str = TEXT_COL,
                drop_text: bool = True,
                mathematical_ids: bool = False,
                include_span: bool = False,
                **chunker_kwargs
                ) -> pd.DataFrame:
        """
        Split texts in a pandas DataFrame column into chunks. Returns a df
        with chunks as rows, including chunk text, chunk index, and optionally
        start and end indices of chunks in the original text. Original text id
        and number of chunks in text and, optionally, text string are kept in
        the output.

        Args:
            input_df: pd.DataFrame containing data data
            column: name of the column containing data data
            drop_text: whether to drop the original data column
            mathematical_ids: whether to increment chunk IDs by 1 to avoid 0
            include_span: whether to include start and end indices of chunks

        Returns:
            pd.DataFrame: DataFrame with chunks as rows
        """

        df = input_df.copy()

        df[CHUNKS_COL] = pd.Series(self.chunk_list(df[column].tolist(),
                                                   as_tuples=True,
                                                   include_span=include_span,
                                                   **chunker_kwargs
                                                   )
                                   )

        df = df.explode(CHUNKS_COL).reset_index(drop=True)

        # uunpack chunk data into separate columns
        chunks_df = pd.DataFrame(df[CHUNKS_COL].tolist())
        if include_span:
            df[[CHUNK_ID_COL, CHUNK_SPAN_COL, CHUNK_COL]] = chunks_df
        else:
            df[[CHUNK_ID_COL, CHUNK_COL]] = chunks_df

        # count number of chunks in each text
        df[CHUNK_N_COL] = df.groupby(column)[column].transform("size")

        if mathematical_ids:
            df[CHUNK_ID_COL] = df[CHUNK_ID_COL].map(lambda x: x + 1)

        # keep only desired columns for output dataframe
        columns = [c for c in column_list(CHUNK_COL, column) if c in df.columns]

        if drop_text:
            columns.remove(column)

        return df[columns]


def split_chunks(
        input_df: pd.DataFrame,
        column: str = TEXT_COL,
        drop_text: bool = True,
        mathematical_ids: bool =False,
        drop_empty: bool = True,
        strategy: str = DEFAULT_STRATEGY,
        specs: Optional[dict] = None,
        sent_specs: Optional[dict] = None
        ) \
        -> pd.DataFrame:
    """
    Split texts in a pandas DataFrame column into chunks of embeddings based on
    semantic similarity and max token length. Requires a pandas DataFrame and
    a spaCy language model as input. Chunks are inserted in a new column
    exploded to one row per chunk, keeping chunks associated with original data
    data.

    Parameters:
    input_df: pd.DataFrame containing data data
    nlp: instantiated spaCy language model used to process data
    column: name of the column containing data data
    drop_text: whether to drop the original data column
    mathematical_ids: whether to increment chunk IDs by 1 to avoid 0
    drop_empty: whether to drop empty chunks
    max_length: maximum token length of a chunk (e.g. desired length for
            specific embeddings)
    threshold: cosine similarity threshold for chunking
    metric: whether to use pairwise similarity or cumulative
            similarity for evaluating threshold
    """
    df = input_df.copy()
    df[CHUNKS_COL] = chunk_text_series(texts=df[column],
                                       mathematical_ids=mathematical_ids,
                                       drop_empty=drop_empty,
                                       as_tuples=True,
                                       strategy=strategy,
                                       specs=specs,
                                       sent_specs=sent_specs
                                       )
    df[CHUNK_N_COL] = df[CHUNKS_COL].map(len)
    df = df.explode(CHUNKS_COL)
    df[[CHUNK_ID_COL, CHUNK_COL]] = df[CHUNKS_COL].tolist()
    columns = [c for c in column_list(CHUNK_COL, column) if c in df.columns]
    df.rename(columns={TEXT_COL: column}, inplace=True)
    if drop_text:
        columns.remove(column)

    return df[columns]

def chunk_text_series(
        texts: pd.Series,
        strategy: str = DEFAULT_STRATEGY,
        specs: Optional[dict] = None,
        as_tuples: bool = False,
        mathematical_ids: bool = False,
        drop_empty: bool = True,
        sent_specs: Optional[dict] = None) \
        -> pd.Series:
    text_list = texts.tolist()
    chunks = chunk_multiple_texts(texts=text_list,
                                  mathematical_ids=mathematical_ids,
                                  drop_empty=drop_empty,
                                  as_tuples=as_tuples,
                                  strategy=strategy,
                                  specs=specs,
                                  sent_specs=sent_specs
                                  )

    return pd.Series(chunks, index=texts.index)


def chunk_multiple_texts(
        texts: list,
        strategy: str = DEFAULT_STRATEGY,
        specs: Optional[dict] = None,
        as_tuples: bool = False,
        mathematical_ids: bool = False,
        drop_empty: bool = True,
        sent_specs: Optional[dict] = None) \
        -> list:
    """
    From a list of texts, create chunks of embeddings based on semantic
    similarity. Returns a list of tuples with chunks and optionally a list of tuples with

    """
    print("finding sentences...")
    if sent_specs is not None:
        sent_specs["drop_placeholders"] = []
    sentences = [make_sentences_from_text(text, sentence_specs=sent_specs)
                 for text in tqdm(texts)]

    print("creating embeddings...")
    specs = deepcopy(specs)
    model = specs["model"] if "model" in specs else None
    embeddings = [create_embeddings(sentences=sentences, model=model)
                  for sentences in tqdm(sentences)]

    print("extracting chunks...")
    chunks = [extract_chunks(texts[i],
                             embeddings=embeddings[i],
                             strategy=strategy,
                             specs=specs,
                             drop_empty=drop_empty,
                             as_tuples=as_tuples,
                             sent_specs=sent_specs)
              for i in tqdm(range(len(texts)))]

    if mathematical_ids:
        chunks = [increment_ids(c, 1) for c in chunks]

    return chunks


def chunk_text(
        text: str,
        strategy: str = None,
        specs: Optional[dict] = None,
        drop_empty: bool = True,
        as_tuples: bool = False,
        include_span: bool = False,
        sent_specs: Optional[dict] = None)\
        -> list:
    """
    Segment a text into chunks based on semantic similarity and max token length.
    Returns a list of chunks and optionally a list tuples with id and chunk.
    """
    if sent_specs is not None:
        sent_specs["drop_placeholders"] = []
    sentences = make_sentences_from_text(text, sentence_specs=sent_specs)

    specs = deepcopy(specs)
    embeddings = create_embeddings(sentences=sentences, model=specs["model"])

    return extract_chunks(text=text,
                          embeddings=embeddings,
                          strategy=strategy,
                          specs=specs,
                          drop_empty=drop_empty,
                          as_tuples=as_tuples,
                          include_span=include_span,
                          sent_specs=sent_specs
                          )

def extract_chunks(
        text: str,
        embeddings: list,
        strategy: str = None,
        specs: Optional[dict] = None,
        drop_empty: bool = True,
        as_tuples: bool = False,
        include_span: bool = False,
        sent_specs: Optional[dict] = None)\
        -> list:
    """
    Segment a spacy doc into chunks based on semantic similarity and max token
    length. Returns a list of chunks and optionally a list tuples with id and
    chunk.
    """
    if sent_specs is not None:
        sent_specs["drop_placeholders"] = []
    sentences = make_sentences_from_text(text, sentence_specs=sent_specs)

    if embeddings is None:
        specs = deepcopy(specs)
        embeddings = create_embeddings(sentences=sentences, model=specs["model"])

    chunks = create_chunks(sentences=sentences,
                           embeddings=embeddings,
                           strategy=strategy,
                           specs=specs,
                           )

    if drop_empty:
        chunks = [c for c in chunks if len(c) > 0]

    if include_span:
        indices = [make_indices_from_chunk(chunk, text) for chunk in chunks]
        chunk_texts = [make_text_from_chunk(chunk) for chunk in chunks]
        chunks = list(zip(indices, chunk_texts))
    else:
        chunks = [make_text_from_chunk(chunk) for chunk in chunks]

    if as_tuples:
        chunks = add_id(chunks)

    return chunks

# chunking function
def create_chunks(
        sentences: list,
        embeddings: list,
        strategy: Optional[str] = DEFAULT_STRATEGY,
        specs = Optional[dict],
        ) -> list:
    """
    Segment a list of spaCy embeddings (from a doc object) into chunks based
    on semantic similarity and maximal token length. based on. Returns a list
    of lists containing chunks of embeddings.

    The current algorithm has one major limitation: Chunk length is always one
    sentence minimum. So when the text contains embeddings that exceed maximal
    length specified, the result will contain some longer chunks.

    Parameters:
    embeddings: a list of spaCy embeddings
    max_length: maximum token length of a chunk (e.g. desired length for
            specific embeddings)
    threshold: cosine similarity threshold for chunking
    metric: whether to use pairwise similarity or cumulative
            similarity for evaluating threshold
    """

    if strategy == "linear":
        if "length_metric" not in specs:
            specs["length_metric"] = TokenCounter(specs["model"])

        params = inspect.signature(linear_chunking).parameters
        filtered_specs = {k: v for k, v in specs.items() if k in params}

        return linear_chunking(
            sentences=sentences,
            embeddings=embeddings,
            **filtered_specs
        )
    elif strategy == "graph":
        specs = specs or {
            "K": DEFAULT_SCOPE,
            "resolution": DEFAULT_RESOLUTION
        }
        params = inspect.signature(graph_chunking).parameters
        filtered_specs = {k: v for k, v in specs.items() if k in params}

        return graph_chunking(
            sentences=sentences,
            embeddings=embeddings,
            **filtered_specs
            )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

# Functions to reconstruct data from chunks

def make_text_from_chunk(
        chunk: list
        ) -> str:
    """Reconstruct text data from a chunk."""
    texts = [sent.strip() for sent in chunk]
    return" ".join([text for text in texts])

def make_indices_from_chunk(chunk, text):
    """Reconstruct indices of chunk span in original text."""
    start_sent = chunk[0]
    end_sent = chunk[-1]
    start_idx = find_substring_indices(text, [start_sent])[0][0]
    end_idx = find_substring_indices(text, [end_sent])[0][1]
    return start_idx, end_idx


# Example usage
if __name__ == "__main__":

    # Load spaCy model
    nlp = spacy.load("de_core_news_sm")

    text = """koopstadt – Ein Kooperationsprojekt zur Stadtentwicklung in Bremen, Leipzig, Nürnberg

    "Mit diesem Kooperationsprojekt haben sich drei ähnliche und doch unterschiedliche Städte – Bremen, Leipzig, Nürnberg – zusammengeschlossen, um neue Antworten auf die drängenden Fragen unserer Städte zu entwickeln. Jede für sich und doch miteinander. Voneinander lernend und Beispiel gebend für andere Städte." (Martin zur Nedden, Bürgermeister und Beigeordneter der Stadt Leipzig für Stadtentwicklung und Bau)

Mit- und voneinander lernen – das haben sich die drei Städte Bremen, Leipzig und Nürnberg auf die Fahnen geschrieben und gemeinsam das Pilotprojekt "koopstadt" entwickelt. Von 2007 bis 2015 wollen die Städte einen intensiven Austausch zu Themen der Stadtentwicklung pflegen und innovative Lösungen für die zentralen Herausforderungen der Stadtentwicklung erproben. 

Ziel des Projekts

Mit- und voneinander lernen – das haben sich die drei Städte Bremen, Leipzig und Nürnberg auf die Fahnen geschrieben und gemeinsam das Pilotprojekt "koopstadt" entwickelt. Von 2007 bis 2015 wollen die Städte einen intensiven Austausch zu Themen der Stadtentwicklung pflegen und innovative Lösungen für die zentralen Herausforderungen der Stadtentwicklung erproben. Sie verstehen sich dabei gleichermaßen als Anschauungsobjekt, Werkstatt und Impulsgeber für einen angeregten Dialog über Stadtentwicklung, in den auch Öffentlichkeit einbezogen wird.

Aufgabe

Die Städte haben drei Themenfelder identifiziert, die - als zentrale Herausforderungen der Stadtentwicklung - den inhaltlichen Rahmen des Erfahrungsaustauschs markieren: 

- Ökonomische Innovation und kreative Milieus,
- Urbane Lebensqualität,
- Regionale Kooperation.

Diesen Themenfeldern haben die Städte mittlerweile ca. 30 Einzelprojekte zugeordnet und darin wiederum zu Projektfamilien gebündelt. Diese Einzelprojekte, ihre Qualifizierung und Realisierung bilden den Ausgangspunkt für den gemeinsamen Erfahrungsaustausch.

Umsetzung

Lernprozesse in städteübergreifenden Strukturen, wie koopstadt sie anstrebt, brauchen Zeit. Das Pilotprojekt ist daher langfristig angelegt und gliedert sich in vier Phasen:

- In der ersten Phase (2008) wurde der Grundstein für eine erfolgreiche Kooperation gelegt. Zum Ende dieser Phase wurde eine Konzeptstudie vorgelegt, in der die Inhalte der Zusammenarbeit und beispielgebende Projekte benannt sowie Vorschläge zu Verfahren und Organisation der gemeinsamen Arbeit gemacht werden.
- In der zweiten Phase (Frühjahr 2009) wurde koopstadt durch Beschlüsse in den entsprechenden Gremien im politischen Raum der drei Städte gesichert. Dies war zugleich das offizielle Signal, um mit der Umsetzung zu beginnen.
- Seit Sommer 2009 läuft die dritte Phase des Projekts. Bis 2012 wird es nun darum gehen, die Einzelprojekte weiterzuqualifizieren und ihren Start zu begleiten.
- In der vierten Phase (2012-2015) werden die Realisierung der Projekte und ihre Verstetigung sowie die Evaluation des Gesamtprozesses im Vordergrund stehen. 

Über den engeren Kreis der Projektbeteiligten hinaus möchte koopstadt die Ergebnisse der gemeinsamen Arbeit einer breiten Öffentlichkeit vermitteln und damit die Debatte zu stadtentwicklungspolitischen Themen beleben. Schon in der Konzeptphase konnten sich Experten und Bürger in zahlreichen Veranstaltungen, Konferenzen, Ausstellungen und Stadtspaziergängen über das Projekt informieren und mit den Beteiligten diskutieren. Dieser Dialog wird nun in der Qualifizierungsphase fortgesetzt und mit mehreren, unterschiedlichen Formaten in der Abschlussphase intensiviert. Bereits jetzt kann man sich einen Überblick über die umfangreiche Projektliste und aktuelle Aktivitäten über die Homepage des Projekts (siehe Weblink) verschaffen.
"""

    print("Loading transformer model...")
    transformer_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("Done")

    model = transformer_model

    linear_specs = {
        "max_length": 200,
        "threshold": 0.05,
        "similarity_metric": "cumulative",
        "model": model
    }


    graph_specs = {
        "K": 5,
        "resolution": 2,
        "model": model
    }


    chunks = chunk_text(text, nlp=nlp, strategy="simple", specs=simple_specs)
    print("Simple chunking:")
    for i, chunk in enumerate(chunks):
        approx_length = len(chunk.split(" "))
        print(f"{i + 1} | ({approx_length}): {chunk}")
        print("______________________")

    chunks = chunk_text(text, nlp=nlp, strategy="graph", specs=graph_specs)
    print("graph chunking")
    for i, chunk in enumerate(chunks):
        approx_length = len(chunk.split(" "))
        print(f"{i + 1} | ({approx_length}): {chunk}")
        print("______________________")
