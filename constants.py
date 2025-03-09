# Constants for columns in Dataframe operations
# Default columns are optimized for use with the corpus_builder module

# basic suffixes
ID_SUFFIX = "ID"      # suffix for ID columns
N_SUFFIX = "n"        # suffix for count columns
SPAN_SUFFIX = "span"  # suffix for span columns

# textual data units and corresponding column names
TEXT_COL = "text"       # column with initial data to be chunked
SENT_COL = "sentence"   # column for individual sentences
PARA_COL = "paragraph"  # column for individual paragraphs
CHUNK_COL = "chunk"     # column for individual chunk
TOKEN = "token"         # base name for tokens, not used as columns name
TOKEN_COL = "text"      # column for token text, used as base column name

# names for list columns containing multiple entries
def multi_pattern(base_col):
    """Make column name for column with multiple entries based on base column
    name."""
    return f"{base_col}s"

SENTS_COL = multi_pattern(SENT_COL)
PARAS_COL = multi_pattern(PARA_COL)
CHUNKS_COL = multi_pattern(CHUNK_COL)

# ID Columns
def id_pattern(base_col):
    """Make column name for column with ID based on base column name."""
    return f"{base_col}_{ID_SUFFIX}"

TEXT_ID = id_pattern(TEXT_COL)
SENT_ID_COL = id_pattern(SENT_COL)
PARA_ID_COL = id_pattern(PARA_COL)
CHUNK_ID_COL = id_pattern(CHUNK_COL)
TOKEN_ID = id_pattern(TOKEN)
TOKEN_SENT_ID = SENT_ID_COL

# count columns
def n_pattern(base_col):
    """Make column name for column with count of entries based on base column
    name."""
    return f"{multi_pattern(base_col)}_{N_SUFFIX}"

SENT_N_COL = n_pattern(SENT_COL)
PARA_N_COL = n_pattern(PARA_COL)
CHUNK_N_COL = n_pattern(CHUNK_COL)

# span columns
def span_pattern(base_col):
    """Make column name for column with span based on base column name."""
    return f"{base_col}_{SPAN_SUFFIX}"

SENT_SPAN_COL = span_pattern(SENT_COL)
PARA_SPAN_COL = span_pattern(PARA_COL)
CHUNK_SPAN_COL = span_pattern(CHUNK_COL)

# Constants for token attribute extraction
MANDATORY_ATTRS = ["i", "text"]  # these attributes are always extracted
# Optional attributes can be extracted if specified in function call
OPTIONAL_ATTRS = ["lemma_", "is_stop", "pos_", "tag_", "dep_", "is_alpha",
                  "is_digit", "is_punct", "like_num", "is_space", "is_currency",
                  "is_quote", "is_bracket", "like_num", "like_url",
                  "like_email", "is_oov", "is_sent_start", "idx", "ent_type_",
                  "ent_iob_"]

# Constants to identify certain types of special characters and strings
PLACEHOLDERS = ["[...]", "[…]"]

# Default language codes and arguments
DEFAULT_LANGUAGE = {"ISO 639-1": "de"}
