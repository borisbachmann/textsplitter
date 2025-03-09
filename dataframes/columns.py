# Constants and functions for columns in Dataframe operations
# Default columns are optimized for use with my corpus_builder code

# BASIC BUILDING BLOCKS

# textual data units and corresponding column names
TEXT_COL = "text"       # column with initial data to be chunked
SENT_COL = "sentence"   # column for individual sentences
PARA_COL = "paragraph"  # column for individual paragraphs
CHUNK_COL = "chunk"     # column for individual chunk
# not implemented yet
TOKEN = "token"         # base name for tokens, not used as columns name
TOKEN_COL = "text"      # column for token text, used as base column name

# basic suffixes for token column types
MULTI_SUFFIX = "s"    # suffix for columns with multiple segments
ID_SUFFIX = "ID"      # suffix for ID columns
N_SUFFIX = "n"        # suffix for count columns
SPAN_SUFFIX = "span"  # suffix for span columns

# Columns containing multiple entries
def multi_pattern(base_col):
    """Make column name for column with multiple entries based on base column
    name."""
    return f"{base_col}{MULTI_SUFFIX}"

# direct access for convenience
SENTS_COL = multi_pattern(SENT_COL)
PARAS_COL = multi_pattern(PARA_COL)
CHUNKS_COL = multi_pattern(CHUNK_COL)

# ID Columns
def id_pattern(base_col):
    """Make column name for column with ID based on base column name."""
    return f"{base_col}_{ID_SUFFIX}"

# direct access for convenience
TEXT_ID = id_pattern(TEXT_COL)
SENT_ID_COL = id_pattern(SENT_COL)
PARA_ID_COL = id_pattern(PARA_COL)
CHUNK_ID_COL = id_pattern(CHUNK_COL)
TOKEN_ID = id_pattern(TOKEN)
TOKEN_SENT_ID = SENT_ID_COL

# segment count columns
def n_pattern(base_col):
    """Make column name for column with count of entries based on base column
    name."""
    return f"{multi_pattern(base_col)}_{N_SUFFIX}"

# direct access for convenience
SENT_N_COL = n_pattern(SENT_COL)
PARA_N_COL = n_pattern(PARA_COL)
CHUNK_N_COL = n_pattern(CHUNK_COL)

# span columns
def span_pattern(base_col):
    """Make column name for column with span based on base column name."""
    return f"{base_col}_{SPAN_SUFFIX}"

# direct access for convenience
SENT_SPAN_COL = span_pattern(SENT_COL)
PARA_SPAN_COL = span_pattern(PARA_COL)
CHUNK_SPAN_COL = span_pattern(CHUNK_COL)
