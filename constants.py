# Constants to identify certain types of special characters and strings
PLACEHOLDERS = ["[...]", "[…]"]

# Default language codes and arguments
# (expandable dict for packages and functions requesting language
# specifications)
DEFAULT_LANGUAGE = {"ISO 639-1": "de"}

# Constants for token attribute extraction
# (not implemented yet, preparations for spaCy)
MANDATORY_ATTRS = ["i", "text"]  # these attributes are always extracted
# Optional attributes can be extracted if specified in function call
OPTIONAL_ATTRS = ["lemma_", "is_stop", "pos_", "tag_", "dep_", "is_alpha",
                  "is_digit", "is_punct", "like_num", "is_space", "is_currency",
                  "is_quote", "is_bracket", "like_num", "like_url",
                  "like_email", "is_oov", "is_sent_start", "idx", "ent_type_",
                  "ent_iob_"]
