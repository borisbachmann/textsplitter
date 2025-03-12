from ....dataframes import columns

# Documentation of token attributes https://spacy.io/api/token

# Mapping of spaCy token attributes to column names, mandatory
# these attributes are required to be extracted
MANDATORY_ATTRS = {
    "i": columns.id_pattern(columns.TOKEN_COL),
    "text": columns.TOKEN_COL
}

# Mapping of spaCy token attributes to column names, optional
OPTIONAL_ATTRS = {
    "lemma_": "lemma",
    "is_stop": "is_stop",
    "pos_": "pos",
    "tag_": "tag",
    "dep_": "dep",
    "is_alpha": "is_alpha",
    "is_digit": "is_digit",
    "is_lower": "is_lower",
    "is_upper": "is_upper",
    "is_title": "is_title",
    "is_punct": "is_punct",
    "is_left_punct": "is_left_punct",
    "is_right_punct": "is_right_punct",
    "is_bracket": "is_bracket",
    "is_space": "is_space",
    "is_currency": "is_currency",
    "is_quote": "is_quote",
    "like_num": "like_num",
    "like_url": "like_url",
    "like_email": "like_email",
    "is_sent_start": "is_sent_start",
    "is_sent_end": "is_sent_end",
    "ent_type_": "ent_type",
    "ent_iob_": "ent_iob",
    # 'morph': 'morph',
    "sentiment": "sentiment",
    "idx": "start_idx",
}

# new attributes
SENT_I = "sent_i"  # sentence index based upon spaCy sentence boundaries
END_I = "end_i"    # derived based upon start index and token length

# mapping of derived attributes to column names
DERIVED_ATTRS = {
    "sent_i": columns.id_pattern(columns.SENT_COL),
    "end_i": "end_idx"
}
