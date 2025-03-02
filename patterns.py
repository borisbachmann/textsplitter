import re

# Define regex pattern for splitting data into paragraphs
PARAGRAPH_PATTERN = re.compile(r'''
    (?<=[.?!\])"“:])  # Positive lookbehind for sentence-ending punctuation
    \s*\n+            # Optional whitespace followed by one or more newlines
    |
    \n{2,}            # Always split after two or more newlines
    ''', re.VERBOSE)
PARAGRAPH_PATTERN_SIMPLE = r'\n'

# Excludes lines that start with dates like "12. September", "12.09.", or "12.3."
ENUM_PATTERN_NO_DATE = re.compile(r'^(?!\d{1,2}\.\s*(?:Januar|Februar|März'
                                  r'|April|Mai|Juni|Juli|August|September'
                                  r'|Oktober|November|Dezember'
                                  r'|\d{1,2}\.?\d{0,2})).*\d+\.\s*')