import re

# Define regex pattern for splitting data into paragraphs
PARAGRAPH_PATTERN = re.compile(r'''
    (?<=[.?!\])"“:])  # Positive lookbehind for sentence-ending punctuation
    \s*\n+            # Optional whitespace followed by one or more newlines
    |
    \n{2,}            # Always split after two or more newlines
    ''', re.VERBOSE)

# Simple newline-based pattern
PARAGRAPH_PATTERN_SIMPLE = re.compile(r'\n')

# Pattern to find enumeration paragraphs in German text.
# Excludes lines that start with dates like "12. September", "12.09.", or "12.3."
ENUM_PATTERN_NO_DATE_DE = re.compile(r'''
    ^                      # Start of line
    (?!                    # Negative lookahead to exclude date patterns
        \d{1,2}\.          # 1-2 digits followed by period (day)
        \s*                # Optional whitespace
        (?:                # Non-capturing group for month formats:
            # Month names
            Januar|Februar|März|
            April|Mai|Juni|Juli|
            August|September|
            Oktober|November|Dezember
            |              # OR
            # Numeric date format (like 12.09. or 12.9.)
            \d{1,2}\.?\d{0,2}
        )
    )
    \d+\.                  # Digits followed by period (the enumeration)
    \s+                    # One or more whitespace characters after period
''', re.VERBOSE)