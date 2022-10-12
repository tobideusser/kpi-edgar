import re
import string
from typing import List

from babel.numbers import parse_decimal, NumberFormatError
from tqdm import tqdm

from edgar.data_classes import Corpus, Table, Sentence

# currency_matching_pre = re.compile('(€|EUR|Euro)($|\.|\:|\,)')
# currency_matching_post = re.compile('(^[^\(]*)(€|EUR|Euro)($|\.|\:|\,|\))')
GERMAN_CURRENCY_REGEX = re.compile(r"(€|EUR|Euro|Eur)")
ENGLISH_US_CURRENCY_REGEX = re.compile(r"(?i)(\$|USD|U\.S\. dollar|US dollar)")
GERMAN_DATE_REGEX = re.compile(
    r"(0?[1-9]|[12][0-9]|3[01]) ?([-/\., ])? ?((januar|Januar|februar|Februar|märz|März|april|April|mai|Mai|juni|Juni|"
    r"juli|Juli|august|August|september|September|oktober|Oktober|november|November|dezember|Dezember|jan|Jan|feb|Feb|"
    r"mrz|Mrz|mär|Mär|apr|Apr|mai|Mai|jun|Jun|jul|Jul|aug|Aug|sep|Sep|okt|Okt|nov|Nov|dez|Dez)|"
    r"[XIV]{1,4}|0?[1-9]|1[0-2]) ?[-/\., ]? ?((19|20)[0-9]{2})"
)
ENGLISH_US_DATE_REGEX = re.compile(
    r"(?i)(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|"
    r"november|nov|december|dec|0?[1-9]|1[0-2]) ?([-/\., ])? (0?[1-9]|[12][0-9]|3[01]) ?[-/\., ]? ?((19|20)[0-9]{2})"
)

GERMAN_CURRENCY_UNIT_MAPPING = {
    ("teur", "t", "te", "tsd", "tausend", "k", "keur", "tsdeur"): {"unit": "Tsd. EUR", "multiplier": 1e3},
    ("mio", "millionen", "million", "m", "mioeur"): {"unit": "Mio. EUR", "multiplier": 1e6},
    ("mrd", "milliarden", "milliarde", "mrdeur"): {"unit": "Mrd. EUR", "multiplier": 1e9},
}

ENGLISH_US_CURRENCY_UNIT_MAPPING = {
    ("kusd", "KUSD", "thousand", "Thousand", "thousands", "Thousands", "k", "K", "k$", "K$"): {
        "unit": "Thousand USD",
        "multiplier": 1e3,
    },
    ("musd", "MUSD", "million", "Million", "millions", "Millions", "m", "M", "m$", "M$"): {
        "unit": "Million USD",
        "multiplier": 1e6,
    },
    ("busd", "BUSD", "billion", "Billion", "billions", "Billions", "b", "B", "b$", "B$"): {
        "unit": "Billion USD",
        "multiplier": 1e9,
    },
}


def clean(token: str):
    punctuation_chars = string.punctuation + "–"
    try:
        while token[0] in punctuation_chars or token[0] == " ":
            token = token[1:]
        while token[-1] in punctuation_chars or token[-1] == " ":
            token = token[:-1]
    except IndexError:
        pass
    return token


def get_token_context(tokens: List[str], pos: int, left: int, right: int) -> List[str]:
    # get left token context (exclude '.' tokens)
    context_left = []
    left_shift = 1
    while len(context_left) < left and pos - left_shift >= 0:
        context_token = tokens[max(pos - left_shift, 0)]
        if context_token not in ".":
            context_left.append(context_token)
        left_shift += 1

    # get right token context (exclude '.' tokens)
    context_right = []
    shift_right = 1
    while len(context_right) < right and pos + shift_right + 1 < len(tokens):
        context_token = tokens[min(pos + shift_right, len(tokens) - 1)]
        if context_token not in ".":
            context_right.append(context_token)
        shift_right += 1
    # combine left_context, token and right context
    return context_left[::-1] + [tokens[pos]] + context_right


def convert_to_float(token: str, language="de") -> float:
    return float(parse_decimal(token, locale=f"{language}_{language.upper()}"))


def is_year(token: str):
    years = [str(year) for year in range(1970, 2030)]
    if token in years:
        return True
    else:
        return False


def tag_numeric_tokens(sentence: Sentence, language: str = "en"):
    date_regex = GERMAN_DATE_REGEX if language == "de" else ENGLISH_US_DATE_REGEX
    for token in sentence:
        try:
            if date_regex.search(token.value) or is_year(token.value):
                token.is_numeric = False
            else:
                num_token = convert_to_float(token.value)
                token.value_numeric = num_token
                token.is_numeric = True
                token.value_masked = "<NUM>"
        except NumberFormatError:
            token.is_numeric = False


def tag_currency_tokens(sentence: Sentence, language: str = "en"):
    # Remove punctuation in the beginning and end of the token
    token_values = [token.value for token in sentence]

    # Initialize currency symbol position (left or right)
    currency_pos = None

    currency_regex = GERMAN_CURRENCY_REGEX if language == "de" else ENGLISH_US_CURRENCY_REGEX
    currency_unit_mapping = GERMAN_CURRENCY_UNIT_MAPPING if language == "de" else ENGLISH_US_CURRENCY_UNIT_MAPPING
    curreny_unit = "EUR" if language == "de" else "USD"

    for token in sentence:

        if not token.is_numeric:
            token.is_currency = False
            continue

        context_left = get_token_context(token_values, pos=int(token.id_), left=2, right=0)
        context_right = get_token_context(token_values, pos=int(token.id_), left=0, right=2)

        match = False
        if not currency_pos:
            for context_token in context_left:
                if currency_regex.search(context_token):
                    currency_pos = "left"
                    break
            if not currency_pos:
                for context_token in context_right:
                    if currency_regex.search(context_token):
                        currency_pos = "right"
                        break

        if currency_pos == "left":
            for context_token in context_left:
                if currency_regex.search(context_token):
                    match = True
                    break

        if currency_pos == "right":
            for context_token in context_right:
                if currency_regex.search(context_token):
                    match = True
                    break

        if match:
            for context_token in get_token_context(token_values, pos=int(token.id_), left=2, right=2):
                for queries, unit in currency_unit_mapping.items():
                    if context_token.lower() in queries:
                        token.unit = unit["unit"]
                        token.multiplier = unit["multiplier"]
                        token.value_masked = "<NUM_CY>"
                        token.is_currency = True
                        break
            if not token.unit:
                token.unit = curreny_unit
                token.multiplier = 1.0
                token.value_masked = "<NUM_CY>"
                token.is_currency = True


def tag_numeric_cells(table: Table, language: str = "en"):
    date_regex = GERMAN_DATE_REGEX if language == "de" else ENGLISH_US_DATE_REGEX
    for cell in table:
        cell_cleaned = clean(cell.value)
        try:
            if cell_cleaned == cell.value or f"-{cell_cleaned}" == cell.value:
                num_cell = float(cell_cleaned)
            else:
                num_cell = convert_to_float(cell_cleaned)

            if cell.value == "nan":
                cell.is_numeric = False
            elif date_regex.search(cell_cleaned) or is_year(cell_cleaned):
                cell.is_numeric = False
            else:
                cell.value_numeric = num_cell
                cell.is_numeric = True
        except ValueError:
            cell.is_numeric = False


def tag_currency_cells(table: Table, language: str = "en"):
    """
    If the table is stacked:
    Activa
    ---
    Passiva

    there might be an Anhang-reference column in the activa that
    is not directly in line with the anhang-reference column in the passiva.
    We therefore split the table into segments, starting at each row containing
    and Anhang-reference.
    If a column contains an Anhang-reference, we only tag the numbers in it
    as not currency between to header-rows.
    """

    currency_regex = GERMAN_CURRENCY_REGEX if language == "de" else ENGLISH_US_CURRENCY_REGEX
    currency_unit_mapping = GERMAN_CURRENCY_UNIT_MAPPING if language == "de" else ENGLISH_US_CURRENCY_UNIT_MAPPING
    curreny_unit = "EUR" if language == "de" else "USD"

    # rows that contain the word 'anhang'
    anhang_row_indices = [
        i for i, row in enumerate(table.rows) if any(["anhang" in cell.value.lower() for cell in row])
    ]

    if anhang_row_indices:
        # loop over all parts between anhang references
        for i in range(len(anhang_row_indices)):
            start = anhang_row_indices[i]
            try:
                end = anhang_row_indices[i + 1]
            except IndexError:
                end = len(table.rows)

            # if the word anhang is in a column, the numbers in the column
            # are anhang references, not currency.
            for col_ in table.cols:
                col = col_[start:end]
                is_anhang_col = any([True if "anhang" in cell.value.lower() else False for cell in col])

                for cell in col:
                    cell.is_currency = cell.is_numeric and not is_anhang_col

        # tag the cells before the first anhang reference
        # cell is currency <=> cell is numeric
        for cell in table.cells:
            if cell.is_currency is None:
                cell.is_currency = cell.is_numeric
    else:
        # no 'anhang' refernce -> cell is currency <=> cell is numeric
        for cell in table.cells:
            cell.is_currency = cell.is_numeric

    # Detect unit for currency cells
    currency_unit_cells = {}
    for cell in table.cells:
        if currency_regex.search(cell.value):
            col_unit = curreny_unit
            col_multiplier = 1.0
            for token in cell.words:
                token_cleaned = clean(token.value)
                if currency_regex.search(token_cleaned):
                    token_cleaned = token_cleaned.replace(".1", "")
                for queries, unit in currency_unit_mapping.items():
                    if token_cleaned.lower() in queries:
                        col_unit = unit["unit"]
                        col_multiplier = unit["multiplier"]
                        break
            if cell.col not in currency_unit_cells.keys():
                currency_unit_cells[cell.col] = []
            currency_unit_cells[cell.col].append({"row": cell.row, "unit": col_unit, "multiplier": col_multiplier})

    num_unique_units = len(set([row["unit"] for col in currency_unit_cells.values() for row in col]))

    #  Write unit in cell object
    if len(currency_unit_cells) == 1 or num_unique_units == 1:
        for cell in table.cells:
            if cell.is_currency:
                cell.unit = currency_unit_cells[list(currency_unit_cells)[0]][0]["unit"]
                cell.multiplier = currency_unit_cells[list(currency_unit_cells)[0]][0]["multiplier"]

    else:
        for col_id, col in currency_unit_cells.items():
            for cell in table.cells:
                # if len(col) == 1:
                if col_id == cell.col and cell.is_currency:
                    cell.unit = col[0]["unit"]
                    cell.multiplier = col[0]["multiplier"]
                elif cell.is_currency:
                    cell.unit = curreny_unit
                    cell.multiplier = 1.0
                else:
                    pass


def filter_corpus_by_ccy_sentences(corpus: Corpus) -> Corpus:
    for document in tqdm(corpus):
        segments_filtered = []
        for segment in document:
            if segment.tag == "text":
                sentences_filtered = []
                for sentence in segment:
                    if any([token.is_currency for token in sentence]):
                        sentences_filtered.append(sentence)
                segment.sentences = sentences_filtered
                if segment.sentences:
                    segments_filtered.append(segment)
            elif segment.tag == "table":
                segments_filtered.append(segment)
        document.segments = segments_filtered
    return corpus
