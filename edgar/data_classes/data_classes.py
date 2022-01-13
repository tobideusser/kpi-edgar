from __future__ import annotations

import ast
import html
import json
import unicodedata
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import List, Dict, Optional, Tuple

import lxml
import pandas as pd
from lxml import html as lh
from lxml.html.clean import Cleaner


def create_dataframe_from_json(df: str) -> pd.DataFrame:
    """ Decode dataframe from serialized json string.
    Safely evaluates multi-index keys (tuple strings) as tuples before converting dict to dataframe.
    """

    df = json.loads(df)

    parsed_df = {}
    for k, v in df.items():
        try:
            parsed_key = ast.literal_eval(k)
            if not isinstance(parsed_key, tuple):
                parsed_key = k
        except (ValueError, SyntaxError):
            parsed_key = k
        parsed_df[parsed_key] = v

    return pd.DataFrame(parsed_df)


@dataclass
class Word:
    id_: int
    value: str
    prefix: Optional[str] = None
    suffix: Optional[str] = None
    info: Optional[Dict] = None

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

    def to_dict(self) -> Dict:
        d = self.__dict__
        return d


@dataclass
class Cell:
    id_: int
    value: str
    words: Optional[List[Word]] = None
    row: Optional[int] = None
    col: Optional[int] = None
    is_row_headline: Optional[bool] = None
    is_col_headline: Optional[bool] = None
    _row_headline: Optional[List[Cell]] = None
    _col_headline: Optional[List[Cell]] = None

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx) -> Word:
        return self.words[idx]

    def get_row_headline(self, cells: List[Cell]) -> Optional[List[Cell]]:
        if self._row_headline is None:
            self._row_headline = [cell for cell in cells
                                  if cell.row == self.row and cell.is_row_headline and cell.col < self.col]
        return self._row_headline

    def get_col_headline(self, cells: List[Cell]) -> Optional[List[Cell]]:
        if self._col_headline is None:
            self._col_headline = [cell for cell in cells
                                  if cell.col == self.col and cell.is_col_headline and cell.row < self.row]
        return self._col_headline

    @classmethod
    def from_dict(cls, d: Dict):
        words = d.get('words')
        d['words'] = [Word.from_dict({**{'id_': id_}, **word}) for id_, word in enumerate(words)] if words else []
        return cls(**d)

    def to_dict(self) -> Dict:
        d = self.__dict__
        d['words'] = [word.to_dict() for word in d['words']]
        return d


@dataclass
class Sentence:
    pass

    @classmethod
    def from_dict(cls, d: Dict) -> Segment:
        raise NotImplementedError

    def to_dict(self) -> Dict:
        raise NotImplementedError


@dataclass
class EdgarEntity:
    id_: str
    name: str
    start: int
    end: int
    value: str
    context_ref: Optional[str] = None
    continued_at: Optional[str] = None
    escape: Optional[bool] = None
    gaap_id: Optional[str] = None
    gaap_master_id: Optional[str] = None
    label_id: Optional[str] = None
    location_id: Optional[str] = None
    us_gaap_id: Optional[str] = None
    us_gaap_value: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)

    def to_dict(self) -> Dict:
        d = self.__dict__
        return d


@dataclass
class Segment:
    id_: int
    value: str
    tag: Optional[str] = None

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict) -> Segment:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> Dict:
        raise NotImplementedError


@dataclass
class Paragraph(Segment):
    sentences: List[Sentence] = field(default_factory=list)
    textblock_entity: Optional[EdgarEntity] = None
    edgar_entities: List[EdgarEntity] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict) -> Segment:
        sentences = d.get("sentences")
        d["sentences"] = [Sentence.from_dict(sentence) for sentence in sentences] if sentences else []
        d["textblock_entity"] = EdgarEntity.from_dict(d["textblock_entity"]) if d["textblock_entity"] else None
        edgar_entities = d.get(d["edgar_entities"])
        d["edgar_entities"] = [EdgarEntity.from_dict(entity) for entity in edgar_entities] if d["textblock_entity"] \
            else None
        return cls(**d)

    @abstractmethod
    def to_dict(self) -> Dict:
        d = self.__dict__
        d["sentences"] = [sentence.to_dict() for sentence in self.sentences] if self.sentences else None
        d["textblock_entity"] = self.textblock_entity.to_dict() if self.textblock_entity else None
        d["edgar_entities"] = [entity.to_dict() for entity in self.edgar_entities] if self.edgar_entities else None
        return d


@dataclass
class Table(Segment):

    cells: List[Cell] = field(default_factory=list)
    unique_id: Optional[str] = None
    _df: Optional[pd.DataFrame] = None
    _rows: Optional[List[List[Cell]]] = None
    _cols: Optional[List[List[Cell]]] = None

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx) -> Cell:
        return self.cells[idx]

    @property
    def df(self):
        if not self._df:
            self._df = pd.DataFrame([[cell.value for cell in row] for row in self.rows])
        return self._df

    def show(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)
        print(self.df)

    @property
    def rows(self):
        if self._rows is None:
            _rows = defaultdict(list)
            for cell in self.cells:
                _rows[cell.row].append(cell)
            indices = sorted(_rows.keys())
            self._rows = [[cell for cell in _rows[i]] for i in indices]
        return self._rows

    @property
    def cols(self):
        if self._cols is None:
            _cols = defaultdict(list)
            for cell in self.cells:
                _cols[cell.col].append(cell)
            indices = sorted(_cols.keys())
            self._cols = [[cell for cell in _cols[i]] for i in indices]
        return self._cols

    @classmethod
    def from_dict(cls, d: Dict):
        cells = d.get('cells')
        d['cells'] = [Cell.from_dict(cell) for cell in cells] if cells else []
        d['df'] = create_dataframe_from_json(d['df'])
        return cls(**d)

    def to_dict(self) -> Dict:
        d = self.__dict__
        d['cells'] = [cell.to_dict() for cell in d['cells']]
        d['df'] = d['df'].to_json()
        return d

    @staticmethod
    def extract_text_recursively(root: lxml.html.HtmlElement, out: List, xbrl_tags: List):

        for node in root:

            text = ' '.join(node.text.split()) if node.text else None
            if text:
            # if node.text:
                # prefix = ' '  # node.text[0]
                # try:
                #     first_stripped_char = node.text.lstrip()[0]
                # except IndexError:
                #     first_stripped_char = ''
                #
                # if prefix == first_stripped_char:
                #     prefix = ''
                #
                # suffix = ' '  # node.text[-1]
                # try:
                #     last_stripped_char = node.text.rstrip()[-1]
                # except IndexError:
                #     last_stripped_char = ''
                #
                # if suffix == last_stripped_char:
                #     suffix = ''

                # text = ' '.join(node.text.split())
                out.append({'value': text,
                            # 'prefix': prefix,
                            # 'suffix': suffix,
                            'info': []})

            out = Table.extract_text_recursively(node, out, xbrl_tags)

            if node.tag in xbrl_tags and len(out) > 0:
                info = dict(node.items())
                out[-1]['info'].append(info)

            tail = ' '.join(node.tail.split()) if node.tail else None
            if tail:
            # if node.tail:
                # prefix = ' '  # node.tail[0]
                # try:
                #     first_stripped_char = node.tail.lstrip()[0]
                # except IndexError:
                #     first_stripped_char = ''
                #
                # if prefix == first_stripped_char:
                #     prefix = ''
                #
                # suffix = ' '  # node.tail[-1]
                # try:
                #     last_stripped_char = node.tail.rstrip()[-1]
                # except IndexError:
                #     last_stripped_char = ''
                #
                # if suffix == last_stripped_char:
                #     suffix = ''

                # tail = ' '.join(node.tail.split())
                out.append({'value': tail,
                            # 'prefix': prefix,
                            # 'suffix': suffix,
                            'info': []})

        return out

    @staticmethod
    def _get_num_columns(rows: List[lxml.html.HtmlElement]) -> int:
        rowspans = []  # track pending rowspans
        colcount = 0  # first scan, see how many columns we need
        for row_id, row in enumerate(rows):
            cells = row.xpath('./td | ./th')
            # count columns (including spanned).
            # add active row_spans from preceding rows
            # we *ignore* the colspan value on the last cell, to prevent
            # creating 'phantom' columns with no actual cells, only extended
            # col_spans. This is achieved by hardcoding the last cell width as 1.
            # a colspan of 0 means “fill until the end” but can really only apply
            # to the last cell; ignore it elsewhere.
            colcount = max(
                colcount,
                sum(int(cell.get('colspan', 1)) or 1 for cell in cells[:-1]) + len(cells[-1:]) + len(rowspans))
            # update rowspan bookkeeping; 0 is a span to the bottom.
            rowspans += [int(cell.get('rowspan', 1)) or len(rows) - row_id for cell in cells]
            rowspans = [span - 1 for span in rowspans if span > 1]

        return colcount

    @classmethod
    def from_html(cls, table_tag: lxml.html.HtmlElement):

        rows = table_tag.xpath('./tr')
        num_rows = len(rows)
        num_cols = Table._get_num_columns(rows)

        table_2d = [[{'value': None,
                      'words': []}] * num_cols for _ in range(num_rows)]

        # fill matrix from row data
        rowspans = {}  # track pending rowspans, column number mapping to count
        for row, row_elem in enumerate(rows):
            span_offset = 0  # how many columns are skipped due to row and colspans
            for col, cell in enumerate(row_elem.xpath('./td | ./th')):
                # adjust for preceding row and colspans
                col += span_offset
                while rowspans.get(col, 0):
                    span_offset += 1
                    col += 1

                # fill table data
                rowspan = rowspans[col] = int(cell.get('rowspan', 1)) or num_rows - row
                colspan = int(cell.get('colspan', 1)) or num_cols - col
                # next column is offset by the colspan
                span_offset += colspan - 1

                value = ' '.join(cell.text_content().split())
                value = value if value else None

                words = Table.extract_text_recursively(root=cell, out=[], xbrl_tags=['nonfraction'])

                for drow, dcol in product(range(rowspan), range(colspan)):
                    try:
                        table_2d[row + drow][col + dcol] = {'value': value,
                                                            'words': words}

                        rowspans[col + dcol] = rowspan
                    except IndexError:
                        # rowspan or colspan outside the confines of the table
                        pass

            # update rowspan bookkeeping
            rowspans = {c: s - 1 for c, s in rowspans.items() if s > 1}

        # get valid column ids
        col_ids_to_keep = {0}
        for row_id, row in enumerate(table_2d):
            for col_id, cell_ in enumerate(row):
                for word in cell_['words']:
                    if word['info']:
                        col_ids_to_keep.add(col_id)
                        break

        cells = []
        cell_id = 0
        for row_id, row in enumerate(table_2d):
            for col_id, cell_ in enumerate(row):
                if col_id in col_ids_to_keep:
                    prev_row_id = cells[-1].row if cells else 0
                    prev_col_id = cells[-1].col if cells else -1

                    new_col_id = prev_col_id + 1 if prev_row_id == row_id else 0
                    cells.append(Cell.from_dict({'id_': cell_id,
                                                 'row': row_id,
                                                 'col': new_col_id,
                                                 'value': cell_['value'],
                                                 'words': cell_['words']}))
                    cell_id += 1

        return cls(cells=cells, tag='table', id_=0, value=html.unescape(lh.tostring(table_tag).decode()))


@dataclass
class Document:
    id_: str
    segments: List[Segment]

    @classmethod
    def from_dict(cls, d: Dict):
        d["segments"] = [seg.from_dict() for seg in d["segments"]]
        return cls(**d)

    def to_dict(self):
        d = self.__dict__
        d["segments"] = [seg.to_dict() for seg in self.segments]
        return d


@dataclass
class Corpus:
    documents: List[Document]
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict):
        d["documents"] = [doc.from_dict() for doc in d["documents"]]
        return cls(**d)

    def to_dict(self):
        d = self.__dict__
        d["documents"] = [doc.to_dict() for doc in self.documents]
        return d


# def table_to_2d(table_tag):
#     rowspans = []  # track pending rowspans
#     rows = table_tag.xpath('./tr')
#
#     # first scan, see how many columns we need
#     colcount = 0
#     for r, row in enumerate(rows):
#         cells = row.xpath('./td | ./th')
#         # count columns (including spanned).
#         # add active rowspans from preceding rows
#         # we *ignore* the colspan value on the last cell, to prevent
#         # creating 'phantom' columns with no actual cells, only extended
#         # colspans. This is achieved by hardcoding the last cell width as 1.
#         # a colspan of 0 means “fill until the end” but can really only apply
#         # to the last cell; ignore it elsewhere.
#         colcount = max(
#             colcount,
#             sum(int(c.get('colspan', 1)) or 1 for c in cells[:-1]) + len(cells[-1:]) + len(rowspans))
#         # update rowspan bookkeeping; 0 is a span to the bottom.
#         rowspans += [int(c.get('rowspan', 1)) or len(rows) - r for c in cells]
#         rowspans = [s - 1 for s in rowspans if s > 1]
#
#     # it doesn't matter if there are still rowspan numbers 'active'; no extra
#     # rows to show in the table means the larger than 1 rowspan numbers in the
#     # last table row are ignored.
#
#     # build an empty matrix for all possible cells
#     table = [[None] * colcount for row in rows]
#
#     # fill matrix from row data
#     rowspans = {}  # track pending rowspans, column number mapping to count
#     for row, row_elem in enumerate(rows):
#         span_offset = 0  # how many columns are skipped due to row and colspans
#         for col, cell in enumerate(row_elem.xpath('./td | ./th')):
#             # adjust for preceding row and colspans
#             col += span_offset
#             while rowspans.get(col, 0):
#                 span_offset += 1
#                 col += 1
#
#             # fill table data
#             rowspan = rowspans[col] = int(cell.get('rowspan', 1)) or len(rows) - row
#             colspan = int(cell.get('colspan', 1)) or colcount - col
#             # next column is offset by the colspan
#             span_offset += colspan - 1
#             value = cell.text_content()
#             for drow, dcol in product(range(rowspan), range(colspan)):
#                 try:
#                     table[row + drow][col + dcol] = value
#                     rowspans[col + dcol] = rowspan
#                 except IndexError:
#                     # rowspan or colspan outside the confines of the table
#                     pass
#
#         # update rowspan bookkeeping
#         rowspans = {c: s - 1 for c, s in rowspans.items() if s > 1}
#
#     return table


def main():
    from rich import print as pprint

    path = '/scratch/data/sec-edgar/tesla_xbrl/'
    file_name = 'tsla-10k_20191231.htm'
    file = f'{path}{file_name}'

    with open(file, 'r', encoding='utf-8') as fp:
        html_text = fp.read()

    html_text = html.unescape(html_text)
    html_text = unicodedata.normalize("NFKC", html_text)

    cleaner = Cleaner(scripts=False,
                      javascript=False,
                      comments=False,
                      style=True,
                      inline_style=True,
                      links=False,
                      meta=False,
                      page_structure=False,
                      processing_instructions=False,
                      embedded=False,
                      frames=False,
                      forms=False,
                      annoying_tags=False,
                      remove_tags=None,
                      allow_tags=None,
                      kill_tags=None,
                      remove_unknown_tags=False,
                      safe_attrs_only=False,
                      add_nofollow=False,
                      host_whitelist=(),
                      whitelist_tags={'iframe', 'embed'})

    html_text = cleaner.clean_html(html_text.encode()).decode()

    tree = lh.fromstring(html_text)

    tables = tree.xpath('//table')

    for tab in tables[48: 52]:

        tab_str = lh.tostring(tab, encoding=str)

        tab = Table.from_html(tab)
        tab.show()


def recursive():
    from rich import print as pprint

    bla = """
        <td valign="top"> some test
    <p>Preferred stock; $<nonfraction unitref="U_iso4217USD_xbrlishares"
    id="F_000107" name="us-gaap:PreferredStockParOrStatedValuePerShare"
    contextref="C_0001318605_20191231" decimals="INF"><nonfraction
    unitref="U_iso4217USD_xbrlishares" id="F_000108"
    name="us-gaap:PreferredStockParOrStatedValuePerShare"
    contextref="C_0001318605_20181231"
    decimals="INF">0.001</nonfraction></nonfraction> par value; <nonfraction
    unitref="U_xbrlishares" id="F_000109"
    name="us-gaap:PreferredStockSharesAuthorized" contextref="C_0001318605_20191231"
    decimals="INF" scale="6"><nonfraction unitref="U_xbrlishares" id="F_000110"
    name="us-gaap:PreferredStockSharesAuthorized" contextref="C_0001318605_20181231"
    decimals="INF" scale="6">100</nonfraction></nonfraction> shares authorized;</p>
    <p>   <nonfraction unitref="U_xbrlishares" id="F_000111"
    name="us-gaap:PreferredStockSharesIssued" contextref="C_0001318605_20191231"
    decimals="INF" format="ixt-sec:numwordsen" scale="6"><nonfraction
    unitref="U_xbrlishares" id="F_000112" name="us-gaap:PreferredStockSharesIssued"
    contextref="C_0001318605_20181231" decimals="INF" format="ixt-sec:numwordsen"
    scale="6"><nonfraction unitref="U_xbrlishares" id="F_000113"
    name="us-gaap:PreferredStockSharesOutstanding"
    contextref="C_0001318605_20191231" decimals="INF" format="ixt-sec:numwordsen"
    scale="6"><nonfraction unitref="U_xbrlishares" id="F_000114"
    name="us-gaap:PreferredStockSharesOutstanding"
    contextref="C_0001318605_20181231" decimals="INF" format="ixt-sec:numwordsen"
    scale="6">no</nonfraction></nonfraction></nonfraction></nonfraction> shares
    issued and outstanding</p> hallo </td>
    """

    cell = lh.fromstring(bla)
    l = []
    out = extract_text_recursively(cell, l, ['nonfraction'])
    pprint(out)

    # text, entities = iter_elem1(cell, text='', entities=[])
    # pprint(text)
    # pprint(entities)


def iter_elem1(node, text: str, entities: List):

    start_entity = len(text)
    text += ' '.join(node.text.split()) if node.text else ''

    children = node.getchildren()
    if children:
        for child in children:
            text, entities = iter_elem1(child, text, entities)

    if node.tag == 'nonfraction':
        info = dict(node.items())
        entities.append({'info': info,
                         'start': start_entity,
                         'end': len(text)})

    text += ' '.join(node.tail.split()) if node.tail else ''

    return text, entities


def extract_text_recursively(root: lxml.html.HtmlElement, out: List, xbrl_tags: List):

    for node in root:

        text = ' '.join(node.text.split()) if node.text else None
        if text:
            out.append({'text': text,
                        'info': []})

        out = extract_text_recursively(node, out, xbrl_tags)

        if node.tag in xbrl_tags:
            info = dict(node.items())
            out[-1]['info'].append(info)

        tail = ' '.join(node.tail.split()) if node.tail else None
        if tail:
            out.append({'text': tail,
                        'info': []})

    return out


def iter_elem(node, l):
    text = ' '.join(node.text.split()) if node.text else None
    if text:
        l.append({'text': text,
                  'info': []})

    children = node.getchildren()
    if children:
        for child in children:
            iter_elem(child, l)

    if node.tag == 'nonfraction':
        info = dict(node.items())
        l[-1]['info'].append(info)

    tail = ' '.join(node.tail.split()) if node.tail else None
    if tail:
        l.append({'text': tail,
                  'info': []})

    return l


if __name__ == '__main__':
    main()
