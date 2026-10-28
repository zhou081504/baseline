"""Read segmented call workbooks using only the Python standard library.

Only id, s, merged_text and sample-level label are decoded. turn_label and
all other columns are ignored, even when they contain formulas or errors.
"""
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
RID = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
FIELDS = {'id', 's', 'merged_text', 'label'}
SEGMENT = re.compile(r's([1-9][0-9]*)\Z')


def _text(element):
    # Exclude phonetic annotation text (rPh) from rich strings.
    return ''.join(child.text or '' for child in element.findall(NS+'t')) + ''.join(
        child.text or '' for run in element.findall(NS+'r') for child in run.findall(NS+'t'))


def _elements(archive, member, tag):
    """Stream XML and discard completed rows/strings, including styled blanks."""
    with archive.open(member) as handle:
        stack = []
        for event, node in ET.iterparse(handle, events=('start', 'end')):
            if event == 'start':
                stack.append(node)
            else:
                if node.tag == NS+tag:
                    yield node
                    if len(stack) > 1:
                        stack[-2].remove(node)
                    node.clear()
                stack.pop()


def _value(cell, strings):
    if cell.find(NS+'f') is not None:
        raise ValueError(f'Formula not allowed in input field {cell.get("r")}')
    kind = cell.get('t')
    if kind == 'inlineStr':
        node = cell.find(NS+'is')
        return _text(node) if node is not None else ''
    value = cell.findtext(NS+'v', '')
    if kind == 'e':
        raise ValueError(f'Excel error in input field {cell.get("r")}: {value}')
    return strings[int(value)] if kind == 's' and value else value


def load_segmented_rows(path):
    """Yield one (id, original segments joined with newline, label) per c.

    Use the first worksheet. Merged id/label values are carried only within
    their sample. Samples retain first-appearance order; s is sorted numerically.
    Missing/conflicting labels and duplicate s are errors, never guessed.
    """
    with ZipFile(Path(path)) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [_text(node) for node in _elements(archive, 'xl/sharedStrings.xml', 'si')]
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheet = workbook.find(NS+'sheets/'+NS+'sheet')
        if sheet is None:
            raise ValueError('Workbook has no worksheets')
        relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = next((r.get('Target') for r in relationships if r.get('Id') == sheet.get(RID)), None)
        if not target:
            raise ValueError('Cannot resolve first worksheet')
        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('xl', target))
        columns = None
        current_id = None
        groups = {}
        for row in _elements(archive, member, 'row'):
            cells = {re.sub(r'\d+$', '', c.get('r', '')): c for c in row.findall(NS+'c')}
            if columns is None:
                header = {col: _value(cell, strings).strip() for col, cell in cells.items()}
                if not any(header.values()):
                    continue
                columns = {col: name for col, name in header.items() if name in FIELDS}
                if set(columns.values()) != FIELDS or len(columns) != len(FIELDS):
                    raise ValueError(f'First worksheet requires unique columns {sorted(FIELDS)}')
                continue
            values = {name: _value(cells[col], strings) if col in cells else '' for col, name in columns.items()}
            if not any(values.values()):
                continue
            ident = values['id'].strip()
            if ident:
                if not re.fullmatch(r'c[1-9][0-9]*', ident):
                    raise ValueError(f'Invalid sample id {ident!r} at row {row.get("r")}')
                current_id = ident
            if current_id is None:
                raise ValueError(f'Segment has no sample id at row {row.get("r")}')
            group = groups.setdefault(current_id, {'segments': {}, 'labels': set()})
            label = values['label'].strip()
            if label:
                group['labels'].add(label)
            s = values['s'].strip()
            match = SEGMENT.fullmatch(s)
            if not match or not values['merged_text'].strip():
                raise ValueError(f'Missing/invalid s or text for {current_id}, row {row.get("r")}')
            index = int(match[1])
            if index in group['segments']:
                raise ValueError(f'Duplicate segment {current_id}/{s}')
            group['segments'][index] = values['merged_text']
        if not groups:
            raise ValueError('Workbook contains no samples')
        for ident, group in groups.items():
            if len(group['labels']) != 1:
                raise ValueError(f'Missing/conflicting sample label for {ident}: {sorted(group["labels"])}')
            yield ident, '\n'.join(text for _, text in sorted(group['segments'].items())), next(iter(group['labels']))
