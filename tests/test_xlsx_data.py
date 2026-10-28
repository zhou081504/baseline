"""Synthetic OOXML fixtures for the input adapter, no model downloads needed."""
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from xml.sax.saxutils import escape
from zipfile import ZipFile

from common.data import load_records, make_splits, sample_unit


def workbook(path, rows, shared=False):
    strings = []
    xml = []
    for i, row in enumerate([['id','s','merged_text','turn_label','label']]+rows, 1):
        cells = []
        for col, value in zip('ABCDE', row):
            if value is None:
                continue
            ref=f'{col}{i}'
            if col == 'D' and i > 1:
                # Even an Excel error/formula in turn_label must not be decoded.
                cells.append(f'<c r="{ref}" t="e"><f>1/0</f><v>#DIV/0!</v></c>')
            elif shared:
                strings.append(value)
                cells.append(f'<c r="{ref}" t="s"><v>{len(strings)-1}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>')
        xml.append(f'<row r="{i}">{"".join(cells)}</row>')
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    with ZipFile(path,'w') as z:
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="data" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml',f'<worksheet xmlns="{ns}"><sheetData>{"".join(xml)}<row r="1048576"><c r="A1048576"/></row></sheetData><mergeCells><mergeCell ref="A2:A4"/><mergeCell ref="E2:E4"/></mergeCells></worksheet>')
        if shared:
            z.writestr('xl/sharedStrings.xml',f'<sst xmlns="{ns}">'+''.join(f'<si><t xml:space="preserve">{escape(v)}</t></si>' for v in strings)+'</sst>')


class XlsxDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'calls.xlsx'

    def tearDown(self):
        self.temp.cleanup()

    def test_merged_cells_numeric_order_and_exact_text(self):
        rows=[['c20','s1',' 客服：甲\n客户：乙 ',1,'信访风险件'],[None,'s10','客户：十',1,None],[None,'s2','客服：二',None,None],['c3','s1','客户：正常',1,'正常']]
        for shared in (False,True):
            workbook(self.path,rows,shared)
            records=load_records(self.path)
            self.assertEqual([r.id for r in records],['c20','c3'])
            self.assertEqual(records[0].raw_text,' 客服：甲\n客户：乙 \n客服：二\n客户：十')
            self.assertEqual([r.label for r in records],[1,0])
            self.assertEqual(sample_unit(self.path),'one_c_all_s_segments')

    def test_turn_labels_do_not_change_records(self):
        rows=[['c1','s1','客户：正文',1,'正常']]
        workbook(self.path,rows)
        before=load_records(self.path)
        rows[0][3]=None
        workbook(self.path,rows)
        self.assertEqual(before,load_records(self.path))

    def test_invalid_rows_fail_without_guessing(self):
        bad_cases=[
            [['c1','s1','正文',None,None]],
            [['c1','s1','正文',None,'正常'],[None,'s2','正文',None,'信访风险件']],
            [['c1','s1','正文',None,'正常'],[None,'s1','正文',None,None]],
            [[None,'s1','正文',None,'正常']],
            [['c1','s1','',None,'正常']],
            [['c1','sX','正文',None,'正常']],
            [['c1','s1','正文',None,'错误标签']],
            [['c1','s1','正文',None,'正常'],['c2','s1','正文',None,None]],
        ]
        for rows in bad_cases:
            with self.subTest(rows=rows):
                workbook(self.path,rows)
                with self.assertRaises(ValueError):load_records(self.path)

    def test_split_after_grouping_reproducible_no_overlap(self):
        rows=[]
        for i in range(1,81):
            rows.extend([[f'c{i}','s1',f'客服：{i}',1,'信访风险件' if i%2 else '正常'],[None,'s2','客户：结束',None,None]])
        workbook(self.path,rows)
        records=load_records(self.path)
        a=make_splits(records); b=make_splits(records)
        ids={k:{r.id for r in v} for k,v in a.items()}
        self.assertEqual(a,b)
        self.assertEqual(sum(map(len,ids.values())),80)
        self.assertFalse(ids['train'] & ids['test'] or ids['train'] & ids['val'] or ids['val'] & ids['test'])
        self.assertTrue(all('客户：结束' in r.raw_text for rs in a.values() for r in rs))

    def test_legacy_csv_unchanged(self):
        path=self.path.with_suffix('.csv')
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.writer(f);w.writerow(['id','merged_text','label','turn_label']);w.writerow(['C000001','甲\n乙','正常','ignored'])
        record=load_records(path)[0]
        self.assertEqual((record.id,record.raw_text,record.label),('C000001','甲\n乙',0))


if __name__ == '__main__':
    unittest.main()
