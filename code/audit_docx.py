import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {
    'w': W,
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'cp': 'http://schemas.openxmlformats.org/package/2006/metadata/core-properties',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
}
FONT = 'Times New Roman'
BRITISH = re.compile(
    r'\b(colour\w*|behaviour\w*|analys(?:e|ed|ing)|optimis\w*|modelling|normalis\w*|centre\w*|labelled|labelling|favour\w*|programme\w*)\b',
    re.I)
TOOLS = re.compile(r'claude|anthropic|codex|openai|chatgpt|\bgpt|copilot|grammarly|google|pandoc|libreoffice|python|lxml|latex2|\bllm\b|assistant|generated', re.I)
PROSE_TOOLS = re.compile(r'claude|anthropic|codex|openai|chatgpt|copilot|grammarly|pandoc|libreoffice|language model|\bllm\b', re.I)
REVISIONS = ['ins', 'del', 'moveFrom', 'moveTo', 'rPrChange', 'pPrChange', 'tblPrChange', 'tblGridChange', 'sectPrChange',
             'trPrChange', 'tcPrChange', 'commentRangeStart', 'commentRangeEnd', 'commentReference', 'vanish', 'specVanish']
KEEP_IDENTICAL = ['word/styles.xml', 'word/settings.xml', 'word/theme/theme1.xml', 'word/numbering.xml', 'word/fontTable.xml']
REMOVED_OK = ('customXML/', 'docProps/custom.xml', 'word/media/')


class Report:
    def __init__(self):
        self.fail = 0
        self.warn = 0

    def ok(self, msg):
        print('PASS  ' + msg)

    def bad(self, msg):
        self.fail += 1
        print('FAIL  ' + msg)

    def note(self, msg):
        self.warn += 1
        print('WARN  ' + msg)

    def check(self, cond, msg, detail=''):
        if cond:
            self.ok(msg)
        else:
            self.bad(msg + (': ' + detail if detail else ''))


def canon(e):
    return etree.tostring(e, method='c14n')


def ptext(p):
    return ''.join(t.text or '' for t in p.iter('{%s}t' % W))


def ptext_tabs(p):
    out = []
    for e in p.iter():
        if e.tag == '{%s}t' % W:
            out.append(e.text or '')
        elif e.tag == '{%s}tab' % W and e.getparent().tag == '{%s}r' % W:
            out.append('\t')
    return ''.join(out)


def wval(e, tag):
    c = e.find('w:' + tag, NS)
    if c is None:
        return None
    v = c.get('{%s}val' % W)
    return v not in ('0', 'false', 'off')


def sz_of(rpr):
    s = rpr.find('w:sz', NS) if rpr is not None else None
    return int(s.get('{%s}val' % W)) if s is not None else None


def font_ok(rpr):
    f = rpr.find('w:rFonts', NS) if rpr is not None else None
    return f is not None and all(f.get('{%s}%s' % (W, k)) == FONT for k in ('ascii', 'hAnsi', 'cs', 'eastAsia'))


def audit(path, template, render, rep):
    z = zipfile.ZipFile(path)
    zt = zipfile.ZipFile(template)
    names = z.namelist()
    tnames = zt.namelist()
    parts = {}
    for n in names:
        if n.endswith('.xml') or n.endswith('.rels'):
            try:
                parts[n] = etree.fromstring(z.read(n))
            except etree.XMLSyntaxError as e:
                rep.bad('malformed XML in %s: %s' % (n, e))
    rep.ok('opened package with %d parts' % len(names))
    doc = parts['word/document.xml']
    body = doc.find('w:body', NS)
    tbody = etree.fromstring(zt.read('word/document.xml')).find('w:body', NS)

    rep.check(canon(body.find('w:sectPr', NS)) == canon(tbody.find('w:sectPr', NS)), 'sectPr identical to template')
    for n in KEEP_IDENTICAL:
        rep.check(n in names and z.read(n) == zt.read(n), 'part unchanged from template: ' + n)
    extra = [n for n in names if n not in tnames]
    lost = [n for n in tnames if n not in names and not n.startswith(REMOVED_OK)]
    rep.check(not lost, 'no template parts lost', ', '.join(lost))
    rep.check(not [n for n in extra if not n.startswith('word/media/')], 'no unexpected new parts', ', '.join(extra))
    rep.check(not any('comments' in n or 'custom' in n.lower() for n in names), 'no comments or custom property parts')
    ct = z.read('[Content_Types].xml').decode('utf8')
    rep.check(not re.search(r'custom|comments', ct, re.I), 'content types free of custom and comment entries')
    rels = z.read('word/_rels/document.xml.rels').decode('utf8') + z.read('_rels/.rels').decode('utf8')
    rep.check(not re.search(r'custom|comments', rels, re.I), 'relationships free of custom and comment entries')
    broken = []
    for rn in ('word/_rels/document.xml.rels', '_rels/.rels'):
        base = 'word/' if rn.startswith('word') else ''
        for r in parts[rn]:
            t = r.get('Target')
            if r.get('TargetMode') != 'External' and (base + t).replace('word/../', '') not in names:
                broken.append(t)
    rep.check(not broken, 'all relationship targets exist', ', '.join(broken))

    hits = [r for r in REVISIONS if doc.xpath('//w:%s' % r, namespaces=NS)]
    rep.check(not hits, 'no revisions, comments, or hidden text', ', '.join(hits))

    paras = list(doc.iter('{%s}p' % W))
    bad_font, bad_size, bad_mark = 0, [], 0
    for r in doc.iter('{%s}r' % W):
        if not font_ok(r.find('w:rPr', NS)):
            bad_font += 1
    for ppr in doc.iter('{%s}pPr' % W):
        mark = ppr.find('w:rPr', NS)
        if mark is not None and not font_ok(mark):
            bad_mark += 1
    rep.check(bad_font == 0 and bad_mark == 0, 'Times New Roman set on every run and paragraph mark',
              '%d runs, %d marks' % (bad_font, bad_mark))
    mathfont = [r for r in doc.iter('{%s}r' % NS['m']) if r.find('w:rPr', NS) is None]
    rep.check(not mathfont, 'math runs carry explicit size and font', str(len(mathfont)))

    top = [p for p in body if p.tag == '{%s}p' % W]
    title_p = next(p for p in top if ptext(p).strip())
    author_p = [p for p in top if ptext(p).strip()][1]
    allowed = {id(title_p): 24, id(author_p): 22}
    for p in paras:
        want = allowed.get(id(p), 20)
        for rpr_e in list(p.iter('{%s}rPr' % W)):
            s = sz_of(rpr_e)
            if s is not None and s != want:
                bad_size.append((ptext(p)[:30], s))
    rep.check(not bad_size, 'font sizes: title 12, authors 11, everything else 10 pt', str(bad_size[:3]))
    rep.check(ptext(title_p) == ptext(title_p).upper() and wval(title_p.find('w:r', NS).find('w:rPr', NS), 'b'), 'title bold and uppercase')
    rep.check(all(wval(r.find('w:rPr', NS), 'b') and wval(r.find('w:rPr', NS), 'i') for r in author_p.findall('w:r', NS)),
              'author line bold italic')

    texts = [(p, ptext(p)) for p in paras]
    allprose = '\n'.join(t for _, t in texts)
    rep.check('\u2014' not in allprose and '\u2014' not in z.read('docProps/core.xml').decode('utf8'), 'no em dashes')
    ref_idx = next((i for i, (_, t) in enumerate(texts) if t.strip() == 'REFERENCES'), None)
    prose_end = ref_idx if ref_idx is not None else len(texts)
    endash = [t[:40] for _, t in texts[:prose_end] if '–' in t]
    rep.check(not endash, 'no en dashes in prose', str(endash[:3]))
    refdash = [t[:40] for _, t in texts[prose_end:] if re.search(r'(?<!\d)–|–(?!\d)', t)]
    rep.check(not refdash, 'en dashes in references only inside numeric ranges', str(refdash[:3]))
    brit = sorted({m.group(0).lower() for _, t in texts[:prose_end] for m in BRITISH.finditer(t)})
    rep.check(not brit, 'no British spellings in prose', ', '.join(brit))
    rbrit = sorted({m.group(0).lower() for _, t in texts[prose_end:] for m in BRITISH.finditer(t)})
    if rbrit:
        rep.note('British spellings inside reference titles (verify they are verbatim): ' + ', '.join(rbrit))

    skip = ('word/document.xml', 'word/styles.xml', 'word/theme/theme1.xml', 'word/settings.xml')
    blob = ''.join(z.read(n).decode('utf8', 'ignore') for n in names if n.endswith(('.xml', '.rels')) and n not in skip)
    blob += ' '.join(names)
    tool_hits = sorted({m.group(0).lower() for m in TOOLS.finditer(blob)})
    rep.check(not tool_hits, 'no tool names in package metadata or part names', ', '.join(tool_hits))
    disclosure = [t for _, t in texts[:prose_end] if 'generative AI' in t]
    rep.check(len(disclosure) == 1, 'AI-use disclosure present once in Acknowledgments')
    prose_body = [t for _, t in texts[:prose_end] if 'generative AI' not in t]
    prose_hits = sorted({m.group(0).lower() for t in prose_body for m in PROSE_TOOLS.finditer(t)})
    rep.check(not prose_hits, 'software names only in the disclosure', ', '.join(prose_hits))
    rep.check(not any(re.search(r'\bAI\b', t) for t in prose_body), 'AI mentioned only in the disclosure')

    core = parts['docProps/core.xml']
    g = lambda t: (core.find(t, NS).text if core.find(t, NS) is not None else None)
    rep.check(g('dc:creator') == 'Qi Ding Ng' and g('cp:lastModifiedBy') == 'Qi Ding Ng', 'core.xml creator and lastModifiedBy')
    rep.check(bool(g('dc:title')) and ptext(title_p).upper() == g('dc:title').upper(), 'core.xml title matches manuscript title')

    abs_i = next((i for i, (_, t) in enumerate(texts) if t.strip() == 'ABSTRACT'), None)
    rep.check(abs_i is not None, 'ABSTRACT heading present')
    if abs_i is not None:
        ah = texts[abs_i][0]
        rr = ah.find('w:r', NS).find('w:rPr', NS)
        rep.check(wval(rr, 'b') and wval(rr, 'i'), 'ABSTRACT heading bold italic')
        ap = next(p for p, t in texts[abs_i + 1:] if t.strip())
        words = len(ptext(ap).split())
        rep.check(100 <= words <= 200, 'abstract word count %d within 100 to 200' % words)
        runs = [r for r in ap.findall('w:r', NS) if ptext(r).strip()]
        rep.check(all(wval(r.find('w:rPr', NS), 'i') for r in runs), 'abstract italic')
        rep.check(ap.find('w:pPr/w:jc', NS).get('{%s}val' % W) == 'both', 'abstract justified')
    kw = next((t for _, t in texts if t.startswith('Keywords:')), None)
    rep.check(kw is not None, 'Keywords line present')
    if kw:
        n = len([x for x in kw[len('Keywords:'):].strip().rstrip('.').split(';') if x.strip()])
        rep.check(n <= 8, 'keyword count %d is at most 8' % n)

    h1 = []
    for p, t in texts:
        ppr = p.find('w:pPr', NS)
        ind = ppr.find('w:ind', NS) if ppr is not None else None
        if ind is not None and ind.get('{%s}hanging' % W) == '567' and ind.get('{%s}left' % W) == '567' and re.match(r'^\d+\.0\t', ptext_tabs(p)):
            h1.append(ptext_tabs(p))
    rep.check(all(x.split('\t')[1] == x.split('\t')[1].upper() and int(x.split('.')[0]) == i + 1 for i, x in enumerate(h1)),
              'level-1 headings numbered N.0 in sequence and uppercase (%d found)' % len(h1))
    sub = [ptext_tabs(p) for p, t in texts if re.match(r'^\d+\.\d+(\.\d+)?\t', ptext_tabs(p)) and not re.match(r'^\d+\.0\t', ptext_tabs(p))]
    small = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'for', 'to', 'with', 'by', 'at', 'vs', 'via', 'from', 'as'}
    odd = []
    for s in sub:
        words = s.split('\t')[1].split()
        for j, w in enumerate(words):
            if w[0].isalpha() and w[0].islower() and (j == 0 or w not in small):
                odd.append(s)
                break
    if odd:
        rep.note('subheadings not in Title Case: ' + '; '.join(odd[:4]))
    else:
        rep.ok('subheadings in Title Case (%d)' % len(sub))

    children = list(body)

    def nonblank_next(i):
        j = i + 1
        while j < len(children) and children[j].tag == '{%s}p' % W and not ptext(children[j]).strip() and not list(children[j].iter('{%s}drawing' % W)):
            j += 1
        return children[j] if j < len(children) else None

    def nonblank_prev(i):
        j = i - 1
        while j >= 0 and children[j].tag == '{%s}p' % W and not ptext(children[j]).strip() and not list(children[j].iter('{%s}drawing' % W)):
            j -= 1
        return children[j] if j >= 0 else None

    figs, tabs = [], []
    for i, e in enumerate(children):
        if e.tag != '{%s}p' % W:
            continue
        t = ptext(e)
        jc = e.find('w:pPr/w:jc', NS)
        centered = jc is not None and jc.get('{%s}val' % W) == 'center'
        m = re.match(r'^Fig\. (\d+): (\S.*)$', t)
        if m:
            figs.append(int(m.group(1)))
            prev = nonblank_prev(i)
            rep.check(centered and m.group(2)[0].isupper(), 'caption format Fig. %s centered with capital title' % m.group(1))
            rep.check(prev is not None and bool(list(prev.iter('{%s}drawing' % W))), 'Fig. %s caption sits below its figure' % m.group(1))
        m = re.match(r'^Table (\d+): (\S.*)$', t)
        if m and (nonblank_next(i) is not None and nonblank_next(i).tag == '{%s}tbl' % W):
            tabs.append(int(m.group(1)))
            rep.check(centered and m.group(2)[0].isupper(), 'caption format Table %s centered with capital title' % m.group(1))
    rep.check(figs == list(range(1, len(figs) + 1)), 'figure numbering sequential (%d figures)' % len(figs))
    rep.check(tabs == list(range(1, len(tabs) + 1)), 'table numbering sequential (%d tables)' % len(tabs))
    ndraw = len(list(body.iter('{%s}drawing' % W)))
    rep.check(ndraw == len(figs), 'every drawing has a caption', '%d drawings, %d captions' % (ndraw, len(figs)))
    data_tables = [e for e in children if e.tag == '{%s}tbl' % W and e.find('w:tblPr/w:tblDescription', NS) is None]
    rep.check(len(data_tables) == len(tabs), 'every data table has a caption', '%d tables, %d captions' % (len(data_tables), len(tabs)))
    capre = re.compile(r'^(Fig\. \d+|Table \d+): ')
    body_text = '\n'.join(t for _, t in texts[:prose_end] if not capre.match(t))
    for n in figs:
        rep.check(re.search(r'\bFig\. %d\b' % n, body_text) is not None, 'Fig. %d referenced in text' % n)
    for n in tabs:
        rep.check(re.search(r'\bTable %d\b' % n, body_text) is not None, 'Table %d referenced in text' % n)
    for t in data_tables:
        pr = t.find('w:tblPr', NS)
        jc = pr.find('w:jc', NS)
        rep.check(jc is not None and jc.get('{%s}val' % W) == 'center', 'data table centered')
        first = t.find('w:tr/w:tc/w:p', NS)
        rep.check(all(wval(r.find('w:rPr', NS), 'b') for r in first.findall('w:r', NS)), 'table header row bold')

    eq_nums = [int(m.group(1)) for _, t in texts for m in [re.fullmatch(r'\((\d+)\)', t.strip())] if m]
    rep.check(eq_nums == list(range(1, len(eq_nums) + 1)), 'equation numbering sequential (%d numbered)' % len(eq_nums))
    nmath = len(list(doc.iter('{%s}oMath' % NS['m'])))
    rep.ok('%d native Word equations (OMML)' % nmath)

    refs = []
    for p, t in texts[prose_end:]:
        m = re.match(r'^\[(\d+)\]\t', ptext_tabs(p))
        if m:
            refs.append(int(m.group(1)))
            ind = p.find('w:pPr/w:ind', NS)
            if ind is None or ind.get('{%s}hanging' % W) != '567' or ind.get('{%s}left' % W) != '567':
                rep.bad('reference [%s] lacks 1 cm hanging indent' % m.group(1))
    rep.check(refs == list(range(1, len(refs) + 1)), 'reference list numbered sequentially, count %d' % len(refs))
    cited, first_seen = set(), []
    for _, t in texts[:prose_end]:
        for grp in re.findall(r'\[(\d+(?:\]\s*,\s*\[\d+)*)\]', t):
            for n in re.findall(r'\d+', grp):
                n = int(n)
                cited.add(n)
                if n not in first_seen:
                    first_seen.append(n)
    rep.check(cited <= set(refs), 'all citations resolve to a reference', str(sorted(cited - set(refs))))
    rep.check(set(refs) <= cited, 'every reference is cited', str(sorted(set(refs) - cited)))
    rep.check(first_seen == sorted(first_seen), 'citations numbered in order of first appearance')
    ref_fonts_ok = all(font_ok(r.find('w:rPr', NS)) for p, _ in texts[prose_end:] for r in p.findall('w:r', NS))
    rep.check(ref_fonts_ok, 'reference entries use Times New Roman')
    rep.check(any(t.strip() in ('REFERENCES',) for _, t in texts), 'REFERENCES heading present')
    if any(t.strip() == 'BIOGRAPHY' for _, t in texts):
        bi = next(i for i, (_, t) in enumerate(texts) if t.strip() == 'BIOGRAPHY')
        rep.check(bi > prose_end, 'BIOGRAPHY follows REFERENCES')
    else:
        rep.note('no BIOGRAPHY section')

    if render:
        soffice = shutil.which('soffice') or shutil.which('libreoffice')
        if not soffice:
            rep.note('LibreOffice not found, skipped PDF conversion')
        else:
            out = Path(render)
            out.mkdir(parents=True, exist_ok=True)
            subprocess.run([soffice, '--headless', '--convert-to', 'pdf', '--outdir', str(out), str(path)], capture_output=True, timeout=180)
            pdf = out / (Path(path).stem + '.pdf')
            rep.check(pdf.exists() and pdf.stat().st_size > 0, 'converted to PDF with LibreOffice')
            if pdf.exists() and shutil.which('pdftoppm'):
                subprocess.run(['pdftoppm', '-r', '80', '-f', '1', '-l', '2', '-png', str(pdf), str(out / 'page')], capture_output=True)
                rep.ok('rendered pages to ' + str(out / 'page-*.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('docx', nargs='?', default=str(ROOT / 'paper' / 'manuscript.docx'))
    ap.add_argument('--template', default=str(ROOT / 'paper' / 'mjcs_template.docx'))
    ap.add_argument('--render', nargs='?', const=str(ROOT / 'paper' / 'render'), default=None)
    a = ap.parse_args()
    rep = Report()
    audit(a.docx, a.template, a.render, rep)
    print('\n%d failed, %d warnings' % (rep.fail, rep.warn))
    sys.exit(1 if rep.fail else 0)


if __name__ == '__main__':
    main()
