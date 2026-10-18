import argparse
import copy
import datetime
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import yaml
from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
}
FONT = 'Times New Roman'
TEXT_WIDTH = 9072
EMU_CM = 360000
CREATOR = 'Qi Ding Ng'
PROTO = dict(title=0, authors=3, aff_name=5, aff_addr=6, emails=14, received=18, abs_head=21, abs_text=23,
             keywords=25, h1=28, h2=33, h3=41, body=30, figcap=55, tabcap=58, ref_head=72, ref_entry=76)
INLINE = re.compile(
    r'(?P<esc>\\[$*^~\[\]{}\\])'
    r'|(?P<math>\$(?!\s)(?P<mt>[^$]+?)(?<!\s)\$)'
    r'|(?P<cite>\[@[^\]]+\])'
    r'|(?P<ref>\{\{(?:fig|tab|eq):[\w.-]+\}\})'
    r'|(?P<bold>\*\*(?P<bt>.+?)\*\*)'
    r'|(?P<ital>(?<![\w*])\*(?P<it>[^*\s][^*]*?)\*(?![\w*]))'
    r'|(?P<sup>\^(?P<st>[^^\s]+?)\^)'
    r'|(?P<sub>~(?P<bs>[^~\s]+?)~)')
LIST_ITEM = re.compile(r'^(?:[-*] |\d+\. )')
LABEL = re.compile(r'\s*\{([^{}]*)\}\s*$')


def q(tag):
    p, n = tag.split(':')
    return '{%s}%s' % (NS[p], n)


def el(tag, attrs=None, *children):
    e = etree.Element(q(tag))
    for k, v in (attrs or {}).items():
        e.set(q(k), str(v))
    for c in children:
        e.append(c)
    return e


def val(tag, v='1'):
    return el(tag, {'w:val': v})


def rpr(b=False, i=False, sz=20, sup=False, sub=False, color=None):
    r = el('w:rPr', None, el('w:rFonts', {'w:ascii': FONT, 'w:hAnsi': FONT, 'w:cs': FONT, 'w:eastAsia': FONT}))
    if b:
        r.extend([val('w:b'), val('w:bCs')])
    if i:
        r.extend([val('w:i'), val('w:iCs')])
    if color:
        r.append(val('w:color', color))
    r.extend([val('w:sz', sz), val('w:szCs', sz)])
    if sup or sub:
        r.append(val('w:vertAlign', 'superscript' if sup else 'subscript'))
    return r


def run(text, **fmt):
    r = el('w:r', None, rpr(**fmt))
    for part in re.split(r'(\t)', text):
        if part == '\t':
            r.append(el('w:tab'))
        elif part:
            t = el('w:t')
            t.text = part
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            r.append(t)
    return r


def smart_quotes(text):
    def fix(m):
        s = m.group(0)
        if len(s) > 1:
            return s
        prev = m.string[m.start() - 1] if m.start() else ' '
        nxt = m.string[m.end()] if m.end() < len(m.string) else ' '
        opening = prev.isspace() or prev in '([{‘“'
        if s == '"':
            return '“' if opening else '”'
        return '‘' if opening and nxt.isalnum() else '’'
    return re.sub(r'\\\$|\$[^$]*\$|"|\'', fix, text)


def plain_text(text):
    text = re.sub(r'\$([^$]+)\$', r'\1', text)
    text = re.sub(r'\[@[^\]]+\]|\{\{[^}]*\}\}', '', text)
    return re.sub(r'[*^~]', '', text).strip()


def parse_length(spec):
    m = re.fullmatch(r'([\d.]+)\s*(cm|mm|in|%)', spec.strip())
    if not m:
        raise ValueError('bad length: ' + spec)
    n, u = float(m.group(1)), m.group(2)
    cm = {'cm': n, 'mm': n / 10, 'in': n * 2.54, '%': TEXT_WIDTH / 567 * n / 100}[u]
    return cm


def split_front(text):
    m = re.match(r'---\n(.*?)\n---\n', text, re.S)
    if not m:
        raise ValueError('missing front matter')
    return yaml.safe_load(m.group(1)), text[m.end():]


def split_row(line):
    cells = re.split(r'(?<!\\)\|', line.strip())
    if cells and cells[0] == '':
        cells = cells[1:]
    if cells and cells[-1] == '':
        cells = cells[:-1]
    return [c.strip().replace('\\|', '|') for c in cells]


def parse_blocks(text):
    lines = text.split('\n')
    blocks, i, n = [], 0, len(lines)

    def starts_block(s):
        s = s.strip()
        return (s.startswith('#') or s.startswith('![') or s.startswith('Table:') or s.startswith('$$')
                or bool(LIST_ITEM.match(s)) or s.startswith('|'))

    while i < n:
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith('#'):
            level = len(s) - len(s.lstrip('#'))
            body = s[level:].strip()
            numbered = True
            if body.endswith('{-}'):
                numbered, body = False, body[:-3].strip()
            blocks.append(dict(t='h', level=level, text=body, numbered=numbered))
            i += 1
        elif s.startswith('!['):
            m = re.fullmatch(r'!\[(.*)\]\((.+?)\)(?:\{(.*)\})?', s)
            if not m:
                raise ValueError('bad figure line: ' + s)
            attrs = m.group(3) or ''
            label = re.search(r'#(fig:[\w.-]+)', attrs)
            width = re.search(r'width=(\S+)', attrs)
            blocks.append(dict(t='fig', caption=m.group(1), path=m.group(2), label=label.group(1) if label else None,
                               width=parse_length(width.group(1)) if width else 15.0))
            i += 1
        elif s.startswith('Table:'):
            cap = s[6:].strip()
            lm = LABEL.search(cap)
            label = None
            if lm and lm.group(1).startswith('#tab:'):
                label, cap = lm.group(1)[1:], cap[:lm.start()].strip()
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            rows = []
            while i < n and lines[i].strip().startswith('|'):
                rows.append(lines[i])
                i += 1
            if len(rows) < 2:
                raise ValueError('table needs header and separator: ' + cap)
            header = split_row(rows[0])
            aligns = []
            for c in split_row(rows[1]):
                aligns.append('center' if c.startswith(':') and c.endswith(':') else 'right' if c.endswith(':') else 'left')
            body = [split_row(r) for r in rows[2:]]
            for r in body:
                if len(r) != len(header):
                    raise ValueError('ragged table row in: ' + cap)
            blocks.append(dict(t='tab', caption=cap, label=label, header=header, rows=body, aligns=aligns))
        elif s.startswith('$$'):
            buf = [s]
            while not re.search(r'\$\$(\s*\{[^{}]*\})?\s*$', buf[-1]) or (len(buf) == 1 and s == '$$'):
                i += 1
                if i >= n:
                    raise ValueError('unterminated display equation')
                buf.append(lines[i].strip())
            i += 1
            joined = ' '.join(buf)
            lm = LABEL.search(joined)
            label, numbered = None, True
            if lm:
                joined = joined[:lm.start()]
                if lm.group(1) == '-':
                    numbered = False
                elif lm.group(1).startswith('#eq:'):
                    label = lm.group(1)[1:]
            blocks.append(dict(t='eq', latex=joined.strip()[2:-2].strip(), label=label, numbered=numbered))
        elif LIST_ITEM.match(s):
            items = []
            while i < n and lines[i].strip():
                cur = lines[i].strip()
                mm = LIST_ITEM.match(cur)
                if mm:
                    items.append([mm.group(0).strip(), cur[mm.end():]])
                elif items:
                    items[-1][1] += ' ' + cur
                else:
                    break
                i += 1
            blocks.append(dict(t='list', items=items))
        else:
            buf = []
            while i < n and lines[i].strip() and (not buf or not starts_block(lines[i])):
                buf.append(lines[i].strip())
                i += 1
            blocks.append(dict(t='p', text=' '.join(buf)))
    return blocks


def block_texts(blocks):
    for b in blocks:
        t = b['t']
        if t == 'p':
            yield b['text']
        elif t == 'h':
            yield b['text']
        elif t == 'fig':
            yield b['caption']
        elif t == 'tab':
            yield b['caption']
            for c in b['header']:
                yield c
            for r in b['rows']:
                for c in r:
                    yield c
        elif t == 'list':
            for _, x in b['items']:
                yield x
        elif t == 'eq':
            yield '$' + b['latex'] + '$'


def run_pandoc(args, text, cwd=None):
    p = subprocess.run(['pandoc'] + args, input=text.encode('utf8'), capture_output=True, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode('utf8'))
    return p.stdout, p.stderr.decode('utf8')


def latex_to_omml(snippets):
    if not snippets:
        return {}
    md = '\n\n'.join('$' + s + '$' for s in snippets) + '\n'
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'math.docx'
        _, err = run_pandoc(['-f', 'markdown', '-o', str(out)], md)
        if 'Could not convert' in err:
            raise RuntimeError(err)
        with zipfile.ZipFile(out) as z:
            root = etree.fromstring(z.read('word/document.xml'))
    paras = root.find('w:body', NS).findall('w:p', NS)
    if len(paras) != len(snippets):
        raise RuntimeError('math conversion mismatch: %d vs %d' % (len(paras), len(snippets)))
    result = {}
    for s, p in zip(snippets, paras):
        om = p.find('m:oMath', NS)
        if om is None:
            raise RuntimeError('no math produced for: ' + s)
        result[s] = om
    return result


def style_math(om, sz=20):
    for r in om.iter(q('m:r')):
        if r.find('w:rPr', NS) is not None:
            continue
        wr = el('w:rPr', None, el('w:rFonts', {'w:ascii': 'Cambria Math', 'w:hAnsi': 'Cambria Math'}),
                val('w:sz', sz), val('w:szCs', sz))
        mp = r.find('m:rPr', NS)
        r.insert(1 if mp is not None else 0, wr)
    return om


def resolve_citations(order, bib, csl):
    if not order:
        return {}, {}
    md = '\n\n'.join('[@%s]' % k for k in order) + '\n'
    out, err = run_pandoc(['-f', 'markdown', '--citeproc', '--bibliography', str(bib), '--csl', str(csl), '-t', 'json'], md)
    missing = re.findall(r'citation (\S+) not found', err)
    if missing:
        raise RuntimeError('missing bibliography keys: ' + ', '.join(missing))
    doc = json.loads(out)
    entries = {}
    for blk in doc['blocks']:
        if blk['t'] == 'Div' and 'refs' == blk['c'][0][0]:
            for d in blk['c'][1]:
                entries[d['c'][0][0][4:]] = d['c'][1][0]['c']
    numbers = {k: i + 1 for i, k in enumerate(order)}
    for k in order:
        if k not in entries:
            raise RuntimeError('missing bibliography key: ' + k)
    return numbers, entries


def ast_runs(nodes, fmt):
    out = []
    for n in nodes:
        t = n['t']
        if t == 'Str':
            if n['c']:
                out.append(run(n['c'], **fmt))
        elif t in ('Space', 'SoftBreak', 'LineBreak'):
            out.append(run(' ', **fmt))
        elif t == 'Emph':
            out += ast_runs(n['c'], dict(fmt, i=True))
        elif t == 'Strong':
            out += ast_runs(n['c'], dict(fmt, b=True))
        elif t in ('Span', 'SmallCaps', 'Underline'):
            out += ast_runs(n['c'][1] if t == 'Span' else n['c'], fmt)
        elif t == 'Link':
            out += ast_runs(n['c'][1], fmt)
        elif t == 'Quoted':
            lo, hi = ('“', '”') if n['c'][0]['t'] == 'DoubleQuote' else ('‘', '’')
            out += [run(lo, **fmt)] + ast_runs(n['c'][1], fmt) + [run(hi, **fmt)]
        elif t == 'Superscript':
            out += ast_runs(n['c'], dict(fmt, sup=True))
        elif t == 'Subscript':
            out += ast_runs(n['c'], dict(fmt, sub=True))
        elif t == 'Code':
            out.append(run(n['c'][1], **fmt))
    return out


def strip_label(nodes):
    nodes = list(nodes)
    for n in nodes:
        if n['t'] == 'Span' and 'csl-right-inline' in n['c'][0][1]:
            return n['c'][1]
    return nodes


class Builder:
    def __init__(self, source, out, template, csl):
        self.source, self.out, self.template, self.csl = Path(source), Path(out), Path(template), Path(csl)
        self.base = self.source.parent
        self.images = []
        self.docpr = 0
        with zipfile.ZipFile(self.template) as z:
            self.tdoc = etree.fromstring(z.read('word/document.xml'))
        body = self.tdoc.find('w:body', NS)
        self.tparas = [e for e in body]
        self.sect = copy.deepcopy(body.find('w:sectPr', NS))
        tbl = body.find('w:tbl', NS)
        self.tbl_pr = copy.deepcopy(tbl.find('w:tblPr', NS))
        rows = tbl.findall('w:tr', NS)
        self.hdr_ppr = copy.deepcopy(rows[0].find('w:tc', NS).find('w:p', NS).find('w:pPr', NS))
        self.cell_ppr = copy.deepcopy(rows[1].find('w:tc', NS).find('w:p', NS).find('w:pPr', NS))

    def ppr(self, key, mark, keep=False, jc=None, ind=None, spacing=None, proto=None):
        src = proto if proto is not None else self.tparas[PROTO[key]].find('w:pPr', NS)
        ppr = copy.deepcopy(src)
        for c in ppr.findall('w:rPr', NS) + ppr.findall('w:pStyle', NS):
            ppr.remove(c)
        if keep:
            for c in ppr.findall('w:keepNext', NS):
                ppr.remove(c)
            ppr.insert(0, val('w:keepNext'))
        if jc:
            for c in ppr.findall('w:jc', NS):
                ppr.remove(c)
            ppr.append(val('w:jc', jc))
        if ind:
            for c in ppr.findall('w:ind', NS):
                ppr.remove(c)
            jcs = ppr.find('w:jc', NS)
            pos = list(ppr).index(jcs) if jcs is not None else len(ppr)
            ppr.insert(pos, el('w:ind', ind))
        if spacing:
            for c in ppr.findall('w:spacing', NS):
                ppr.remove(c)
            jcs = ppr.find('w:jc', NS)
            pos = list(ppr).index(jcs) if jcs is not None else len(ppr)
            ppr.insert(pos, el('w:spacing', spacing))
        ppr.append(rpr(**mark))
        return ppr

    def para(self, key, content, mark, **kw):
        p = el('w:p')
        p.append(self.ppr(key, mark, **kw))
        p.extend(content)
        return p

    def blank(self, keep=False):
        return self.para('body', [], dict(sz=20), keep=keep)

    def prepare(self, meta, blocks):
        texts = ([meta.get('abstract', '')] + list(block_texts(blocks)) + [meta.get('biography', '')])
        order, math = [], []
        for t in texts:
            for m in re.finditer(r'\[@([^\]]+)\]', t):
                for k in re.findall(r'@([\w:.-]+)', '[@' + m.group(1) + ']'):
                    if k not in order:
                        order.append(k)
            for m in INLINE.finditer(t):
                if m.group('math') and m.group('mt') not in math:
                    math.append(m.group('mt'))
        self.omml = latex_to_omml(math)
        bib = meta.get('bibliography')
        if order and not bib:
            raise RuntimeError('citations present but no bibliography in front matter')
        self.cite_order = order
        self.cite_num, self.cite_entries = resolve_citations(order, (self.base / bib) if bib else None, self.csl)
        self.nums = {}
        cnt = dict(fig=0, tab=0, eq=0)
        for b in blocks:
            kind = {'fig': 'fig', 'tab': 'tab', 'eq': 'eq'}.get(b['t'])
            if kind is None:
                continue
            if b['t'] == 'eq' and not b['numbered']:
                continue
            cnt[kind] += 1
            b['num'] = cnt[kind]
            if b.get('label'):
                self.nums[b['label']] = cnt[kind]

    def inline(self, text, fmt):
        return self._inline(smart_quotes(text), fmt)

    def _inline(self, text, fmt):
        out, pos = [], 0

        def plain(s):
            if s:
                out.append(run(s, **fmt))
        for m in INLINE.finditer(text):
            plain(text[pos:m.start()])
            pos = m.end()
            if m.group('esc'):
                plain(m.group('esc')[1])
            elif m.group('math'):
                out.append(style_math(copy.deepcopy(self.omml[m.group('mt')]), fmt.get('sz', 20)))
            elif m.group('cite'):
                keys = re.findall(r'@([\w:.-]+)', m.group('cite'))
                plain(', '.join('[%d]' % n for n in sorted({self.cite_num[k] for k in keys})))
            elif m.group('ref'):
                kind, name = m.group('ref')[2:-2].split(':', 1)
                label = kind + ':' + name
                if label not in self.nums:
                    raise RuntimeError('unresolved reference: ' + label)
                n = self.nums[label]
                plain({'fig': 'Fig. %d', 'tab': 'Table %d', 'eq': '(%d)'}[kind] % n)
            elif m.group('bold'):
                out += self._inline(m.group('bt'), dict(fmt, b=True))
            elif m.group('ital'):
                out += self._inline(m.group('it'), dict(fmt, i=True))
            elif m.group('sup'):
                out += self._inline(m.group('st'), dict(fmt, sup=True))
            elif m.group('sub'):
                out += self._inline(m.group('bs'), dict(fmt, sub=True))
        plain(text[pos:])
        return out

    def front(self, meta):
        out = []
        out.append(self.para('title', [run(meta['title'].upper(), b=True, sz=24, color='000000')], dict(b=True, sz=24, color='000000')))
        out += [self.blank(), self.blank()]
        authors = meta['authors']
        aff = meta['affiliations']
        content = []
        fmt = dict(b=True, i=True, sz=22)
        for k, a in enumerate(authors):
            sep = ', ' if k and k < len(authors) - 1 else (' and ' if len(authors) == 2 and k == 1 else ', and ' if k else '')
            content.append(run(sep + a['name'], **fmt))
            sup = ','.join(str(x) for x in a.get('affiliations', []))
            if a.get('corresponding'):
                sup += '*'
            if sup:
                content.append(run(sup, **dict(fmt, sup=True)))
        out.append(self.para('authors', content, fmt))
        out.append(self.blank())
        for k, a in enumerate(aff, 1):
            out.append(self.para('aff_name', [run(str(k), sz=20, sup=True), run(a['name'], sz=20)], dict(sz=20)))
            if a.get('address'):
                out.append(self.para('aff_addr', [run(a['address'], sz=20)], dict(sz=20)))
            out.append(self.blank())
        mails = [('*' if a.get('corresponding') else ','.join(str(x) for x in a.get('affiliations', [])), a['email']) for a in authors if a.get('email')]
        content = [run('Email: ' if len(mails) == 1 else 'Emails: ', sz=20)]
        for j, (k, e) in enumerate(mails):
            content.append(run(('' if j == 0 else ', ') + e, sz=20))
            content.append(run(k, sz=20, sup=True))
        out.append(self.para('emails', content, dict(sz=20)))
        out += [self.blank(), self.blank()]
        out.append(self.para('received', [run('Received: \t\t\tRevised:\t\t\t\tAccepted:', i=True, sz=20)], dict(i=True, sz=20)))
        out += [self.blank(), self.blank()]
        out.append(self.para('abs_head', [run('ABSTRACT', b=True, i=True, sz=20)], dict(b=True, i=True, sz=20)))
        out.append(self.blank())
        out.append(self.para('abs_text', self.inline(' '.join(meta['abstract'].split()), dict(i=True, sz=20)), dict(sz=20)))
        out.append(self.blank())
        kw = '; '.join(meta['keywords']) + '.'
        out.append(self.para('keywords', [run('Keywords: ', b=True, i=True, sz=20), run(kw, b=True, i=True, sz=20)], dict(b=True, i=True, sz=20)))
        out += [self.blank(), self.blank()]
        return out

    def heading(self, b, counters):
        lvl = b['level']
        if not b['numbered']:
            return [self.para('ref_head', [run(b['text'].upper(), b=True, sz=20)], dict(b=True, sz=20), keep=True), self.blank(keep=True)]
        if lvl == 1:
            counters[:] = [counters[0] + 1, 0, 0]
            num, key, text = '%d.0' % counters[0], 'h1', b['text'].upper()
        elif lvl == 2:
            counters[1:] = [counters[1] + 1, 0]
            num, key, text = '%d.%d' % (counters[0], counters[1]), 'h2', b['text']
        else:
            counters[2] += 1
            num, key, text = '%d.%d.%d' % tuple(counters), 'h3', b['text']
        fmt = dict(b=True, sz=20)
        return [self.para(key, [run(num + '\t' + text, **fmt)], fmt, keep=True), self.blank(keep=True)]

    def image_run(self, b):
        path = (self.base / b['path']).resolve()
        ext = path.suffix.lower().lstrip('.')
        if ext not in ('png', 'jpg', 'jpeg'):
            raise RuntimeError('figure must be png or jpeg: ' + str(path))
        with Image.open(path) as im:
            w, h = im.size
        cm = min(b['width'], TEXT_WIDTH / 567)
        cx = int(cm * EMU_CM)
        cy = int(cx * h / w)
        self.images.append((path, 'jpeg' if ext in ('jpg', 'jpeg') else 'png'))
        k = len(self.images)
        self.docpr += 1
        rid = 'rIdFig%d' % k
        alt = escape(plain_text(b['caption']), {'"': '&quot;'})
        xml = (
            '<w:r xmlns:w="%(w)s" xmlns:wp="%(wp)s" xmlns:a="%(a)s" xmlns:pic="%(pic)s" xmlns:r="%(r)s">'
            '<w:rPr><w:rFonts w:ascii="%(f)s" w:hAnsi="%(f)s" w:cs="%(f)s" w:eastAsia="%(f)s"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
            '<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="%(cx)d" cy="%(cy)d"/>'
            '<wp:docPr id="%(id)d" name="Figure %(k)d" descr="%(alt)s"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="%(pic)s"><pic:pic><pic:nvPicPr><pic:cNvPr id="%(id)d" name="figure%(k)d.%(e)s"/><pic:cNvPicPr/></pic:nvPicPr>'
            '<pic:blipFill><a:blip r:embed="%(rid)s"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%(cx)d" cy="%(cy)d"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        ) % dict(NS, f=FONT, cx=cx, cy=cy, id=self.docpr, k=k, alt=alt, e='png' if ext == 'png' else 'jpg', rid=rid)
        return etree.fromstring(xml)

    def figure(self, b):
        pic = self.para('figcap', [self.image_run(b)], dict(sz=20), keep=True)
        cap = self.para('figcap', self.inline('Fig. %d: %s' % (b['num'], b['caption']), dict(sz=20)), dict(sz=20))
        cppr = cap.find('w:pPr', NS)
        if cppr is not None:
            cppr.insert(0, val('w:keepLines'))
        return [pic, cap, self.blank(), self.blank()]

    def col_widths(self, b):
        cols = len(b['header'])
        want, low = [], []
        for c in range(cols):
            cells = [b['header'][c]] + [r[c] for r in b['rows']]
            texts = [plain_text(x) for x in cells]
            want.append(min(max(len(t) for t in texts) * 100 + 220, TEXT_WIDTH))
            body_word = max(len(w) for t in texts[1:] for w in t.split() or ['']) if len(texts) > 1 else 0
            head_word = max([len(w) for w in texts[0].split()] or [0])
            low.append(min(max(body_word * 105, head_word * 120) + 220, TEXT_WIDTH))
        want = [max(w, l) for w, l in zip(want, low)]
        if sum(want) <= TEXT_WIDTH:
            return want
        spare = TEXT_WIDTH - sum(low)
        if spare <= 0:
            scale = TEXT_WIDTH / sum(low)
            return [int(x * scale) for x in low]
        extra = [w - l for w, l in zip(want, low)]
        tot = sum(extra) or 1
        widths = [l + int(spare * e / tot) for l, e in zip(low, extra)]
        widths[-1] += TEXT_WIDTH - sum(widths)
        return widths

    def cell(self, text, width, header, first, last, top_rule, align, keep=False):
        borders = el('w:tcBorders')
        line = {'w:color': '000000', 'w:space': 0, 'w:sz': 6, 'w:val': 'single'}
        nil = {'w:color': '000000', 'w:space': 0, 'w:sz': 0, 'w:val': 'nil'}
        if header:
            borders.append(el('w:top', line))
            if first:
                borders.append(el('w:left', line))
            borders.append(el('w:bottom', line))
            if last:
                borders.append(el('w:right', line))
        else:
            borders.append(el('w:top', nil))
            if first:
                borders.append(el('w:left', line))
            if last:
                borders.append(el('w:right', line))
        mar = el('w:tcMar', None, el('w:top', {'w:w': 0, 'w:type': 'dxa'}), el('w:bottom', {'w:w': 0, 'w:type': 'dxa'}))
        tcpr = el('w:tcPr', None, el('w:tcW', {'w:w': width, 'w:type': 'dxa'}), borders, mar, val('w:vAlign', 'center'))
        fmt = dict(b=True, sz=20) if header else dict(sz=20)
        proto = self.hdr_ppr if header else self.cell_ppr
        spacing = {'w:before': 60, 'w:after': 60, 'w:lineRule': 'auto'} if header else {'w:before': 30, 'w:after': 30, 'w:lineRule': 'auto'}
        p = self.para(None, self.inline(text, fmt), fmt, keep=keep, jc='center' if header else align, proto=proto, spacing=spacing)
        return el('w:tc', None, tcpr, p)

    def table(self, b):
        cap = self.para('tabcap', self.inline('Table %d: %s' % (b['num'], b['caption']), dict(sz=20)), dict(sz=20), keep=True)
        widths = self.col_widths(b)
        tbl = el('w:tbl')
        pr = copy.deepcopy(self.tbl_pr)
        pr.find('w:tblW', NS).set(q('w:w'), str(sum(widths)))
        tbl.append(pr)
        tbl.append(el('w:tblGrid', None, *[el('w:gridCol', {'w:w': w}) for w in widths]))
        cols = len(widths)
        nrows = len(b['rows']) + 1
        keep_rows = nrows <= 9
        for r, cells in enumerate([b['header']] + b['rows']):
            trpr = el('w:trPr', None, val('w:cantSplit'))
            if r == 0:
                trpr.append(val('w:tblHeader'))
            tr = el('w:tr', None, trpr)
            for c, text in enumerate(cells):
                align = {'left': 'left', 'center': 'center', 'right': 'right'}[b['aligns'][c]]
                tr.append(self.cell(text, widths[c], r == 0, c == 0, c == cols - 1, r == 0, align,
                                    keep=(r == 0) or (keep_rows and r < nrows - 1)))
            tbl.append(tr)
        return [cap, self.blank(keep=True), tbl, self.blank(), self.blank()]

    def equation(self, b):
        om = style_math(copy.deepcopy(self.omml[b['latex']]))
        mp = el('m:oMathPara', None, el('m:oMathParaPr', None, el('m:jc', {'m:val': 'center'})), om)
        mid = self.para('body', [mp], dict(sz=20), jc='center')
        num = '(%d)' % b['num'] if b['numbered'] else ''
        left = self.para('body', [], dict(sz=20), jc='left')
        right = self.para('body', [run(num, sz=20)] if num else [], dict(sz=20), jc='right')
        widths = [800, TEXT_WIDTH - 1600, 800]
        tbl = el('w:tbl')
        tbl.append(el('w:tblPr', None, el('w:tblW', {'w:w': TEXT_WIDTH, 'w:type': 'dxa'}), val('w:jc', 'center'),
                      el('w:tblLayout', {'w:type': 'fixed'}), val('w:tblDescription', 'equation layout')))
        tbl.append(el('w:tblGrid', None, *[el('w:gridCol', {'w:w': w}) for w in widths]))
        tr = el('w:tr', None, el('w:trPr', None, val('w:cantSplit')))
        for w, p in zip(widths, (left, mid, right)):
            tr.append(el('w:tc', None, el('w:tcPr', None, el('w:tcW', {'w:w': w, 'w:type': 'dxa'}), val('w:vAlign', 'center')), p))
        tbl.append(tr)
        return [tbl, self.blank()]

    def listing(self, b):
        out = []
        for k, (mark, text) in enumerate(b['items'], 1):
            lead = '•' if mark in ('-', '*') else '%d.' % k
            fmt = dict(sz=20)
            content = [run(lead + '\t', **fmt)] + self.inline(text, fmt)
            p = self.para('body', content, fmt, ind={'w:left': 567, 'w:hanging': 567})
            out.append(p)
        return out + [self.blank()]

    def references(self):
        out = [self.para('ref_head', [run('REFERENCES', b=True, sz=20)], dict(b=True, sz=20), keep=True), self.blank(keep=True)]
        for k in self.cite_order:
            fmt = dict(sz=20)
            runs = [run('[%d]\t' % self.cite_num[k], **fmt)] + ast_runs(strip_label(self.cite_entries[k]), fmt)
            out += [self.para('ref_entry', runs, fmt), self.blank()]
        return out

    def body(self, meta, blocks):
        out = self.front(meta)
        counters = [0, 0, 0]
        for b in blocks:
            t = b['t']
            if t == 'h':
                out += self.heading(b, counters)
            elif t == 'p':
                out += [self.para('body', self.inline(b['text'], dict(sz=20)), dict(sz=20)), self.blank()]
            elif t == 'fig':
                out += self.figure(b)
            elif t == 'tab':
                out += self.table(b)
            elif t == 'eq':
                out += self.equation(b)
            elif t == 'list':
                out += self.listing(b)
        out.append(self.blank())
        if self.cite_order:
            out += self.references()
        if meta.get('biography'):
            out.append(self.blank())
            out.append(self.para('ref_head', [run('BIOGRAPHY', b=True, sz=20)], dict(b=True, sz=20), keep=True))
            out.append(self.blank(keep=True))
            for chunk in re.split(r'\n\s*\n', meta['biography'].strip()):
                out += [self.para('body', self.inline(' '.join(chunk.split()), dict(sz=20)), dict(sz=20)), self.blank()]
        return out

    def core(self, meta, stamp):
        kw = '; '.join(meta['keywords'])
        who = escape(meta.get('creator', CREATOR))
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                '<dc:title>%s</dc:title><dc:creator>%s</dc:creator><cp:keywords>%s</cp:keywords>'
                '<cp:lastModifiedBy>%s</cp:lastModifiedBy>'
                '<dcterms:created xsi:type="dcterms:W3CDTF">%s</dcterms:created>'
                '<dcterms:modified xsi:type="dcterms:W3CDTF">%s</dcterms:modified></cp:coreProperties>'
                % (escape(meta['title']), who, escape(kw), who, stamp, stamp))

    def build(self):
        raw = self.source.read_text(encoding='utf8')
        meta, rest = split_front(raw)
        blocks = parse_blocks(rest)
        self.prepare(meta, blocks)
        content = self.body(meta, blocks)
        root = copy.deepcopy(self.tdoc)
        body = root.find('w:body', NS)
        for c in list(body):
            body.remove(c)
        body.extend(content)
        body.append(copy.deepcopy(self.sect))
        doc_xml = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        day = str(meta.get('date') or datetime.datetime.fromtimestamp(self.source.stat().st_mtime, datetime.timezone.utc).date())
        stamp = day + 'T00:00:00Z'
        y, mo, d = (int(x) for x in day.split('-'))
        self.write_package(doc_xml, self.core(meta, stamp), (max(y, 1980), mo, d, 0, 0, 0))

    def write_package(self, doc_xml, core_xml, when):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        drop = ('customXML/', 'docProps/custom.xml', 'word/media/')
        with zipfile.ZipFile(self.template) as zin, zipfile.ZipFile(self.out, 'w', zipfile.ZIP_DEFLATED) as zout:
            def put(name, data):
                zi = zipfile.ZipInfo(name, when)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = 0o644 << 16
                zout.writestr(zi, data)
            for name in zin.namelist():
                if name.startswith(drop):
                    continue
                data = zin.read(name)
                if name == '[Content_Types].xml':
                    data = self.content_types(data.decode('utf8')).encode('utf8')
                elif name == '_rels/.rels':
                    data = re.sub(r'<Relationship [^>]*custom-properties[^>]*/>', '', data.decode('utf8')).encode('utf8')
                elif name == 'word/_rels/document.xml.rels':
                    data = self.doc_rels(data.decode('utf8')).encode('utf8')
                elif name == 'word/document.xml':
                    data = doc_xml
                elif name == 'docProps/core.xml':
                    data = core_xml.encode('utf8')
                put(name, data)
            for k, (path, kind) in enumerate(self.images, 1):
                put('word/media/figure%d.%s' % (k, 'png' if kind == 'png' else 'jpg'), path.read_bytes())

    def content_types(self, xml):
        xml = re.sub(r'<Override [^>]*PartName="/(?:customXML/[^"]*|docProps/custom\.xml)"[^>]*/>', '', xml)
        xml = re.sub(r'<Override [^>]*PartName="/customXML/[^"]*"[^>]*/>', '', xml)
        kinds = {k for _, k in self.images}
        have = set(re.findall(r'Extension="(\w+)"', xml))
        add = ''
        if 'png' in kinds and 'png' not in have:
            add += '<Default ContentType="image/png" Extension="png"/>'
        if 'jpeg' in kinds:
            add += '<Default ContentType="image/jpeg" Extension="jpg"/>'
        return xml.replace('</Types>', add + '</Types>')

    def doc_rels(self, xml):
        xml = re.sub(r'<Relationship [^>]*customXml[^>]*/>', '', xml)
        xml = re.sub(r'<Relationship [^>]*relationships/image[^>]*/>', '', xml)
        add = ''
        for k, (_, kind) in enumerate(self.images, 1):
            add += ('<Relationship Id="rIdFig%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                    'Target="media/figure%d.%s"/>' % (k, k, 'png' if kind == 'png' else 'jpg'))
        return xml.replace('</Relationships>', add + '</Relationships>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default=str(ROOT / 'paper' / 'manuscript.md'))
    ap.add_argument('--out', default=str(ROOT / 'paper' / 'manuscript.docx'))
    ap.add_argument('--template', default=str(ROOT / 'paper' / 'mjcs_template.docx'))
    ap.add_argument('--csl', default=str(ROOT / 'paper' / 'ieee.csl'))
    a = ap.parse_args()
    try:
        Builder(a.source, a.out, a.template, a.csl).build()
    except (RuntimeError, ValueError, KeyError) as e:
        sys.exit('build failed: %s' % e)
    print(a.out)


if __name__ == '__main__':
    main()
