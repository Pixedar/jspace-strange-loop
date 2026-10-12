"""Build the report page (HTML artifact) from REPORT_V3.md: custom header and findings ledger, the markdown body with
the five-question lead-ins styled, tables wrapped for horizontal scroll, figures inlined by inject_figs.py.
    python build_report.py            (from research_v3/)"""
import re, pathlib, subprocess, sys, html as H
import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent
md = (ROOT/'REPORT_V3.md').read_text(encoding='utf-8')
head = (ROOT/'report_template_head.html').read_text(encoding='utf-8')
LEDGER = ROOT/'ledger.html'                   # hand-written findings ledger (chips + one-line verdicts)

body_md = md[md.index('## 1. What the program asks'):]
# Python-Markdown needs a blank line before a list and before a table that follows a paragraph line
lines, prev = [], ''
for ln in body_md.split('\n'):
    is_item = bool(re.match(r'^(\* |\d+\. )', ln));is_tab = ln.startswith('|')
    prev_item = bool(re.match(r'^(\* |\d+\. |  \S)', prev));prev_tab = prev.startswith('|')
    if (is_item and prev.strip() and not prev_item) or (is_tab and prev.strip() and not prev_tab):lines.append('')
    lines.append(ln);prev = ln
body_md = '\n'.join(lines).replace('R^2', 'R²')
body = markdown.markdown(body_md, extensions=['tables', 'sane_lists'])
def tbl(m):                                     # tables with long text cells wrap and align left
    t = m.group(0);cells = [re.sub('<.*?>', '', c) for c in re.findall(r'<td[^>]*>(.*?)</td>', t, flags=re.S)]
    cls = ' class="text"' if cells and max(len(c) for c in cells) > 30 else ''
    return f'<div class="tbl"><table{cls}>'+t[len('<table>'):]+'</div>'


body = re.sub(r'<table>.*?</table>', tbl, body, flags=re.S)
for q in ('What changed physically', 'What happened numerically', 'What happened conceptually', 'Ordinary explanations tested',
          'Ordinary explanation tested', 'What remains unestablished'):
    body = re.sub(rf'<p><strong>{q}\.</strong>\s*', f'<p class="lead"><span class="q">{q}</span>', body)
# section ids for the contents list
toc = []


def sec(m):
    title = re.sub(r'^\d+\.\s*', '', re.sub('<.*?>', '', m.group(1)));sid = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:48]
    toc.append((sid, title));return f'<h2 id="{sid}">{m.group(1)}</h2>'


body = re.sub(r'<h2>(.*?)</h2>', sec, body)
extra_css = """<style>
p.lead{margin-top:1.1rem} .q{display:block;font-family:var(--f-display);font-weight:650;font-size:.95rem;color:var(--accent);margin-bottom:.15rem}
nav.toc{margin:1.6rem 0 0;padding:.9rem 1rem;background:var(--surface);border:1px solid var(--rule);border-radius:6px}
nav.toc ol{margin:.3rem 0 0;padding-left:1.2rem;columns:2 16rem;column-gap:2rem;font-size:.9rem} nav.toc li{margin:.15rem 0;break-inside:avoid}
nav.toc .eyebrow{display:block}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.6rem;margin:1.2rem 0}
@media (max-width:640px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
.kpi{background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:.6rem .75rem;min-width:0}
.kpi b{display:block;font-family:var(--f-display);font-size:1.45rem;font-weight:720;font-stretch:85%;color:var(--ink);font-variant-numeric:tabular-nums}
.kpi span{font-size:.8rem;color:var(--muted);line-height:1.35;display:block}
table.text th,table.text td{white-space:normal;text-align:left;vertical-align:top;min-width:7rem}
table.text td:last-child,table.text th:last-child{text-align:right;min-width:4rem}
</style>"""
ledger = LEDGER.read_text(encoding='utf-8') if LEDGER.exists() else ''
toc_html = '<nav class="toc" aria-label="Contents"><span class="eyebrow">Contents</span><ol>'+''.join(
    f'<li><a href="#{i}">{H.escape(t)}</a></li>' for i, t in toc)+'</ol></nav>'
page = head+extra_css+f"""
<div class="wrap">
<header class="top">
  <div class="eyebrow">Research v3 · Qwen3-8B / 14B · Huginn-0125 · Ouro-2.6B · October 2026</div>
  <h1>Does the J-space workspace close a strange loop?</h1>
  <p class="lede">The plan was to treat Qwen3's J-space workspace as the system's bottleneck: find out what it carries, feed it
  back into the model as a loop, and see whether a self-model coupled to that loop produces organised, non-trivial dynamics.
  This page reports what the frozen protocol measured on Qwen3-8B, a Qwen3-14B replication and two trained
  recurrent-depth models.</p>
  <p class="meta">Overnight run, 11–12 Oct 2026 · frozen protocol PROTOCOL_V3.md (amendments dated) · raw arrays in
  the private HF dataset Pixedar/jspace-v3</p>
</header>
{ledger}
{toc_html}
{body}
<footer>Built from research_v3/REPORT_V3.md. Numbers are from the compact result files in research_v3/results/; every
experiment was run once at the stated scale with the stated seeds. Exploratory arms (X, strong coupling, epistemic
steering, label control) were added after the protocol was frozen and are labelled as such.</footer>
</div>
"""
pre = ROOT/'report_v3_pre.html';pre.write_text(page, encoding='utf-8')
figs = sys.argv[1] if len(sys.argv) > 1 else str(ROOT/'results'/'figs')
subprocess.run([sys.executable, str(ROOT/'code'/'inject_figs.py'), str(pre), str(ROOT/'report_v3.html'), figs], check=True)
