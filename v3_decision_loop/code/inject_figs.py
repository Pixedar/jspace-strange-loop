"""Inline result figures into the report page: every <!--FIG:name.png|alt text|caption--> placeholder in the template
becomes a <figure> with the PNG as a data URI (caption optional).
    python inject_figs.py template.html out.html figs_dir"""
import sys, re, base64, pathlib, html as H

src, dst, figs = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
page = pathlib.Path(src).read_text(encoding='utf-8')


def sub(m):
    name, alt, cap = (m.group(1).split('|', 2)+['', ''])[:3]
    p = figs/name.strip()
    if not p.exists():return f'<p class="missing">Figure {H.escape(name.strip())} not produced.</p>'
    b = base64.b64encode(p.read_bytes()).decode()
    fc = f'<figcaption>{cap.strip()}</figcaption>' if cap.strip() else ''
    return f'<figure class="plate"><img src="data:image/png;base64,{b}" alt="{H.escape(alt.strip())}" loading="lazy">{fc}</figure>'


page = re.sub(r'<!--FIG:(.*?)-->', sub, page, flags=re.S)
pathlib.Path(dst).write_text(page, encoding='utf-8');print('wrote', dst, len(page)//1024, 'KB')
