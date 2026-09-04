"""Render article/uplift_article.md to self-contained HTML (images embedded).

Produces:
    article/uplift_article.html   full standalone doc (for PDF / browser)
    article/uplift_artifact.html  body-only fragment (for the Artifact tool)

    python -m src.build_html
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

from . import config as C

MD = C.ROOT / "article" / "uplift_article.md"
HTML_FULL = C.ROOT / "article" / "uplift_article.html"
HTML_FRAG = C.ROOT / "article" / "uplift_artifact.html"

EYEBROW = "Causal ML &middot; Uplift Modeling &middot; Criteo 14M randomized exposures"

STYLE = """
<style>
  :root{
    --bg:#f5f6f8; --surface:#ffffff; --ink:#182029; --muted:#5b6672;
    --rule:#dbe1e8; --hair:#e7ebf0; --accent:#a12c3f; --accent-ink:#7d1f2f;
    --note:#f6ecee; --thead:#eceff3; --code:#eef1f5;
    --serif:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
    --mono:"SF Mono",ui-monospace,"Cascadia Mono","Cascadia Code",Consolas,Menlo,monospace;
  }
  @media (prefers-color-scheme:dark){
    :root{
      --bg:#12151a; --surface:#171b21; --ink:#e7eaef; --muted:#98a2b0;
      --rule:#2a313b; --hair:#232932; --accent:#e8697c; --accent-ink:#f0899a;
      --note:#221a1d; --thead:#1b212a; --code:#1a1f27;}
  }
  :root[data-theme="light"]{
    --bg:#f5f6f8; --surface:#ffffff; --ink:#182029; --muted:#5b6672;
    --rule:#dbe1e8; --hair:#e7ebf0; --accent:#a12c3f; --accent-ink:#7d1f2f;
    --note:#f6ecee; --thead:#eceff3; --code:#eef1f5;}
  :root[data-theme="dark"]{
    --bg:#12151a; --surface:#171b21; --ink:#e7eaef; --muted:#98a2b0;
    --rule:#2a313b; --hair:#232932; --accent:#e8697c; --accent-ink:#f0899a;
    --note:#221a1d; --thead:#1b212a; --code:#1a1f27;}

  .wrap{max-width:720px;margin:0 auto;padding:8px 22px 88px;background:var(--bg);
        color:var(--ink);font-family:var(--serif);font-size:1.06rem;line-height:1.72;
        -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;}

  /* Masthead */
  .masthead{margin:14px 0 8px;padding-bottom:22px;border-bottom:2px solid var(--ink);}
  .eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.16em;
        text-transform:uppercase;color:var(--accent);margin:0 0 14px;}
  .masthead h1{font-size:2.15rem;line-height:1.14;margin:0 0 .35em;font-weight:700;
        letter-spacing:-.012em;text-wrap:balance;}
  .masthead .subtitle{font-size:1.06rem;line-height:1.5;color:var(--muted);
        font-style:italic;margin:0;text-wrap:pretty;}

  h2{font-size:1.34rem;font-weight:700;line-height:1.22;margin:2.4em 0 .55em;
        padding-top:.85em;border-top:1px solid var(--rule);letter-spacing:-.008em;
        text-wrap:balance;}
  h3{font-size:1.08rem;font-weight:700;margin:1.6em 0 .4em;}
  p{margin:.75em 0;}
  strong{font-weight:700;color:var(--ink);}
  em{color:inherit;}
  a{color:var(--accent);text-decoration:none;border-bottom:1px solid transparent;
        transition:border-color .15s;}
  a:hover{border-bottom-color:var(--accent);}
  a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px;}

  code{font-family:var(--mono);font-size:.82em;background:var(--code);
        padding:.1em .38em;border-radius:4px;}
  pre{background:var(--code);padding:14px 16px;border-radius:8px;overflow-x:auto;
        border:1px solid var(--hair);}
  pre code{background:none;padding:0;font-size:.8rem;}

  /* Note / callout cards from blockquotes */
  blockquote{margin:1.5em 0;padding:.2em 1.15em;background:var(--note);
        border-left:3px solid var(--accent);border-radius:0 8px 8px 0;}
  blockquote p{margin:.7em 0;font-size:.97rem;}
  blockquote strong:first-child{color:var(--accent-ink);}

  /* Figures */
  figure{margin:1.9em 0;display:flex;flex-direction:column;gap:.6em;}
  figure img{max-width:100%;height:auto;display:block;background:#fff;
        border:1px solid var(--rule);border-radius:8px;
        box-shadow:0 1px 3px rgba(20,30,45,.06);}
  figcaption{font-family:var(--mono);font-size:.72rem;line-height:1.5;color:var(--muted);
        letter-spacing:.02em;padding-left:2px;}
  figcaption b{color:var(--accent);font-weight:600;letter-spacing:.08em;
        text-transform:uppercase;margin-right:.5em;}

  /* Tables */
  .table-wrap{overflow-x:auto;margin:1.6em 0;border:1px solid var(--rule);
        border-radius:8px;}
  table{border-collapse:collapse;width:100%;font-family:var(--mono);
        font-size:.8rem;font-variant-numeric:tabular-nums;}
  thead th{background:var(--thead);text-align:left;font-weight:600;
        text-transform:uppercase;letter-spacing:.05em;font-size:.68rem;
        color:var(--muted);padding:.7em .85em;border-bottom:1px solid var(--rule);
        white-space:nowrap;}
  td{padding:.6em .85em;border-bottom:1px solid var(--hair);white-space:nowrap;
        color:var(--ink);}
  tbody tr:last-child td{border-bottom:none;}
  td strong{font-weight:700;color:var(--accent-ink);}
  td[align="right"],th[align="right"]{text-align:right;}
  td[align="center"],th[align="center"]{text-align:center;}

  hr{border:none;border-top:1px solid var(--rule);margin:2.6em 0;}
  ol,ul{padding-left:1.3em;}
  li{margin:.4em 0;}
  ol{counter-reset:none;}

  @media (max-width:520px){
    .wrap{padding:8px 16px 64px;font-size:1.02rem;}
    .masthead h1{font-size:1.7rem;}
  }
  @media (prefers-reduced-motion:reduce){*{transition:none!important;}}

  @media print{
    :root{--bg:#fff;--surface:#fff;}
    .wrap{max-width:100%;padding:0;font-size:11pt;}
    figure,.table-wrap,blockquote{break-inside:avoid;}
    h2,h3{break-after:avoid;}
    figure img{box-shadow:none;}
  }
</style>
"""


def wrap_figures(html: str) -> str:
    """Turn markdown's <p><img alt="cap" src=".."/></p> into numbered <figure>s."""
    counter = {"n": 0}

    def repl(m):
        counter["n"] += 1
        alt, src = m.group("alt"), m.group("src")
        return (f'<figure><img alt="{alt}" src="{src}"/>'
                f'<figcaption><b>Fig {counter["n"]}</b>{alt}</figcaption></figure>')

    return re.sub(r'<p><img alt="(?P<alt>[^"]*)" src="(?P<src>[^"]+)"\s*/?></p>',
                  repl, html)


def embed_images(html: str) -> str:
    def repl(m):
        src = m.group("src")
        p = (MD.parent / src).resolve()
        if not p.exists():
            return m.group(0)
        b64 = base64.b64encode(p.read_bytes()).decode()
        return m.group(0).replace(src, f"data:image/png;base64,{b64}")
    return re.sub(r'<img[^>]*src="(?P<src>[^"]+)"[^>]*>', repl, html)


def build_masthead(html: str) -> str:
    """Wrap the leading <h1> + italic subtitle paragraph into a masthead header."""
    m = re.search(r'<h1>(?P<title>.*?)</h1>\s*<p><em>(?P<sub>.*?)</em></p>',
                  html, flags=re.DOTALL)
    if not m:
        return html
    header = (f'<header class="masthead"><p class="eyebrow">{EYEBROW}</p>'
              f'<h1>{m.group("title")}</h1>'
              f'<p class="subtitle">{m.group("sub")}</p></header>')
    return html[:m.start()] + header + html[m.end():]


def main() -> None:
    md_text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    body = re.sub(r"(<table>.*?</table>)", r'<div class="table-wrap">\1</div>', body,
                  flags=re.DOTALL)
    body = wrap_figures(body)
    body = build_masthead(body)

    title = "From ATE to CATE — Criteo Uplift"

    # lightweight preview: images by URL (fast to render in a browser pane)
    preview_body = body.replace("../results/figures/", "/results/figures/")
    preview = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
               f'<meta name="viewport" content="width=device-width,initial-scale=1">'
               f'<title>{title}</title>{STYLE}</head><body>'
               f'<div class="wrap">\n{preview_body}\n</div></body></html>')
    (C.ROOT / "article" / "uplift_preview.html").write_text(preview, encoding="utf-8")

    body = embed_images(body)
    frag = f"<title>{title}</title>\n" + STYLE + f'<div class="wrap">\n{body}\n</div>'
    HTML_FRAG.write_text(frag, encoding="utf-8")

    full = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>From ATE to CATE — Criteo Uplift</title>{STYLE}</head>'
            f'<body>{frag}</body></html>')
    HTML_FULL.write_text(full, encoding="utf-8")

    # Root index.html for GitHub Pages (self-contained, same content as the full doc)
    (C.ROOT / "index.html").write_text(full, encoding="utf-8")

    print(f"wrote {HTML_FRAG.name} ({len(frag):,} bytes), {HTML_FULL.name} "
          f"({len(full):,} bytes), and index.html for GitHub Pages")


if __name__ == "__main__":
    main()
