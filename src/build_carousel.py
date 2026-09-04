"""Build a 6-slide LinkedIn carousel (4:5 portrait) as self-contained HTML.

    python -m src.build_carousel
    # then: Edge --headless --print-to-pdf on article/carousel.html

Design: same "quant working-paper" identity as the article — cool-slate paper,
garnet accent, serif headlines, mono labels — with dark title/takeaway slides
bracketing light data slides for scroll-stopping rhythm.
"""
from __future__ import annotations

import base64

from . import config as C

OUT = C.ROOT / "article" / "carousel.html"
FOOTER_NAME = "Bruno Karoubi"          # <-- edit to taste before exporting


def fig(name: str) -> str:
    p = C.FIG_DIR / name
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


CSS = """
<style>
  @page { size: 1080px 1350px; margin: 0; }
  *{box-sizing:border-box;margin:0;padding:0;}
  :root{
    --paper:#f3f4f6; --ink:#181f29; --muted:#5c6672; --accent:#a12c3f;
    --card:#ffffff; --rule:#d9dee5;
    --dark:#14181f; --dark-ink:#eef0f3; --dark-muted:#9aa4b2; --dark-accent:#e8697c;
    --serif:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    --mono:"SF Mono",ui-monospace,"Cascadia Code",Consolas,Menlo,monospace;
  }
  body{font-family:var(--serif);}
  .slide{width:1080px;height:1350px;overflow:hidden;position:relative;
         padding:96px 88px;display:flex;flex-direction:column;
         page-break-after:always;}
  .slide:last-child{page-break-after:auto;}
  .light{background:var(--paper);color:var(--ink);}
  .dark{background:var(--dark);color:var(--dark-ink);}

  .eyebrow{font-family:var(--mono);font-size:24px;letter-spacing:.22em;
           text-transform:uppercase;color:var(--accent);margin-bottom:40px;}
  .dark .eyebrow{color:var(--dark-accent);}

  h1{font-size:82px;line-height:1.04;font-weight:700;letter-spacing:-.02em;
     text-wrap:balance;}
  h2{font-size:64px;line-height:1.08;font-weight:700;letter-spacing:-.015em;
     text-wrap:balance;}
  .sub{font-size:34px;line-height:1.4;color:var(--muted);margin-top:36px;
       font-style:italic;max-width:20ch;}
  .dark .sub{color:var(--dark-muted);}

  .figwrap{background:var(--card);border:1px solid var(--rule);border-radius:16px;
           padding:22px;margin:44px 0;display:flex;align-items:center;
           justify-content:center;flex:1;min-height:0;}
  .figwrap img{max-width:100%;max-height:100%;object-fit:contain;}

  .caption{font-size:33px;line-height:1.45;font-weight:400;}
  .caption b{color:var(--accent);font-weight:700;}
  .dark .caption b{color:var(--dark-accent);}

  .spacer{flex:1;}
  .bignum{font-size:150px;line-height:1;font-weight:700;color:var(--accent);
          letter-spacing:-.03em;font-variant-numeric:tabular-nums;margin:8px 0;}
  .statrow{display:flex;flex-direction:column;gap:52px;margin-top:56px;}
  .stat .k{font-family:var(--mono);font-size:26px;letter-spacing:.04em;
           color:var(--muted);text-transform:uppercase;}
  .light .stat .k{color:var(--muted);}
  .stat .v{font-size:40px;line-height:1.25;margin-top:10px;}
  .stat .v b{color:var(--accent);}

  ul.take{list-style:none;display:flex;flex-direction:column;gap:44px;margin-top:20px;}
  ul.take li{font-size:40px;line-height:1.3;padding-left:52px;position:relative;}
  ul.take li:before{content:"";position:absolute;left:0;top:18px;width:26px;height:5px;
        background:var(--dark-accent);border-radius:3px;}

  .footer{font-family:var(--mono);font-size:24px;letter-spacing:.06em;
          color:var(--muted);display:flex;justify-content:space-between;
          align-items:center;padding-top:36px;border-top:1px solid var(--rule);}
  .dark .footer{color:var(--dark-muted);border-top-color:#2a313b;}
  .cta{font-size:38px;line-height:1.35;margin-top:36px;}
  .cta b{color:var(--dark-accent);}
  .swipe{font-family:var(--mono);font-size:26px;letter-spacing:.1em;
         color:var(--dark-accent);margin-top:auto;}
</style>
"""


def slides() -> str:
    name = FOOTER_NAME
    S = []

    # 1 — title (dark)
    S.append(f"""
    <section class="slide dark">
      <p class="eyebrow">Causal ML &middot; Uplift Modeling</p>
      <h1>The best model wasn't the fanciest one.</h1>
      <p class="sub">Benchmarking six causal-ML estimators on 14&nbsp;million randomized ad exposures.</p>
      <p class="swipe">swipe &rarr;</p>
      <div class="footer"><span>Criteo uplift benchmark</span><span>{name}</span></div>
    </section>""")

    # 2 — targeting (light, figure)
    S.append(f"""
    <section class="slide light">
      <p class="eyebrow">01 &middot; Targeting</p>
      <h2>You don't need to treat everyone.</h2>
      <div class="figwrap"><img src="{fig('qini_curves_full_visit.png')}"/></div>
      <p class="caption">The top <b>10%</b> of users by predicted uplift carry <b>~82%</b>
      of the incremental visits. Half the audience is dead weight — and invisible if
      you only look at the average effect.</p>
    </section>""")

    # 3 — disagreement (light, big stats, no figure)
    S.append("""
    <section class="slide light">
      <p class="eyebrow">02 &middot; Estimator choice</p>
      <h2>Same data. Six methods. Different answers.</h2>
      <div class="statrow">
        <div class="stat"><div class="k">Spread in targeting quality (Qini, @500k)</div>
          <div class="bignum">2.6&times;</div>
          <div class="v">between the best and worst estimator</div></div>
        <div class="stat"><div class="k">Users flagged as actively "harmed"</div>
          <div class="v"><b>4%</b> to <b>44%</b> &mdash; on identical data, depending only
          on which estimator you picked.</div></div>
      </div>
      <div class="spacer"></div>
      <div class="footer"><span>S / T / X-learner &middot; DR &middot; R-learner &middot; causal forest</span><span></span></div>
    </section>""")

    # 4 — the twist / scaling (light, figure)
    S.append(f"""
    <section class="slide light">
      <p class="eyebrow">03 &middot; The twist</p>
      <h2>The disagreement was a sample-size illusion.</h2>
      <div class="figwrap"><img src="{fig('scaling_qini_visit.png')}"/></div>
      <p class="caption">From 500k to 4M rows the estimators converge — the gap between
      best and worst shrinks <b>0.43 &rarr; 0.12</b>. Estimator choice matters most
      exactly when you have the <b>least</b> data.</p>
    </section>""")

    # 5 — placebo (light, figure)
    S.append(f"""
    <section class="slide light">
      <p class="eyebrow">04 &middot; The rigor check</p>
      <h2>Shuffle the treatment. A real model finds nothing.</h2>
      <div class="figwrap"><img src="{fig('placebo_full_visit.png')}"/></div>
      <p class="caption">The DR-learner vs. its own placebo null: <b>z = 1.4</b> at 500k
      (fails) &rarr; <b>z = 7.0</b> at 4M (passes). It was never overfitting —
      it was <b>under-powered</b>.</p>
    </section>""")

    # 6 — takeaways + CTA (dark)
    S.append(f"""
    <section class="slide dark">
      <p class="eyebrow">Takeaways</p>
      <ul class="take">
        <li>Targeting, not averages, is the business case.</li>
        <li>The regularized methods (S-learner, causal forest) were robust at every scale.</li>
        <li>The placebo test is the discipline that tells overfitting from under-power.</li>
      </ul>
      <div class="spacer"></div>
      <p class="cta">Full write-up + reproducible code &rarr; <b>link in the comments.</b></p>
      <div class="footer"><span>14M-row Criteo uplift benchmark</span><span>{name}</span></div>
    </section>""")

    return "\n".join(S)


def main() -> None:
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>Uplift carousel</title>{CSS}</head><body>{slides()}</body></html>')
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
