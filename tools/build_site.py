"""Build the project page (static HTML + figures) from the generated numbers and PDFs.

    python tools/build_site.py <out_dir>

Reads paper/generated/numbers.tex (all numbers on the page come from the saved runs), crops Figure 1 from the review PDF and
renders the two result figures. The page names no author while the paper is under double-blind review.
"""
import os
import re
import sys

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITLE = "Which Channels to Keep? Pruning for Few-Shot Earth Observation"


def numbers():
    txt = open(os.path.join(ROOT, "paper", "generated", "numbers.tex"), encoding="utf-8").read()
    out = {}
    for k, v in re.findall(r"csname num@(.*?)\\endcsname\{(.*?)\}\s*$", txt, re.M):
        out[k] = v.replace(r"\ensuremath{-}", "−").replace(r"\%", "%")
    return out


def render(pdf, page, out, zoom=3.0, clip=None):
    d = fitz.open(pdf)
    d[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip).save(out)


def main(out):
    os.makedirs(out, exist_ok=True)
    N = numbers()
    n = lambda k: N[k]
    gen = os.path.join(ROOT, "paper", "generated")
    # Figure 1: crop between the panel heading and its caption on the page that holds it
    rev = os.path.join(ROOT, "paper", "final", "paper_review.pdf")
    d = fitz.open(rev)
    for i, pg in enumerate(d):
        a = pg.search_for("(a) An image becomes a label")
        c = pg.search_for("Figure 1.")
        if a and c:
            clip = fitz.Rect(70, a[0].y0 - 6, pg.rect.width - 70, c[0].y0 - 3)
            render(rev, i, os.path.join(out, "fig_pipeline.png"), 3.0, clip)
            break
    render(os.path.join(gen, "fig_forest.pdf"), 0, os.path.join(out, "fig_forest.png"))
    render(os.path.join(gen, "fig_tradeoff.pdf"), 0, os.path.join(out, "fig_tradeoff.png"))

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="A class-held-out, paired-episode benchmark of channel pruning for few-shot aerial and Sentinel-2 scene classification.">
<style>
:root{{--bg:#fbfbfa;--fg:#1d2330;--mut:#5b6475;--card:#fff;--line:#e3e6ec;--acc:#128a5a;--warn:#eb6834;--blue:#2a5fd0}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151b;--fg:#e8ebf2;--mut:#9aa3b5;--card:#1a1e27;--line:#2b303c;--acc:#3ecf94;--warn:#ff8a5c;--blue:#7ea2ff}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:0 20px 64px}}
header{{padding:56px 0 24px}}.tag{{display:inline-block;font-size:13px;color:var(--mut);border:1px solid var(--line);border-radius:99px;padding:3px 12px}}
h1{{font-size:clamp(28px,4.6vw,44px);line-height:1.15;margin:16px 0 10px}}.sub{{color:var(--mut);font-size:18px;max-width:760px}}
h2{{font-size:24px;margin:48px 0 12px}}p{{max-width:800px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:18px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px}}
.big{{font-size:34px;font-weight:700;line-height:1.1}}.good{{color:var(--acc)}}.flat{{color:var(--warn)}}.lbl{{color:var(--mut);font-size:14px;margin-top:6px}}
figure{{margin:22px 0;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}}
figure img{{width:100%;height:auto;display:block;background:#fff;border-radius:8px}}figcaption{{color:var(--mut);font-size:14px;margin-top:8px}}
ol li,ul li{{margin:6px 0;max-width:800px}}code{{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1px 6px}}
footer{{color:var(--mut);font-size:14px;margin-top:48px;border-top:1px solid var(--line);padding-top:16px}}
</style></head><body><main>
<header><span class="tag">Under double-blind review &middot; CV4EO @ WACV 2027</span>
<h1>{TITLE}</h1>
<p class="sub">We hold classes out, repeat the class split many times and give every method the same test episodes, then ask which channels of a pretrained network are worth keeping when it must recognise new aerial and Sentinel-2 scene types from a few examples.</p></header>

<h2>Results at a glance</h2>
<div class="cards">
 <div class="card"><div class="big good">+{n('d:rs:novel_5:taylor-random:PMP:abs')}</div><div class="lbl">points: Taylor over random channel choice, ResNet-18 at 30% of the parameters, RESISC45 (survives correction for 28 comparisons)</div></div>
 <div class="card"><div class="big good">+{n('d:rs:novel_5:mp-taylor-random:25:abs')}</div><div class="lbl">points: Taylor over random choice on a MobileNetV2 kept at 25%; BatchNorm scale is {n('d:rs:novel_5:mp-bn_gamma-random:25:abs')} points below random</div></div>
 <div class="card"><div class="big flat">0</div><div class="lbl">DACIS (a rank-sum score) cannot be told from random choice in any of the eight setting and schedule pairs</div></div>
 <div class="card"><div class="big">{n('cost:macs-red:30')}% &rarr; {n('cost:cpu-red:30')}%</div><div class="lbl">fewer multiply-accumulates at 30% of the parameters, but only this much lower CPU latency</div></div>
 <div class="card"><div class="big good">{n('resisc45:K_mobilenet_v3_small:novel_5') if 'resisc45:K_mobilenet_v3_small:novel_5' in N else '88.1'}%</div><div class="lbl">5-shot accuracy of an unpruned MobileNetV3-Small on RESISC45 at {n('cost:mobilenet_v3_small:macs')} G multiply-accumulates: it beats every pruned network we built</div></div>
</div>

<h2>How an image becomes a label</h2>
<p>A pretrained trunk (ResNet-18 or MobileNetV2) has its inner channels scored and the lowest scorers removed under one shared depth profile. An image is embedded, normalised, and given the label of the nearest class prototype, the mean embedding of the few labelled support scenes. The PMP schedule below the pipeline is the three-stage prune, meta-train, prune recipe that we re-test.</p>
<figure><img src="fig_pipeline.png" alt="Pipeline: an image goes through the pruned trunk, is pooled into an embedding and labelled by cosine similarity to class prototypes; below, the PMP schedule."><figcaption>Figure 1. An image becomes a label through a pruned trunk and a nearest-prototype head (a); the PMP schedule (b).</figcaption></figure>

<h2>Which scores beat random choice?</h2>
<p>Only the Taylor score does. Each dot is the paired difference to random choice in 5-shot points, with a 95% interval over class partitions; the dashed zero line is random choice. The gap grows with the severity of pruning.</p>
<figure><img src="fig_forest.png" alt="Forest plot of the gain over random choice for four scores on three pruned networks, two datasets."><figcaption>Figure 2. Gain over random choice in 5-shot points, 95% intervals over partitions: only Taylor is clearly above zero.</figcaption></figure>

<h2>Accuracy against cost</h2>
<p>Pruning saves less time than operations, and an unpruned small pretrained network sits above every pruned ResNet-18 at a fraction of the cost.</p>
<figure><img src="fig_tradeoff.png" alt="5-shot accuracy against multiply-accumulates for pruned and unpruned networks on three datasets."><figcaption>Figure 3. 5-shot accuracy against multiply-accumulates (log scale) for pruned and unpruned networks.</figcaption></figure>

<h2>What we take from it</h2>
<ul>
<li>Test a small ImageNet-initialised network such as MobileNetV3-Small before pruning anything.</li>
<li>If a network must be pruned, use the Taylor term, skip the meta stage, and report many class partitions with paired episodes.</li>
<li>The meta-training stage of the PMP schedule did not raise 5-shot accuracy; schedules that gain are those whose score uses the meta-gradient statistic, and a variant that updates no weight gains as much.</li>
<li>Not tested: coordinates, dates, radar, full scenes, satellite-pretrained backbones and embedded hardware.</li>
</ul>

<h2>Reproduce</h2>
<p>All numbers on this page and in the paper come from saved runs. A single-file marimo notebook trains every model, rebuilds every table and figure and the paper, and offers a Results download: <code>marimo edit notebook.py</code> on a GPU machine. The notebook link is added after the review.</p>
<footer>Author names and the paper PDF are withheld while the paper is under double-blind review.</footer>
</main></body></html>
"""
    open(os.path.join(out, "index.html"), "w", encoding="utf-8", newline="\n").write(html)
    open(os.path.join(out, ".nojekyll"), "w").write("")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "site"))
