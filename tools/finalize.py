"""Turn a results.zip (from the notebook) into the final tables, figures, PDFs and the submission folder.

    python tools/finalize.py path/to/results.zip [--submission DIR]

Steps: unpack the runs into results/, rebuild tables/figures/numbers into paper/generated, compile the anonymous
paper, the author version and the supplement with Tectonic, check the anonymous files for leaks, and copy the files
that go to OpenReview into the submission folder.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from pmp_dacis import paper as PP  # noqa: E402
from pmp_dacis import report as RP  # noqa: E402


def repl_numbers(R):
    """Numbers from a replication on a second machine, if its runs are in results_replication/."""
    rep = os.path.join(ROOT, "results_replication")
    if not os.path.isdir(os.path.join(rep, "runs")):
        return {}
    return RP.replication_numbers(R, RP.Results(rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--submission", default=os.path.join(ROOT, "..", "submission"))
    ap.add_argument("--tectonic", default=None)
    ap.add_argument("--ext", default=None, help="results_extension.zip of the extension notebook (more partitions, pruned MobileNetV2)")
    a = ap.parse_args()
    res = os.path.join(ROOT, "results")
    if os.path.exists(res):
        shutil.rmtree(res)
    os.makedirs(res)
    z = zipfile.ZipFile(a.zip)
    for n in z.namelist():
        if n.startswith(("runs/", "tuning_m_lr", "host.json", "log.txt", "config_used.json", "claims_check.txt", "paper_build_log.txt", "failures.json")):
            z.extract(n, res)
    if a.ext:
        ze = zipfile.ZipFile(a.ext)
        for n in ze.namelist():
            if n.startswith("runs/"):
                ze.extract(n, res)
            elif n == "tuning_m_lr_resisc45_low.json":
                ze.extract(n, res)
            elif n == "log.txt":
                open(os.path.join(res, "log_extension.txt"), "wb").write(ze.read(n))
            elif n == "failures.json":
                open(os.path.join(res, "failures_extension.json"), "wb").write(ze.read(n))
    print("unpacked", len(os.listdir(res)), "entries into", res)
    gen = os.path.join(ROOT, "paper", "generated")
    for f in glob.glob(os.path.join(gen, "*")):
        if not f.endswith(".gitkeep"):
            os.remove(f)
    R = RP.build_all(res, gen)
    RP.write_numbers(R, os.path.join(gen, "numbers.tex"), extra=repl_numbers(R))
    readme = os.path.join(ROOT, "README.md")
    txt = open(readme, encoding="utf-8").read()
    i, j = txt.index("<!-- RESULTS:BEGIN -->"), txt.index("<!-- RESULTS:END -->")
    open(readme, "w", encoding="utf-8").write(txt[:i] + "<!-- RESULTS:BEGIN -->" + chr(10) + RP.readme_block(R) + txt[j:])
    claims = RP.claims_check(R)
    open(os.path.join(res, "claims_check.txt"), "w").write(claims)
    print(claims)
    tect = a.tectonic or PP.ensure_tectonic(os.path.join(ROOT, "..", "_work", "bin"))
    paper = os.path.join(ROOT, "paper")
    for name in ("paper_review", "paper_final", "paper_supp"):
        pdf, log = PP.compile_pdf(tect, paper, name + ".tex")
        print(name, "OK" if pdf else "FAILED")
        if not pdf:
            print(log[-2500:])
            sys.exit(1)
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_numbers.py"), paper]
                   + [os.path.join(paper, n + ".pdf") for n in ("paper_review", "paper_final", "paper_supp")])
    final = os.path.join(paper, "final")
    os.makedirs(final, exist_ok=True)
    for name in ("paper_review", "paper_final", "paper_supp"):
        shutil.copy(os.path.join(paper, name + ".pdf"), os.path.join(final, name + ".pdf"))
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_notebook.py")], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "make_submission.py"), a.submission], check=True)
    rc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "check_anonymity.py"),
                         os.path.join(a.submission, "paper_review.pdf"), os.path.join(a.submission, "supplement.pdf"),
                         os.path.join(a.submission, "supplementary_anonymous.zip")]).returncode
    shutil.copy(a.zip, os.path.join(a.submission, "results_molab.zip"))
    print("anonymity check exit code", rc)


if __name__ == "__main__":
    main()
