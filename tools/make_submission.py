"""Collect everything that goes to OpenReview into one folder.

    python tools/make_submission.py OUT_DIR

* paper_review.pdf            anonymous PDF (upload this)
* paper_authors.pdf           same paper with the author block (for the camera-ready and for preprint servers)
* supplement.pdf              extra tables (anonymous)
* supplementary_anonymous.zip notebook + library + tests, no names
* openreview_fields.md        title, abstract, keywords and the author list to type into the submission form
"""
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

AUTHORS = [
    ("Mohammed Mudassir Uddin", "mohd.mudassiruddin7@gmail.com", "Osmania University, Hyderabad, India"),
    ("Fahmina Taranum", "ftaranum@mjcollege.ac.in", "Osmania University, Hyderabad, India"),
    ("Tasneem Bano Rehman", "tasneem.bano@mjcollege.ac.in", "Muffakham Jah College of Engineering and Technology, Hyderabad, India"),
    ("Sumrana Siddiqui", "sumrana.siddiqui@mjcollege.ac.in", "Muffakham Jah College of Engineering and Technology, Hyderabad, India"),
    ("Sara Rizwan", "sara.rizwan014@gmail.com", "Jawaharlal Nehru Technological University Hyderabad, India"),
    ("Samaanah Abdus Salam", "samaanah345@gmail.com", "Jawaharlal Nehru Technological University Hyderabad, India"),
    ("Quratulain Nayeem", "quratulainnayeem@gmail.com", "Osmania University, Hyderabad, India"),
    ("Summaiya Unnisa Begum", "summaiyaunisa@gmail.com", "Jawaharlal Nehru Technological University Hyderabad, India"),
    ("Sadaf Kauser", "sadafkauser8143@gmail.com", "Birla Institute of Technology and Science Pilani, Dubai Campus"),
    ("Muzammil Shareef", "mshareef1@learn.athabascau.ca", "Athabasca University, Canada"),
    ("Abdul Raheem Ahmed", "Aahmed011@my.wilmu.edu", "Wilmington University, United States"),
    ("Mohammed Aqib Abdullah", "aqib.abdullah13@gmail.com", "Osmania University, Hyderabad, India"),
]


def numbers_map(numbers_tex):
    """key -> value from the generated numbers.tex (\\num{key} in the paper)."""
    if not os.path.exists(numbers_tex):
        return {}
    txt = open(numbers_tex, encoding="utf-8").read()
    return dict(re.findall(r"csname num@(.*?)\\endcsname\{(.*?)\}\s*$", txt, re.M))


def abstract_text(tex_path, numbers_tex=None):
    """Abstract as plain text for the OpenReview form: \\num{...} filled in, simple LaTeX turned into text."""
    s = open(tex_path, encoding="utf-8").read()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    t = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    nums = numbers_map(numbers_tex) if numbers_tex else {}
    t = re.sub(r"\\num\{([^}]*)\}", lambda k: nums.get(k.group(1), "??"), t)
    t = t.replace("\\ensuremath{-}", "-").replace("\\ell_1", "L1").replace("\\%", "%").replace("~", " ")
    t = re.sub(r"\$([^$]*)\$", r"\1", t)                      # inline math: keep the content
    return re.sub(r"\s+", " ", t).strip()


def main(out):
    os.makedirs(out, exist_ok=True)
    paper = os.path.join(ROOT, "paper")
    for name in ("paper_review.pdf", "paper_authors.pdf", "supplement.pdf"):
        if os.path.exists(os.path.join(paper, "final", name)):
            shutil.copy(os.path.join(paper, "final", name), os.path.join(out, name))
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import build_notebook
    with zipfile.ZipFile(os.path.join(out, "supplementary_anonymous.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        # the notebook of the zip is rebuilt without the files that name the authors
        z.writestr(zipfile.ZipInfo("pmp_dacis_code/notebooks/pmp_dacis_cv4eo.py", (2025, 1, 1, 0, 0, 0)),
                   build_notebook.build_text(anonymous=True))
        z.writestr(zipfile.ZipInfo("pmp_dacis_code/notebooks/pmp_dacis_cv4eo_ext.py", (2025, 1, 1, 0, 0, 0)),
                   build_notebook.build_ext_text())
        readme = "\n".join([
            "Code for the submission 'Channel Importance Scores for Few-Shot Earth Observation: A Benchmark on Unseen Classes'.",
            "",
            "notebooks/pmp_dacis_cv4eo.py   single-file marimo notebook: press Run all on a GPU machine (about two hours on an",
            "                               RTX 6000-class GPU); it downloads four public datasets from the Hugging Face Hub,",
            "                               reruns every experiment, rebuilds every table and figure, and packs results.zip.",
            "notebooks/pmp_dacis_cv4eo_ext.py   extension run: class partitions 5 to 9 and a pruned MobileNetV2 (about 1.5 hours)",
            "pmp_dacis/                     the library that the notebooks embed (readable copy)",
            "tests/test_core.py             unit tests (python tests/test_core.py)",
            "supplement.pdf                 extra tables and figures that the paper cites",
            "results_tables/                the generated tables, figures and numbers of the reported run",
            ""])
        z.writestr(zipfile.ZipInfo("pmp_dacis_code/README.txt", (2025, 1, 1, 0, 0, 0)), readme)
        supp_pdf = os.path.join(paper, "final", "supplement.pdf")
        if os.path.exists(supp_pdf):                                   # the extra tables travel inside the same upload
            z.write(supp_pdf, "pmp_dacis_code/supplement.pdf")
        for rel in ("requirements.txt", "tests/test_core.py", "tools/build_notebook.py", "tools/notebook_template.py", "tools/notebook_template_ext.py"):
            if os.path.exists(os.path.join(ROOT, rel)):
                z.write(os.path.join(ROOT, rel), "pmp_dacis_code/" + rel)
        for base, dirs, files in os.walk(os.path.join(ROOT, "pmp_dacis")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(base, f)
                    z.write(full, "pmp_dacis_code/" + os.path.relpath(full, ROOT).replace("\\", "/"))
        for base, dirs, files in os.walk(os.path.join(ROOT, "paper", "generated")):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, "pmp_dacis_code/results_tables/" + f)
    title = re.search(r"\\title\{(.*?)\}", open(os.path.join(paper, "paper_title.tex"), encoding="utf-8").read(), re.S).group(1).replace("\\\\", " ")
    with open(os.path.join(out, "openreview_fields.md"), "w", encoding="utf-8") as f:
        f.write("# OpenReview submission form (4th CV4EO workshop at WACV 2027)\n\n")
        f.write(f"**Title:** {title}\n\n**Track / type:** full paper (8 pages plus references), archival, double-blind\n\n")
        f.write(f"**Abstract:** {abstract_text(os.path.join(paper, 'sec_abstract.tex'), os.path.join(paper, 'generated', 'numbers.tex'))}\n\n")
        f.write("**Keywords:** channel pruning; few-shot learning; Earth observation; Sentinel-2 multispectral imagery; aerial scene classification; on-board inference\n\n")
        f.write("**TL;DR:** On classes held out from RESISC45 and EuroSAT-MS, only the Taylor score beats random channel choice, and an unpruned MobileNetV3-Small beats every pruned network.\n\n")
        f.write("**Authors (enter in this order; every author needs an up-to-date OpenReview profile):**\n\n")
        for i, (n, e, a) in enumerate(AUTHORS, 1):
            f.write(f"{i}. {n} <{e}>, {a}\n")
        f.write("\nUpload `paper_review.pdf` as the paper (it is anonymous) and `supplementary_anonymous.zip` as supplementary material.\n")
    with open(os.path.join(out, "SUBMIT_CHECKLIST.md"), "w", encoding="utf-8") as f:
        f.write("""# Before the deadline (Oct 16, 2026, 23:59 AoE = Oct 17, 17:29 IST)

1. Open https://openreview.net/group?id=thecvf.com/WACV/2027/Workshop/CV4EO and create a new submission (full paper).
2. Fill title, abstract, keywords and the 12 authors from `openreview_fields.md`. Every author needs an up-to-date OpenReview
   profile; the form rejects e-mails that have none.
3. Upload `paper_review.pdf` (anonymous; main text ends on page 8, references follow). Upload `supplementary_anonymous.zip` as supplementary
   material (code, both notebooks, generated tables and `supplement.pdf` with the extra tables).
4. The PDF shows `Paper ID *****` and `Submission #*****` in its header. OpenReview assigns the number after you create the
   submission; the number is optional for review and can stay as it is. The header also carries the label `Applications Track`;
   the WACV style file requires one of its track options in review mode (paper/paper_review.tex), so it stays unless the
   workshop asks for something else.
5. Do not upload `paper_authors.pdf` to OpenReview: it names the authors. The raw result zips stay outside this folder.
7. The paper does not cite the arXiv preprint 2601.02353 that introduced DACIS and PMP, and that preprint's numbers do not
   reproduce under this protocol. A reviewer who finds it may ask about the overlap; consider posting a corrected version.
8. Camera-ready is due Nov 20, 2026 (WACV format, authors visible): rebuild with `paper/paper_final.tex`.
""")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "submission"))
