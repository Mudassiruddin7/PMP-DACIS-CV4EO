"""Fail loudly if the anonymous submission leaks an author, an affiliation, an e-mail or a repository owner.

    python tools/check_anonymity.py submission/paper_review.pdf [submission/supplement.pdf] [submission/supplementary_anonymous.zip]
"""
import re
import sys
import zipfile

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    import fitz

NAMES = ["Uddin", "Mudassir", "Taranum", "Fahmina", "Rehman", "Tasneem", "Siddiqui", "Sumrana", "Rizwan", "Samaanah",
         "Nayeem", "Quratulain", "Summaiya", "Kauser", "Sadaf", "Shareef", "Muzammil", "Raheem", "Aqib", "Abdullah",
         "Shahnawaz", "Pasha", "Afroze", "Alam"]
PLACES = ["Osmania", "Muffakham", "MJCET", "Jawaharlal", "JNTU", "BITS Pilani", "Athabasca", "Wilmington", "Hyderabad"]
OTHER = ["github.com/", "Mudassiruddin", "@gmail", "@my.wilmu", "@mjcollege", "@learn.athabascau", "molab.marimo.io"]
# the reference list may name the authors of the earlier preprint in third person; that is allowed and reported, not failed
ALLOWED_CONTEXT = ["arXiv:2601.02353", "Meta-Learning Guided Pruning for Few-Shot Plant Pathology"]


def embedded(py_text):
    """Yield (member name, text) for every file inside the gzip+base64 payloads of a notebook (PKG_B64 / PAPER_B64)."""
    import base64
    import io
    import tarfile
    for m in re.finditer(r'[A-Z]+_B64 = """([A-Za-z0-9+/=\n]+)"""', py_text):
        try:
            tf = tarfile.open(fileobj=io.BytesIO(base64.b64decode(m.group(1))), mode="r:gz")
        except Exception:
            continue
        for mem in tf.getmembers():
            if mem.isfile() and mem.name.endswith((".py", ".tex", ".bib", ".bst", ".sty", ".md", ".txt")):
                yield mem.name, tf.extractfile(mem).read().decode("utf-8", "ignore")


def scan(text, label):
    bad = []
    for w in NAMES + PLACES + OTHER:
        for m in re.finditer(re.escape(w), text):
            ctx = text[max(0, m.start() - 80): m.end() + 80].replace("\n", " ")
            if w == "github.com/" and text[m.start(): m.start() + 40].startswith(("github.com/tectonic-typesetting", "github.com/cvpr-org")):
                continue                                  # the LaTeX engine download, not an author repository
            if any(a in text[max(0, m.start() - 400): m.end() + 400] for a in ALLOWED_CONTEXT):
                print(f"  note ({label}): '{w}' appears next to the third-person preprint reference: ...{ctx}...")
            else:
                bad.append((w, ctx))
    return bad


def main(paths):
    problems = 0
    for p in paths:
        if p.endswith(".pdf"):
            d = fitz.open(p)
            meta = {k: v for k, v in d.metadata.items() if v}
            print(p, "pages", d.page_count, "metadata", meta)
            for k in ("author", "creator", "producer", "title", "subject", "keywords"):
                v = (d.metadata.get(k) or "")
                if any(n in v for n in NAMES):
                    print("  METADATA LEAK:", k, v)
                    problems += 1
            dests = []
            try:                                              # hyperref writes cite keys into named destinations
                dests = list(d.resolve_names().keys())
            except Exception:
                pass
            for pg in d:
                dests += [l.get("nameddest", "") for l in pg.get_links() if l.get("nameddest")]
            for dn in dests:
                if any(n.lower() in dn.lower() for n in NAMES):
                    print("  NAMED-DESTINATION LEAK:", dn)
                    problems += 1
            text = "\n".join(pg.get_text() for pg in d)
            for w, ctx in scan(text, p):
                print(f"  LEAK in {p}: '{w}': ...{ctx}...")
                problems += 1
        elif p.endswith(".zip"):
            z = zipfile.ZipFile(p)
            for n in z.namelist():
                if n.endswith((".py", ".tex", ".md", ".txt", ".json", ".bib", ".yaml", ".toml")):
                    text = z.read(n).decode("utf-8", "ignore")
                    for w, ctx in scan(text, f"{p}:{n}"):
                        print(f"  LEAK in {p}:{n}: '{w}': ...{ctx}...")
                        problems += 1
                    for member, inner in embedded(text):          # the notebook carries gzip+base64 copies of the sources
                        for w, ctx in scan(inner, f"{p}:{n}:{member}"):
                            print(f"  LEAK in {p}:{n}:{member}: '{w}': ...{ctx}...")
                            problems += 1
    print("anonymity check:", "FAILED" if problems else "passed", f"({problems} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
