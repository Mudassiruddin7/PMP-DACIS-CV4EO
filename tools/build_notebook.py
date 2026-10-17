"""Build notebooks/pmp_dacis_cv4eo.py: a single-file marimo notebook that embeds the library and paper sources.

    python tools/build_notebook.py

`build_text(anonymous=True)` returns the same notebook without the files that name the authors (author block, the
bibliography with the authors of the earlier preprint, the author-version wrapper). The anonymous supplementary
material of the double-blind submission is built from it.
"""
import base64
import io
import os
import tarfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAMING_FILES = ("paper_authors.tex", "refs_names.bib", "paper_final.tex")


def tar_b64(base_dir, rel_dir, keep=lambda p: True, arc_prefix=""):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        top = os.path.join(base_dir, rel_dir)
        for dp, dn, fn in os.walk(top):
            dn[:] = [d for d in dn if d not in ("__pycache__", "generated", "final")]
            for f in sorted(fn):
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, top)
                if not keep(rel):
                    continue
                info = tf.gettarinfo(full, arcname=os.path.join(arc_prefix, rel).replace("\\", "/"))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 1760000000  # fixed, after 1980, so extracted files can be zipped
                with open(full, "rb") as fh:
                    tf.addfile(info, fh)
    b = base64.b64encode(buf.getvalue()).decode()
    return "\n".join(b[i:i + 110] for i in range(0, len(b), 110))


def build_text(anonymous=False):
    pkg = tar_b64(ROOT, "pmp_dacis", keep=lambda r: r.endswith(".py"), arc_prefix="pmp_dacis")

    def keep_paper(r):
        if r.endswith((".pdf", ".aux", ".log", ".out", ".bbl", ".blg")):
            return False
        return not (anonymous and r.replace("\\", "/") in NAMING_FILES)

    paper = tar_b64(ROOT, "paper", keep=keep_paper)
    tpl = open(os.path.join(ROOT, "tools", "notebook_template.py"), encoding="utf-8").read()
    assert "@@PKG_B64@@" in tpl and "@@PAPER_B64@@" in tpl
    return tpl.replace("@@PKG_B64@@", pkg).replace("@@PAPER_B64@@", paper)


def build_ext_text():
    """The extension notebook (more partitions, pruned MobileNetV2): library only, no paper sources."""
    pkg = tar_b64(ROOT, "pmp_dacis", keep=lambda r: r.endswith(".py"), arc_prefix="pmp_dacis")
    tpl = open(os.path.join(ROOT, "tools", "notebook_template_ext.py"), encoding="utf-8").read()
    return tpl.replace("@@PKG_B64@@", pkg)


def main():
    out_ext = os.path.join(ROOT, "notebooks", "pmp_dacis_cv4eo_ext.py")
    os.makedirs(os.path.dirname(out_ext), exist_ok=True)
    open(out_ext, "w", encoding="utf-8", newline="\n").write(build_ext_text())
    print(f"wrote {out_ext}")
    nb = build_text(False)
    out = os.path.join(ROOT, "notebooks", "pmp_dacis_cv4eo.py")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8", newline="\n").write(nb)
    print(f"wrote {out}  ({len(nb) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
