# Paper downloads

| File | What it is |
|---|---|
| [`gwshm-tsfm-icml.pdf`](gwshm-tsfm-icml.pdf) | The paper in ICML 2026 format (anonymous submission version) |
| [`gwshm-tsfm-icml-source.zip`](gwshm-tsfm-icml-source.zip) | Its LaTeX source, ready to compile or upload to Overleaf |

The zip holds `main.tex`, the bibliography (`references.bib`, plus the generated `main.bbl`), the official
ICML 2026 style files, the figure PDFs and the generated table rows. Compile with `latexmk -pdf main.tex`.

For a version with author names, change `\usepackage{icml2026}` in `main.tex` to
`\usepackage[preprint]{icml2026}` (or `[accepted]` for camera-ready) and fill in the affiliation and
e-mail placeholders.

The working source is in [`../paper/`](../paper), where `make_tables.py` and `figures/make_figures.py`
regenerate the tables and figures from the results; rebuild this zip from there after any change.
