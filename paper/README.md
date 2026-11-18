# The paper

`ruin.pdf` is a four-page, two-column paper that explains RUIN in plain language: what it is
for, the model, its main formulas, what it adds to research and conceptualization, what it has
found so far, and its limits and future work.

- `ruin.tex`: the source (pdflatex, Computer Modern, A4).
- `make_figures.py`: draws the four figures. Figure 1 simulates one shift of the reference
  scenario; the others use the cached analyses in `data/`, which come from `ruin structure`,
  `ruin destiny --blind --observer --incidents`, and `ruin organize`. `--refresh` recomputes them.
- `data/`: those cached results, as JSON.
- `Makefile`: `make` rebuilds the PDF.

Every number in the paper is taken from the repository's README and changelog, which record the
runs that produced them.
