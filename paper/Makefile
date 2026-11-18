# Build the paper. Needs a LaTeX installation (pdflatex) and the ruin package installed.
#
#   make          rebuild ruin.pdf from the cached figures
#   make figures  redraw the figures (simulates one shift; uses the cached analyses in data/)
#   make refresh  recompute the cached analyses too (a few minutes)

.PHONY: all figures refresh clean

all: ruin.pdf

ruin.pdf: ruin.tex figures/anatomy.pdf figures/structure.pdf figures/cause_effect.pdf figures/organization.pdf
	pdflatex -interaction=nonstopmode -halt-on-error ruin.tex
	pdflatex -interaction=nonstopmode -halt-on-error ruin.tex

figures:
	cd .. && python paper/make_figures.py

refresh:
	cd .. && python paper/make_figures.py --refresh

clean:
	rm -f ruin.aux ruin.log ruin.out build.log
