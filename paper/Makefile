# Build the preprint with tectonic (brew install tectonic). `make figs` regenerates plots from docs/perf/data.
PY ?= ../scripts/qiskit-baseline/.venv/bin/python
all: main.pdf
main.pdf: main.tex refs.bib sections/*.tex
	tectonic -X compile main.tex
figs:
	$(PY) figs/make_figs.py
clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out
.PHONY: all figs clean
