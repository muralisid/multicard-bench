# Build the paper. Tectonic fetches what it needs and caches it.
PDF := build/main.pdf

$(PDF): main.tex sections/*.tex refs.bib figures/*.pdf
	mkdir -p build
	tectonic -X compile main.tex --outdir build

.PHONY: clean arxiv
clean:
	rm -rf build

# arXiv wants the sources plus a compiled .bbl, since it does not run BibTeX.
arxiv: $(PDF)
	mkdir -p build/arxiv/sections build/arxiv/figures
	cp main.tex build/arxiv/
	cp sections/*.tex build/arxiv/sections/
	cp figures/*.pdf build/arxiv/figures/
	cp build/main.bbl build/arxiv/ 2>/dev/null || echo "run with --keep-intermediates to capture main.bbl"
	cd build/arxiv && tar czf ../arxiv-submission.tar.gz .
