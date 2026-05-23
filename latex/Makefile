# ----------------------------------------------------------------------
# LaTeX build Makefile for the journal manuscript PDF
#
# Targets:
#   make             # build both English and Chinese PDFs
#   make en          # build English PDF only
#   make zh          # build Chinese PDF only
#   make regenerate  # only run the md_to_latex.py converter (both langs)
#   make clean       # remove build artefacts (.aux, .log, ...)
#   make cleanall    # also remove the generated .tex and .pdf files
#
# Requirements:
#   - TeX Live (any recent distribution)
#   - lualatex (both EN and ZH use fontspec/luatexja)
# ----------------------------------------------------------------------

ENGINE ?= lualatex

MD_EN     ?= ../outputs/markdown_journal/report.md
MD_ZH     ?= ../outputs/markdown_journal_zh/report.md

TEX_EN    := en/main.tex
TEX_ZH    := zh/main.tex
PDF_EN    := en/main.pdf
PDF_ZH    := zh/main.pdf

AUX_EXT   := aux log toc out fdb_latexmk fls synctex.gz

.PHONY: all en zh regenerate clean cleanall

all: en zh

en: $(PDF_EN)
zh: $(PDF_ZH)

regenerate: $(TEX_EN) $(TEX_ZH)

$(TEX_EN): md_to_latex.py $(MD_EN)
	python3 md_to_latex.py --input $(MD_EN) --output $(TEX_EN) --lang en

$(TEX_ZH): md_to_latex.py $(MD_ZH)
	python3 md_to_latex.py --input $(MD_ZH) --output $(TEX_ZH) --lang zh

# Two passes so \label/\ref and any toc entries resolve.
$(PDF_EN): $(TEX_EN) $(wildcard en/*.png)
	cd en && $(ENGINE) -interaction=nonstopmode main.tex || true
	cd en && $(ENGINE) -interaction=nonstopmode main.tex

$(PDF_ZH): $(TEX_ZH) $(wildcard zh/*.png)
	cd zh && $(ENGINE) -interaction=nonstopmode main.tex || true
	cd zh && $(ENGINE) -interaction=nonstopmode main.tex

clean:
	@rm -f $(addprefix en/main.,$(AUX_EXT)) $(addprefix zh/main.,$(AUX_EXT))

cleanall: clean
	@rm -f $(PDF_EN) $(PDF_ZH) $(TEX_EN) $(TEX_ZH)
