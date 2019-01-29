
all: paper.pdf # supplementary.pdf  

paper.pdf: paper.tex references.bib
	pdflatex paper.tex
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex

