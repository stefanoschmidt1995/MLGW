all: mlgw_NN.pdf

mlgw_NN.pdf: mlgw_NN.tex mlgw_NN.bbl mlgw_NN.aux
	pdflatex -synctex=1 -interaction=nonstopmode "mlgw_NN".tex

mlgw_NN.aux:
	pdflatex -synctex=1 -interaction=nonstopmode "mlgw_NN".tex

mlgw_NN.bbl: mlgw_NN.aux
	bibtex "mlgw_NN".aux

clean:
	rm -f mlgw_NN.bbl mlgw_NN.out mlgw_NNNotes.bib mlgw_NN.aux mlgw_NN.log mlgw_NN.blg mlgw_NN.toc mlgw_NN.synctex.gz mlgw_NN.pdf
	rm -rf scripts/__pycache__
