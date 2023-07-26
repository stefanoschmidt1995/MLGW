DOCNAME=mlgw_NN

all: $(DOCNAME).pdf

$(DOCNAME).pdf: $(DOCNAME).tex biblio
	-pdflatex -synctex=1 -interaction=nonstopmode $(DOCNAME).tex
	-pdflatex -synctex=1 -interaction=nonstopmode $(DOCNAME).tex

biblio: biblio.bib
	-pdflatex -synctex=1 -interaction=nonstopmode $(DOCNAME).tex
	bibtex $(DOCNAME).aux

submit:
	bash my_submission_tarb.sh $(DOCNAME).tex biblio.bib $(DOCNAME).bbl img

remove:
	rm -f $(DOCNAME).bbl $(DOCNAME).out $(DOCNAME)Notes.bib $(DOCNAME).aux $(DOCNAME).log $(DOCNAME).blg $(DOCNAME).toc $(DOCNAME).synctex.gz

clean: remove
	rm $(DOCNAME).pdf
	

#The command pdflatex may fail: we need to put a dash - in fron of the instruction, to make sure the makefile doesn't stop after failure
