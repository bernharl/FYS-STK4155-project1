#!/bin/bash

cd src
echo "Creating estimated prediction error plots"

python r2_scores.py
python bias_variance_error_terrain.py
python bias_variance_error_Franke.py
python beta_variance_ols_plot.py

echo "Build report? (y/n)"
read yn
# If y, compile TeX document. The compilation is run many times because
# bibtex is usually non-cooperative...
if [ "$yn" == "y" ]
then
  cd ../doc/
  pdflatex -synctex=1 -interaction=nonstopmode report_1.tex
  bibtex report_1.aux
  pdflatex -synctex=1 -interaction=nonstopmode report_1.tex
  bibtex report_1.aux
  pdflatex -synctex=1 -interaction=nonstopmode report_1.tex
fi
