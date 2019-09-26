#!/bin/bash

cd src

echo "Run test functions? (y/n)"
read yn_r
if [ "$yn_r" == "y" ]
then 
  pytest -v
fi

echo "Print r2 scores? (y/n)"
read yn_r2
if [ "$yn_r2" == "y" ]
then
  python r2_scores.py
fi

echo "Create plots? (y/n)"
read yn_r3 
if [ "$yn_r3" == "y" ]
then
  python bias_variance_error_terrain.py
  python bias_variance_error_Franke.py
  python beta_variance_ols_plot.py
  python model_plots.py
fi

echo "Build report? (y/n)"
read yn_r4
# If y, compile TeX document. The compilation is run many times because
# bibtex is usually non-cooperative...
if [ "$yn_r4" == "y" ]
then
  cd ../doc/
  pdflatex -synctex=1 -interaction=nonstopmode report_1.tex
  bibtex report_1.aux
  pdflatex -synctex=1 -interaction=nonstopmode report_1.tex
  bibtex report_1.aux
  pdflatex -synctex=1 -interaction=nonstopmode report_1.tex
fi
