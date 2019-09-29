# FYS-STK4155 Project 1
This is our source code for Project 1 in the course FYS-STK4155 Applied Data Analysis and Machine Learning at the University of Oslo.

The project is based on various introductory regression methods and resampling. 
We will be using the following regression methods,
1. Ordinary Least Squares 
2. Ridge
3. Lasso

in combination with our own implementation of k-fold cross-validation in order to eventually model a two-dimensional polynomial fit to real terrain data downloaded from [USGS EarthExplorer](https://earthexplorer.usgs.gov/).

To run all test functions, generate data and plots used in the report, please run main_script.sh.

## Source structure
* src/main.py: Main script containing all classes used in this project.
* src/test_main.py: Contains test functions for main.py. Use [pytest](https://github.com/pytest-dev/pytest) to run tests.
* src/beta_variance_ols_plot.py: Calculates the variance of the regression parameters for OLS, both for Franke and terrain data.Saves plots as .pdf in doc/figs/
* bias_variance_error_Franke.py:	Calculates EPE using k-fold cross validation for OLS, Ridge and LASSO on Franke data using different polynomial degrees and hyperparameters. Plots saved as .pdf to doc/figs/
* bias_variance_error_terrain.py: Calculates EPE using k-fold cross validation for OLS, Ridge and LASSO on Terrain data using different polynomial degrees and hyperparameters. Plots saved as .pdf to doc/figs/
* model_plots.py: Creates 3D plots of our bet OLS, Ridge and LASSO models for both datasets. Figures are saved as .pdf to doc/figs/
* r2_scores.py: Calculates R2 scores of our best models for OLS, Ridge and LASSO models for both datasets. Results are printed in the terminal after running.
* doc/report_1.tex: Main report of the project.
* main_script.sh: Shell script that automatically runs all necessary python scripts and builds the TeX report using the newly generated figures.

## Additional Figures mentioned in report
Unfortunately, Github does not support embedding graphics in pdf format, so we have to link to them instead. The reason we use .pdf is that we want to use vector graphics for figures.
### Section 5.1
* [3D plotted Ridge fit of Franke's function](/doc/figs/3dmodel_Ridge_Franke.pdf).
* [3D plotted LASSO fit of Franke's function](/doc/figs/3dmodel_Lasso_Franke.pdf).

### Section 5.2
* [3D plotted Ridge fit of terrain data](/doc/figs/3dmodel_Ridge_terrain.pdf).
* [3D plotted LASSO fit of terrain data](/doc/figs/3dmodel_Lasso_terrain.pdf).
* [EPE plotted for OLS as a function of model complexity, using only every 150th point in the x and y directions of the terrain data](/doc/figs/biasvariancetradeoff_ols_terrain_150_skip.pdf).
* [EPE plotted for Ridge as a function of the hyperparameter, using only every 150th point in the x and y directions of the terrain data](/doc/figs/biasvariancetradeoff_Ridge_terrain_150_skip.pdf).
* [EPE plotted for LASSO as a function of the hyperparameter, using only every 150th point in the x and y directions of the terrain data](/doc/figs/biasvariancetradeoff_LASSO_terrain_150_skip.pdf).
* [Variance of the OLS parameters, using only every 150th point in the x and y directions of the terrain data](/doc/figs/beta_variance_ols_terrain_150_skip.pdf)
