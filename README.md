# FYS-STK4155 Project 1
This is our source code for Project 1 in the course FYS-STK4155 Applied Data Analysis and Machine Learning at the University of Oslo.

The project is based on various introductory regression methods and resampling. 
We will be using the following regression methods,
1. Ordinary Least Squares 
2. Ridge
3. Lasso

in combination with our own implementation of k-fold cross-validation in order to eventually model a two-dimensional polynomial fit to real terrain data downloaded from [USGS EarthExplorer](https://earthexplorer.usgs.gov/).

To run all test functions, generate data and plots used in the report, please run main_script.sh.

## Additional Figures mentioned in report
### Section 5.1
* [3D plotted Ridge fit of Franke's function](/doc/figs/3dmodel_Ridge_Franke.pdf).
* [3D plotted LASSO fit of Franke's function](/doc/figs/3dmodel_Lasso_Franke.pdf).

### Section 5.2
* [3D plotted Ridge fit of terrain data](/doc/figs/3dmodel_Ridge_terrain.pdf).
* [3D plotted LASSO fit of terrain data](/doc/figs/3dmodel_Lasso_terrain.pdf).
* [EPE plotted for OLS as a function of model complexity, using only every 150th point in the x and y directions](/doc/figs/biasvariancetradeoff_ols_terrain_150_skip.pdf).
* [EPE plotted for Ridge as a function of the hyperparameter, using only every 150th point in the x and y directions](/doc/figs/biasvariancetradeoff_Ridge_terrain_150_skip.pdf).
* [EPE plotted for LASSO as a function of the hyperparameter, using only every 150th point in the x and y directions](/doc/figs/biasvariancetradeoff_LASSO_terrain_150_skip.pdf).
