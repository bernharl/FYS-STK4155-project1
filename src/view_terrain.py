import numpy as np

# from imageio import imread
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from main import OrdinaryLeastSquares, RidgeRegression, LassoRegression

"""
# Load the terrain
terrain1 = imageio.imread("datafiles/SRTM_data_Kolnes_Norway.tif")
# terrain1 = imread("datafiles/SRTM_data_Kolnes_Norway2.tif")
# terrain1 = imread("datafiles/SRTM_data_Norway_1.tif")
# terrain1 = imread("datafiles/SRTM_data_Norway_2.tif")
print(terrain1)
exit()
# Show the terrain
plt.figure()
plt.title("Terrain over Norway 1")
plt.imageio.imshow(terrain1, cmap="plasma")
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""

def read_image_data(self):
    if self.path == None:
        file_name_path = self.filename
    else:
        file_name_path = self.path + self.filename
    imdata = imageio.imread(file_name_path)
    return imdata
terrain_ols = OrdinaryLeastSquares(alt_data=read_image_data, path="datafiles/", filename="SRTM_data_Kolnes_Norway.tif")
