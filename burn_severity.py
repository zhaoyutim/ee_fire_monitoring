from glob import glob

import numpy as np
import rasterio
from matplotlib import pyplot as plt


def read_tiff(file_path):
    with rasterio.open(file_path, 'r') as reader:
        profile = reader.profile
        tif_as_array = reader.read().astype(np.float32)
    return tif_as_array, profile

path = 'BurnSeverityMapV2/'

if __name__=='__main__':
    file_list = glob(path+'*.tif')
    th_low=0.1
    th_medium = 0.269
    th_medium_high=0.439
    th_high=0.659
    for file in file_list:
        array, _ = read_tiff(file)
        array = array.transpose((1,2,0)).squeeze()
        # burn_severity=np.zeros((array.shape))
        # burn_severity[array<th_low]=0
        # burn_severity[np.logical_and(array > th_low, array < th_medium)] = 1
        # burn_severity[np.logical_and(array > th_medium, array < th_medium_high)] = 2
        # burn_severity[np.logical_and(array > th_medium_high, array < th_high)] = 3
        # burn_severity[array > th_high] = 4
        print(array.max())

        plt.axis('off')
        plt.imshow(array, cmap='OrRd', interpolation='nearest')
        plt.colorbar()
        plt.savefig(file.replace('.tif', '.png'), bbox_inches='tight')

        plt.show()