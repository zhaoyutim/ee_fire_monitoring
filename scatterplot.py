import glob

import matplotlib.pyplot as plt
import numpy as np
import rasterio

def read_tiff(file_path):
    with rasterio.open(file_path, 'r') as reader:
        profile = reader.profile
        tif_as_array = reader.read()
    return tif_as_array, profile

def main():
    land_covers = ['broadleaf', 'needle', 'grasslands', 'shrublands', 'savannas']
    for land_cover in land_covers:
        polarization = ['HH', 'HV', 'HV-HH']
        file_list = glob.glob('palsar/'+land_cover+'/*/*.tif')
        for i in range(3):
            arr_list = []
            labels = []
            for file in file_list:
                arr, _ = read_tiff(file)
                burned_arr = np.nan_to_num(arr[i,:,:][arr[3,:,:]==1].reshape(-1))
                unburned_arr = np.nan_to_num(arr[i,:,:][arr[3,:,:]!=1].reshape(-1))
                burned_samples = np.random.choice(burned_arr, 5000)
                unburned_samples = np.random.choice(unburned_arr, 5000)
                arr_list.append(burned_samples)
                arr_list.append(unburned_samples)
                fire_num = file.split('/')[2]
                labels.append('burned' + fire_num)
                labels.append('unburned' + fire_num)
    
            plt.figure(figsize=(20,8))
            plt.boxplot(arr_list, labels=labels, showfliers=False)
            plt.title(land_cover + ' '+ polarization[i])
            plt.savefig(land_cover + ' ' + polarization[i])
            plt.show()

def overview_backscatter():
    land_covers = ['broadleaf', 'needle', 'grasslands', 'shrublands', 'savannas']
    polarization = ['HH', 'HV', 'HV-HH']
    title=['(a)', '(b)', '(c)']
    for i in range(3):
        arr_list = []
        labels = []
        for land_cover in land_covers:
            file_list = glob.glob('palsar/'+land_cover+'/*/*.tif')
            burned_arr_list = []
            unburned_arr_list = []
            for file in file_list:
                arr, _ = read_tiff(file)
                burned_arr = np.nan_to_num(arr[i,:,:][arr[3,:,:]==1].reshape(-1))
                unburned_arr = np.nan_to_num(arr[i,:,:][arr[3,:,:]!=1].reshape(-1))
                burned_samples = np.random.choice(burned_arr, 5000)
                unburned_samples = np.random.choice(unburned_arr, 5000)
                burned_arr_list.append(burned_samples)
                unburned_arr_list.append(unburned_samples)
            labels.append('burned_' + land_cover)
            labels.append('unburned_' + land_cover)
            burned_cover_arr = np.concatenate(burned_arr_list)
            unburned_cover_arr = np.concatenate(unburned_arr_list)
            arr_list.append(burned_cover_arr)
            arr_list.append(unburned_cover_arr)

        plt.rc('font', size=15)
        plt.figure(figsize=(30,8))

        plt.boxplot(arr_list, labels=labels, showfliers=False)
        plt.title(title[i])
        plt.savefig('burned_unburned' + ' ' + polarization[i], bbox_inches='tight')
        plt.show()

if __name__=='__main__':
    overview_backscatter()