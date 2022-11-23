import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

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
            fig = px.box(arr_list, x=labels, y='backscatter')
            fig.show()
            # plt.figure(figsize=(20,8))
            # plt.boxplot(arr_list, labels=labels, showfliers=False)
            # plt.title(land_cover + ' '+ polarization[i])
            # plt.savefig(land_cover + ' ' + polarization[i])
            # plt.show()

def overview_backscatter(dataset='train'):
    if dataset=='train':
        land_covers = ['broadleaf', 'needle', 'grasslands', 'shrublands', 'savannas']
        dir_palsar='/Volumes/yussd/data_proj3/palsar'
        dir_s1='/Volumes/yussd/data_proj3/palsar_s1'
    else:
        land_covers = ['savannas', 'mixed']
        dir_palsar='/Volumes/yussd/data_proj3/palsar_eva'
        dir_s1= '/Volumes/yussd/data_proj3/palsar_s1_eva'
    polarization = ['PALSAR HH', 'PALSAR HV', 'PALSAR HV-HH']
    polarization_s1 = ['S1 VV', 'S1 VH', 'S1 VH-VV']
    title=['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    num_samples=5000

    for j, land_cover in enumerate(land_covers):
        print(land_cover)
        pd_list = []
        fig = go.Figure()

        file_list = glob.glob(dir_palsar+'/'+land_cover+'/*/*.tif')
        for i in range(3):
            for file in file_list:
                if os.path.exists(file):
                    arr, _ = read_tiff(file)
                    burned_arr = np.nan_to_num(arr[i+6,:,:][arr[4,:,:]==1].reshape(-1))
                    unburned_arr = np.nan_to_num(arr[i+6,:,:][arr[4,:,:]!=1].reshape(-1))
                    burned_samples = np.random.choice(burned_arr, num_samples)
                    unburned_samples = np.random.choice(unburned_arr, num_samples)

                if os.path.exists(file.replace(dir_palsar, dir_s1)):
                    arr_s1, _ = read_tiff(file.replace(dir_palsar, dir_s1))
                    burned_arr_s1 = np.nan_to_num(arr_s1[i+6,:,:][arr_s1[4,:,:]==1].reshape(-1))
                    unburned_arr_s1 = np.nan_to_num(arr_s1[i+6,:,:][arr_s1[4,:,:]!=1].reshape(-1))
                    burned_samples_s1 = np.random.choice(burned_arr_s1, num_samples)
                    unburned_samples_s1 = np.random.choice(unburned_arr_s1, num_samples)

            burned = np.concatenate((burned_samples, burned_samples_s1),axis=0)
            burned_label = np.array([polarization[i]]*5000+[polarization_s1[i]]*5000)
            unburned = np.concatenate((unburned_samples, unburned_samples_s1),axis=0)
            unburned_label = np.array([polarization[i]]*5000+[polarization_s1[i]]*5000)
            if i==0:
                fig.add_trace(go.Box(y=burned, x=burned_label, name='Bunred', marker_color = 'indianred', showlegend=True))
                fig.add_trace(go.Box(y=unburned, x=unburned_label, name='Unbunred', marker_color = 'lightseagreen', showlegend=True))
            else:
                fig.add_trace(go.Box(y=burned, x=burned_label, marker_color = 'indianred', showlegend=False))
                fig.add_trace(go.Box(y=unburned, x=unburned_label, marker_color = 'lightseagreen', showlegend=False))
            fig.update_layout(yaxis_title='Backscatter', boxmode='group', width=700, height=450, margin=dict(l=20, r=20, t=20, b=20))
        fig.write_image("svgs/"+land_cover.capitalize()+"_"+dataset+".svg")
        fig.show()

if __name__=='__main__':
    overview_backscatter('test')