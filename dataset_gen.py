import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from scipy import stats as st
from sklearn import preprocessing

with open("config/land_cover_id.yaml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("config/land_cover_id_eva.yaml", "r", encoding="utf8") as f:
    config_eva = yaml.load(f, Loader=yaml.FullLoader)


def read_tiff(file_path):
    with rasterio.open(file_path, 'r') as reader:
        profile = reader.profile
        tif_as_array = reader.read().astype(np.float32)
    return tif_as_array, profile


def write_tiff(file_path, arr, profile):
    with rasterio.Env():
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(arr.astype(rasterio.float32))


def remove_outliers(x, outlierConstant):
    upper_quartile = np.percentile(x, 75)
    lower_quartile = np.percentile(x, 25)
    # print(upper_quartile, lower_quartile)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    # print(quartileSet)

    result = x * (x >= quartileSet[0]) * (x <= quartileSet[1])

    return result


def standardization(x):
    # scaler = preprocessing.StandardScaler().fit(x)
    # x = scaler.transform(x)
    x = (x-x.mean())/x.std()
    return x


def normalization(x):
    return 255*(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))


def dataset_gen(dataset, nchannel):
    if dataset=='palsar':
        file_list = glob.glob('palsar/*/*/*.tif')
    elif dataset == 's1':
        file_list = glob.glob('palsar_s1/*/*/*.tif')
    else:
        file_list = glob.glob('rcm_train/*/*.tif')
    file_list.sort()
    dataset_train_list = []
    dataset_val_list = []
    for idx in range(len(file_list)):
        file_name = file_list[idx]
        print(file_name)
        fire_id = file_name.split('/')[2][:-5]
        landcover = file_name.split('/')[1]
        if dataset != 'rcm':
            th = config.get(landcover).get(int(fire_id)).get('th')
            bbox = config.get(landcover).get(int(fire_id)).get('bbox')
        tif_array, _ = read_tiff(file_name)

        _, size_x, size_y = tif_array.shape
        tif_array = tif_array.transpose((1, 2, 0))
        tif_array = np.nan_to_num(tif_array)
        if np.mean(tif_array)==0:
            print('empty array')
            os.remove(file_name)
            continue
        if dataset!='rcm':
            data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], nchannel + 1))
            if nchannel == 7:
                for i in range(4):
                    data_output[:, :, i] = remove_outliers(tif_array[:, :, i], 1)
                    data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                for i in range(3):
                    data_output[:, :, i + 4] = remove_outliers(tif_array[:, :, i + 6], 1)
                    data_output[:, :, i + 4] = np.nan_to_num(standardization(data_output[:, :, i + 4]))
                if bbox == 1:
                    data_output[:, :, 7] = np.logical_and(tif_array[:, :, 5] > th, tif_array[:, :, 4] > 0)
                else:
                    data_output[:, :, 7] = tif_array[:, :, 5] > th
            elif nchannel == 4:
                for i in range(4):
                    data_output[:, :, i] = remove_outliers(tif_array[:, :, i], 1)
                    data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                if bbox == 1:
                    data_output[:, :, 4] = np.logical_and(tif_array[:, :, 5] > th, tif_array[:, :, 4] > 0)
                else:
                    data_output[:, :, 4] = tif_array[:, :, 5] > th
            elif nchannel == 3:
                for i in range(3):
                    data_output[:, :, i] = remove_outliers(tif_array[:, :, i + 6], 1)
                    data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                if bbox == 1:
                    data_output[:, :, 3] = np.logical_and(tif_array[:, :, 5] > th, tif_array[:, :, 4] > 0)
                else:
                    data_output[:, :, 3] = tif_array[:, :, 5] > th
        else:
            data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], nchannel + 1))
            for i in range(5):
                data_output[:, :, i] = np.nan_to_num(tif_array[:, :, i])
        del tif_array
        data_index_y = size_y // 256
        data_index_x = size_x // 256
        for i in range(data_index_x):
            for j in range(data_index_y):
                # if (i*data_index_y+j)<int(data_index_y*data_index_x*0.8):
                if 'slave_lake' not in file_name:
                    if st.mode(data_output[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, 0].flatten())[0][0]==0:
                        print('discard empty img')
                        continue
                    dataset_train_list.append(data_output[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :])
                else:
                    if st.mode(data_output[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, 0].flatten())[0][0]==0:
                        print('discard empty img')
                        continue
                    dataset_val_list.append(data_output[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :])
        del data_output
    dataset_train = np.stack(dataset_train_list, axis=0)
    dataset_val = np.stack(dataset_val_list, axis=0)
    print(dataset_train)
    if dataset=='palsar':
        np.save('dataset/proj2_train_'+str(nchannel)+'chan.npy', dataset_train)
        np.save('dataset/proj2_val_'+str(nchannel)+'chan.npy', dataset_val)
    elif dataset =='s1':
        np.save('dataset_s1/proj2_train_'+str(nchannel)+'chan_s1.npy', dataset_train)
        np.save('dataset_s1/proj2_val_'+str(nchannel)+'chan_s1.npy', dataset_val)
    else:
        for i in range(4):
            print(dataset_train[..., i].mean())
            print(dataset_train[..., i].std())
        np.save('dataset_rcm/proj6_train_'+str(nchannel)+'chan_rcm.npy', dataset_train)
        np.save('dataset_rcm/proj6_val_'+str(nchannel)+'chan_rcm.npy', dataset_val)
    return dataset_train, dataset_val


def dataset_eva_gen(dataset, nchannel, overlap):
    land_covers = ['needle', 'broadleaf', 'shrublands', 'savannas', 'grasslands', 'mixed']
    for land_cover in land_covers:
        if dataset=='palsar':
            file_list = glob.glob('palsar_eva/'+land_cover+'/*/*.tif')
        else:
            file_list = glob.glob('palsar_s1_eva/'+land_cover+'/*/*.tif')

        for idx in range(len(file_list)):
            file_name = file_list[idx]
            dataset_eva_list = []
            if len(file_name.split('/')[2])==13:
                fire_id = file_name.split('/')[2][:-5]
                landcover = file_name.split('/')[1]
                print(fire_id)
                th = config_eva.get(landcover).get(int(fire_id)).get('th')
                bbox = config_eva.get(landcover).get(int(fire_id)).get('bbox')
                tif_array, _ = read_tiff(file_name)
                _, size_x, size_y = tif_array.shape
                tif_array = tif_array.transpose((1, 2, 0))
                tif_array = np.nan_to_num(tif_array)
                data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], nchannel+1))
                if nchannel==7:
                    for i in range(4):
                        data_output[:, :, i] = remove_outliers(tif_array[:, :, i], 1)
                        data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                    for i in range(3):
                        data_output[:, :, i + 4] = remove_outliers(tif_array[:, :, i + 6], 1)
                        data_output[:, :, i + 4] = np.nan_to_num(standardization(data_output[:, :, i + 4]))
                    if bbox == 1:
                        data_output[:, :, 7] = np.logical_and(tif_array[:, :, 5] > th, tif_array[:, :, 4] > 0)
                    else:
                        data_output[:, :, 7] = tif_array[:, :, 5] > th
                elif nchannel==4:
                    for i in range(4):
                        data_output[:, :, i] = remove_outliers(tif_array[:, :, i], 1)
                        data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                    if bbox == 1:
                        data_output[:, :, 4] = np.logical_and(tif_array[:, :, 5] > th, tif_array[:, :, 4] > 0)
                    else:
                        data_output[:, :, 4] = tif_array[:, :, 5] > th
                elif nchannel==3:
                    for i in range(3):
                        data_output[:, :, i] = remove_outliers(tif_array[:, :, i + 6], 1)
                        data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                    if bbox == 1:
                        data_output[:, :, 3] = np.logical_and(tif_array[:, :, 5] > th, tif_array[:, :, 4] > 0)
                    else:
                        data_output[:, :, 3] = tif_array[:, :, 5] > th
                del tif_array
                # plt.title('c')
                # plt.imshow(data_output[:, :, 7], cmap='Reds')
                # plt.savefig('label', bbox_inches='tight')
                # plt.show()

                data_index_y = size_y // overlap
                data_index_x = size_x // overlap
                while (data_index_x-1)*overlap+256>=size_x:
                    data_index_x-=1
                while (data_index_y-1)*overlap+256>=size_y:
                    data_index_y-=1
                for i in range(data_index_x):
                    for j in range(data_index_y):
                        if (i * overlap) + 256 < size_x and (j * overlap) + 256 < size_y:
                            dataset_eva_list.append(data_output[i * overlap: (i * overlap) + 256, j * overlap: (j * overlap) + 256, :].astype(np.float32))
                dataset_eva = np.stack(dataset_eva_list, axis=0)
                print(dataset_eva.shape)
                if not os.path.exists('dataset\\'+str(nchannel)+'\\'):
                    os.mkdir('dataset\\'+str(nchannel)+'\\')
                if not os.path.exists('dataset_s1\\'+str(nchannel)+'\\'):
                    os.mkdir('dataset_s1\\'+str(nchannel)+'\\')
                if dataset=='palsar':
                    np.save('dataset/'+str(nchannel)+'/' + landcover + fire_id + 'x' + str(data_index_x) + 'y' + str(data_index_y) + 'nchannels_' + str(nchannel) + '.npy',
                            dataset_eva)
                    del dataset_eva
                else:
                    np.save('dataset_s1/'+str(nchannel)+'/' + landcover + fire_id + 'x' + str(data_index_x) + 'y' + str(data_index_y) + 'nchannels_' + str(nchannel) + '.npy',
                            dataset_eva)
                    del dataset_eva

def dataset_eva_gen_swe(dataset='palsar', nchannel=7):
    land_covers = ['savannas']
    for land_cover in land_covers:
        if dataset=='palsar':
            file_list = glob.glob('palsar_evaluate/'+land_cover+'/*/*.tif')
        else:
            file_list = glob.glob('palsar_s1_evaluate/' + land_cover + '/*/*.tif')
        overlap = 96+128
        for idx in range(len(file_list)):
            file_name = file_list[idx]
            dataset_eva_list = []
            if len(file_name.split('/')[2])!=13:
                fire_id = file_name.split('/')[2]
                landcover = file_name.split('/')[1]
                print(fire_id)
                th = config_eva.get(landcover).get(int(fire_id)).get('th')
                bbox = config_eva.get(landcover).get(int(fire_id)).get('bbox')
                tif_array, _ = read_tiff(file_name)
                _, size_x, size_y = tif_array.shape
                tif_array = tif_array.transpose((1, 2, 0))
                tif_array = np.nan_to_num(tif_array)

                if nchannel==7:
                    data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], 8))
                    for i in range(7):
                        data_output[:, :, i] = remove_outliers(tif_array[:, :, i], 1)
                        data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                    data_output[:, :, 7] = tif_array[:, :, 7] > 0
                elif nchannel==4:
                    data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], 5))
                    for i in range(4):
                        data_output[:, :, i] = remove_outliers(tif_array[:, :, i], 1)
                        data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                    data_output[:, :, 4] = tif_array[:, :, 7] > 0
                elif nchannel==3:
                    data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], 4))
                    for i in range(3):
                        data_output[:, :, i] = remove_outliers(tif_array[:, :, i + 4], 1)
                        data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                    data_output[:, :, 3] = tif_array[:, :, 7] > 0
                del tif_array
                img = np.zeros((data_output.shape[0],data_output.shape[1],3))
                for i in range(3):
                    img[:,:,i] = (data_output[:, :, i]-data_output[:, :, i].min())/(data_output[:, :, i].max()-data_output[:, :, i].min())
                plt.title('dataset' + dataset + 'nchannels' + str(nchannel))
                plt.imshow(img)
                plt.show()
                data_index_y = size_y // overlap
                data_index_x = size_x // overlap
                # while (data_index_x-1)*overlap+256>=size_x:
                #     data_index_x-=1
                # while (data_index_y-1)*overlap+256>=size_y:
                #     data_index_y-=1
                for i in range(data_index_x):
                    for j in range(data_index_y):
                        if (i * overlap) + 256 < size_x and (j * overlap) + 256 < size_y:
                            dataset_eva_list.append(data_output[i * overlap: (i * overlap) + 256, j * overlap: (j * overlap) + 256, :].astype(np.float32))
                        elif(i * overlap) + 256 >= size_x and (j * overlap) + 256 < size_y:
                            temp_img = np.zeros((256, 256, nchannel+1))
                            temp_img[:size_x-i * overlap, :,:] = data_output[i * overlap:, j * overlap:j * overlap+256, :]
                            dataset_eva_list.append(temp_img)
                        elif (i * overlap) + 256 < size_x and (j * overlap) + 256 >= size_y:
                            temp_img = np.zeros((256, 256, nchannel+1))
                            temp_img[:, :size_y - j * overlap,:] = data_output[i * overlap:i * overlap+256, j * overlap:, :]
                            dataset_eva_list.append(temp_img)
                        else:
                            temp_img = np.zeros((256, 256, nchannel+1))
                            temp_img[:size_x - i * overlap, :size_y - j * overlap,:] = data_output[i * overlap:, j * overlap:,:]
                            dataset_eva_list.append(temp_img)
                dataset_eva = np.stack(dataset_eva_list, axis=0)
                print(fire_id + 'x' + str(data_index_x) + 'y' + str(data_index_y) + 'nchannels_' + str(nchannel))
                print(dataset_eva.shape)
                if dataset=='palsar':
                    np.save('dataset/'+str(nchannel)+'/' + landcover + fire_id + 'x' + str(data_index_x) + 'y' + str(data_index_y) + 'nchannels_' + str(nchannel) + '.npy',
                            dataset_eva)
                else:
                    np.save('dataset_s1/'+str(nchannel)+'/' + landcover + fire_id + 'x' + str(data_index_x) + 'y' + str(data_index_y) + 'nchannels_' + str(nchannel) + '.npy',
                            dataset_eva)

if __name__ == '__main__':
    # dataset_gen('s1',nchannel=7)
    for data in ['s1', 'palsar']:
        for nchannel in [3,4,7]:
            dataset_eva_gen(data,nchannel=nchannel, overlap = 96+128)
            # dataset_eva_gen_swe(data, nchannel=nchannel)
