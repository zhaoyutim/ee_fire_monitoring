import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from sklearn import preprocessing

with open("dataset_config/land_cover_id.yaml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("dataset_config/land_cover_id_eva.yaml", "r", encoding="utf8") as f:
    config_eva = yaml.load(f, Loader=yaml.FullLoader)


def read_tiff(file_path):
    with rasterio.open(file_path, 'r') as reader:
        profile = reader.profile
        tif_as_array = reader.read()
    return tif_as_array, profile


def write_tiff(file_path, arr, profile):
    with rasterio.Env():
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(arr.astype(rasterio.float32))


def remove_outliers(x, outlierConstant):
    upper_quartile = np.percentile(x, 75)
    lower_quartile = np.percentile(x, 25)
    print(upper_quartile, lower_quartile)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    print(quartileSet)

    result = x * (x >= quartileSet[0]) * (x <= quartileSet[1])

    return result


def standardization(x):
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    return x


def normalization(x):
    return 255*(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))


def dataset_gen():
    file_list = glob.glob('palsar/*/*/*.tif')
    file_list.sort()
    dataset_train_list = []
    dataset_val_list = []
    for idx in range(len(file_list)):
        file_name = file_list[idx]
        print(file_name)
        fire_id = file_name.split('/')[2][:-5]
        landcover = file_name.split('/')[1]
        th = config.get(landcover).get(int(fire_id)).get('th')
        bbox = config.get(landcover).get(int(fire_id)).get('bbox')
        tif_array, _ = read_tiff(file_name)

        _, size_x, size_y = tif_array.shape
        tif_array = tif_array.transpose((1, 2, 0))
        tif_array = np.nan_to_num(tif_array)
        data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], 5))
        for i in range(4):
            data_output[:, :, i] = remove_outliers(tif_array[:, :, i], 1)
            # data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
        # img = (tif_array[:, :, :3] - tif_array[:, :, :3].min()) / (
        #             tif_array[:, :, :3].max() - tif_array[:, :, :3].min())
        # plt.imshow(img)
        # plt.show()
        if bbox == 1:
            data_output[:, :, 4] = np.logical_and(tif_array[:, :, 5] > th, tif_array[:, :, 4] > 0)
        else:
            data_output[:, :, 4] = tif_array[:, :, 5] > th
        img = data_output[:, :, 4]
        # plt.imshow(img)
        # plt.show()
        data_index_y = size_y // 256
        data_index_x = size_x // 256
        for i in range(data_index_x):
            for j in range(data_index_y):
                if (i*data_index_y+j)<int(data_index_y*data_index_x*0.8):
                    dataset_train_list.append(data_output[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :])
                else:
                    dataset_val_list.append(data_output[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :])

    dataset_train = np.stack(dataset_train_list, axis=0)
    dataset_val = np.stack(dataset_val_list, axis=0)

    np.save('dataset/proj2_train_4chan.npy', dataset_train)
    np.save('dataset/proj2_val_4chan.npy', dataset_val)
    return dataset_train, dataset_val


def dataset_eva_gen():
    land_covers = ['needle', 'broadleaf', 'grasslands', 'shrublands', 'savannas', 'mixed']
    for land_cover in land_covers:
        file_list = glob.glob('palsar_eva/'+land_cover+'/*/*.tif')
        overlap = 128
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
                data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], 5))
                img = np.zeros((tif_array.shape[0], tif_array.shape[1], 3))
                for i in range(4):
                    data_output[:, :, i] = remove_outliers(tif_array[:, :, i], 1)
                    # data_output[:, :, i] = np.nan_to_num(standardization(data_output[:, :, i]))
                img = (tif_array[:, :, :3] - tif_array[:, :, :3].min()) / (
                            tif_array[:, :, :3].max() - tif_array[:, :, :3].min())
                plt.imshow(img)
                plt.show()
                if bbox == 1:
                    data_output[:, :, 4] = np.logical_and(tif_array[:, :, 5] > th, tif_array[:, :, 4] > 0)
                else:
                    data_output[:, :, 4] = tif_array[:, :, 5] > th
                plt.title('c')
                plt.imshow(data_output[:, :, 4], cmap='Reds')
                plt.savefig('label', bbox_inches='tight')
                plt.show()

                data_index_y = size_y // overlap
                data_index_x = size_x // overlap
                for i in range(data_index_x):
                    for j in range(data_index_y):
                        if (i * overlap) + 256 < size_x and (j * overlap) + 256 < size_y:
                            dataset_eva_list.append(data_output[i * overlap: (i * overlap) + 256, j * overlap: (j * overlap) + 256, :])
                dataset_eva = np.stack(dataset_eva_list, axis=0)
                np.save('dataset/' + landcover + fire_id + 'x' + str(data_index_x) + 'y' + str(data_index_y) + '.npy',
                        dataset_eva)

def dataset_eva_gen_swe():
    land_covers = ['savannas']
    for land_cover in land_covers:
        file_list = glob.glob('palsar_eva/'+land_cover+'/*/*.tif')
        overlap = 128
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
                data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], 5))
                img = np.zeros((tif_array.shape[0], tif_array.shape[1], 3))
                for i in range(4):
                    data_output[:, :, i] = remove_outliers(tif_array[:, :, i], 1)
                    # data_output[:, :, i] = np.abs(np.nan_to_num(standardization(data_output[:, :, i])))
                img = (tif_array[:, :, 4:7] - tif_array[:, :, 4:7].min()) / (tif_array[:, :, 4:7].max() - tif_array[:, :, 4:7].min())
                plt.imshow(img)
                plt.show()
                label = np.nan_to_num(tif_array[:, :, 7])
                data_output[:, :, 4] = label > 0
                plt.title('c')
                plt.imshow(data_output[:, :, 4], cmap='Reds')
                plt.savefig('label', bbox_inches='tight')
                plt.show()

                data_index_y = size_y // overlap
                data_index_x = size_x // overlap
                for i in range(data_index_x):
                    for j in range(data_index_y):
                        if (i * overlap) + 256 < size_x and (j * overlap) + 256 < size_y:
                            dataset_eva_list.append(data_output[i * overlap: (i * overlap) + 256, j * overlap: (j * overlap) + 256, :])
                dataset_eva = np.stack(dataset_eva_list, axis=0)
                np.save('dataset/' + landcover + fire_id + 'x' + str(data_index_x) + 'y' + str(data_index_y) + '.npy',
                        dataset_eva)

if __name__ == '__main__':
    dataset_gen()
    # dataset_eva_gen()
    # dataset_eva_gen_swe()
