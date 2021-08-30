import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def read_tiff(file_path):
    with rasterio.open(file_path, 'r') as reader:
        profile = reader.profile
        tif_as_array = reader.read()
    return tif_as_array, profile


def write_tiff(file_path, arr, profile):
    with rasterio.Env():
        with rasterio.open(file_path, 'w', **profile) as dst:
            dst.write(arr.astype(rasterio.float32))


def removeOutliers(x, outlierConstant):
    upper_quartile = np.percentile(x, 75)
    lower_quartile = np.percentile(x, 25)
    print(upper_quartile, lower_quartile)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    print(quartileSet)

    result = x*(x >= quartileSet[0])*(x <= quartileSet[1])

    return result

def dataset_gen():
    file_list = glob.glob('palsar/*/*/*.tif')
    file_list.sort()
    dataset_train_list = []
    dataset_test_list = []
    th=[(0.1, 1), (0.2, 1), (0.2, 1), (0.2, 1), (0.15, 0),
        (0.05, 1), (0.15, 0), (0.2, 1), (0.1, 0), (0.1, 1),
        (0.3, 0), (0.4, 1), (-0.2, 1),(-0.2, 1), (0.3, 1), (0.2, 1), (0.1, 1),
        (0.2, 1),
        (0.15, 1), (0.6, 1), (0.45, 1), (0.1, 1), (-0.1, 1),
        (0, 1), (0.3, 1)]
    for idx in range(len(file_list)):
        file_name = file_list[idx]
        print(file_name)
        tif_array, _ = read_tiff(file_name)

        _, size_x, size_y = tif_array.shape
        tif_array = tif_array.transpose((1, 2, 0))
        tif_array = np.nan_to_num(tif_array)
        data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], 4))
        for i in range(3):
            data_output[:,:,i] = removeOutliers(tif_array[:,:,i], 1)
        img = (tif_array[:,:,:3]-tif_array[:,:,:3].min())/(tif_array[:,:,:3].max()-tif_array[:,:,:3].min())
        plt.imshow(img)
        plt.show()
        if th[idx][1] == 1:
            data_output[:,:,3] = np.logical_and(tif_array[:,:,4]>th[idx][0], tif_array[:,:,3]>0)
        else:
            data_output[:, :, 3] = tif_array[:, :, 4] > th[idx][0]
        img = data_output[:,:,3]
        # plt.imshow(img)
        # plt.show()
        data_index_y = size_y // 256
        data_index_x = size_x // 256
        split_index = data_index_x * data_index_y * 0.8
        for i in range(data_index_x):
            for j in range(data_index_y):
                if i * data_index_y + j >= split_index:
                    dataset_test_list.append(data_output[i*256:(i+1)*256, j*256:(j+1)*256, :])
                else:
                    dataset_train_list.append(data_output[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :])

    dataset_train = np.stack(dataset_train_list, axis=0)
    dataset_test = np.stack(dataset_test_list, axis=0)
    np.save('dataset/proj2_train.npy', dataset_train)
    np.save('dataset/proj2_test.npy', dataset_test)
    return dataset_train, dataset_test

def dataset_eva_gen():
    file_list = glob.glob('palsar_evaluate/22994616_2020/*.tif')
    overlap = 128
    th = [(0.2, 0), (0.1, 1)]
    for idx in range(len(file_list)):
        file_name = file_list[idx]
        id = file_name[16:24]
        dataset_eva_list = []
        tif_array, _ = read_tiff(file_name)
        _, size_x, size_y = tif_array.shape
        tif_array = tif_array.transpose((1, 2, 0))
        tif_array = np.nan_to_num(tif_array)
        data_output = np.zeros((tif_array.shape[0], tif_array.shape[1], 4))
        img = np.zeros((tif_array.shape[0], tif_array.shape[1], 3))
        for i in range(3):
            data_output[:,:,i] = removeOutliers(tif_array[:,:,i], 1)
            img[:,:,i] = (tif_array[:,:,i]-tif_array[:,:,i].min())/(tif_array[:,:,i].max()-tif_array[:,:,i].min())
        plt.imshow(img)
        plt.show()
        if th[idx][1] == 1:
            data_output[:, :, 3] = np.logical_and(tif_array[:, :, 4] > th[idx][0], tif_array[:, :, 3] > 0)
        else:
            data_output[:, :, 3] = tif_array[:, :, 4] > th[idx][0]
        plt.title('c')
        plt.imshow(data_output[:,:,3])
        plt.savefig('label', bbox_inches='tight')
        plt.show()

        data_index_y = size_y // 128
        data_index_x = size_x // 128
        for i in range(data_index_x):
            for j in range(data_index_y):
                if (i * 128) + 256 < size_x and (j * 128) + 256 < size_y:
                    dataset_eva_list.append(data_output[i * 128 : (i * 128) + 256, j * 128 : (j * 128) + 256, :])
        dataset_eva = np.stack(dataset_eva_list, axis=0)
        np.save('dataset/proj2_evaluate'+id+str(data_index_x)+str(data_index_y)+'.npy', dataset_eva)

if __name__ == '__main__':
    # dataset_gen()
    dataset_eva_gen()