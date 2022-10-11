import datetime
import os
import time
from glob import glob

import numpy as np
import rasterio
import yaml
import ee
from google.cloud import storage
from matplotlib import pyplot as plt

from ParamsFetching import ParamsFetching

with open("config/sample.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class gedi:
    def download_to_gcloud(self, region_ids=['na'], dataset='train'):
        year = 2019
        dataset = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR') \
            .filter(ee.Filter.date(str(year), str(year + 1)))
        sarHh_log = dataset.select('HH').first().pow(2).log10().multiply(10).subtract(83)
        sarHv_log = dataset.select('HV').first().pow(2).log10().multiply(10).subtract(83)
        sarhvhh = sarHv_log.subtract(sarHh_log).rename('HV-HH')
        composite = ee.Image([sarHh_log, sarHv_log, sarhvhh])
        fnf = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/FNF').filter(ee.Filter.date(str(year), str(year + 1)))
        fnf =  fnf.select('fnf').mosaic()
        l4b = ee.Image('LARSE/GEDI/GEDI04_B_002').select(['PS', 'MU'])
        for region_id in region_ids:
            roi_col = ee.FeatureCollection('users/zhaoyutim/GEDI_SAMPLE_'+region_id.upper())
            size = roi_col.size().getInfo()
            roi_col = roi_col.toList(size)
            for i in range(size):
                roi = ee.Feature(roi_col.get(i).getInfo())
                class_id = roi.args['metadata'].get('class')
                date_pre = dataset.select('date').median().clip(roi).reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=roi.geometry(),
                    scale=100,
                    crs='EPSG:4326').getNumber('date')
                date_pre = ee.Date(ee.Date.fromYMD(2014, 5, 24).advance(date_pre, 'day'))
                gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')\
                    .filterDate(date_pre.difference(15, "day"), date_pre.advance(15, "day"))\
                    .map(self.qualityMask)\
                    .select(['rh40', 'rh50', 'rh60', 'rh70', 'rh98']).mosaic()
                output = ee.Image([composite, fnf, l4b, gedi])
                dir = 'proj4_gedi_palsar' + '/' + region_id.upper() + '/' + 'class_' + class_id + '_' + str(i)
                image_task = ee.batch.Export.image.toCloudStorage(
                    image=output.toFloat(),
                    description='Image Export:' + 'GEDI_PALSAR_' + region_id.upper()+'_CLASS_'+class_id,
                    fileNamePrefix=dir,
                    bucket='ai4wildfire',
                    scale=25,
                    maxPixels=1e11,
                    region=roi.geometry(),
                    fileDimensions=256*5
                )
                image_task.start()
                print('Start with image task (id: {}).'.format(
                    'GEDI-PALSAR Image Export:' + 'GEDI_SAMPLE_'+region_id.upper()+'_INDEX_'+str(i)+'_CLASS_'+class_id))

    def qualityMask(self, img):
        return img.updateMask(img.select('quality_flag').eq(1)).updateMask(img.select('degrade_flag').eq(0))

    def download_to_local_proj4(self, create_time='2022-06-18'):
        storage_client = storage.Client()
        bucket = storage_client.bucket('ai4wildfire')
        blobs = bucket.list_blobs(prefix='proj4_gedi_palsar')
        for blob in blobs:
            if blob.time_created.date() < datetime.datetime.strptime(create_time, '%Y-%m-%d').date():
                continue
            filename = blob.name
            path = os.path.dirname(filename)
            if not os.path.exists(path):
                os.makedirs(path)
            blob.download_to_filename(filename)
            print(
                "Blob {} downloaded to {}.".format(
                    filename, filename
                )
            )

    def read_tiff(self, file_path):
        with rasterio.open(file_path, 'r') as reader:
            profile = reader.profile
            tif_as_array = reader.read().astype(np.float32).transpose((1,2,0))
        return tif_as_array, profile

    def write_tiff(self, file_path, arr, profile):
        with rasterio.Env():
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(arr.astype(rasterio.float32))
    def slice_into_small_tiles(self, array, division_factor, concat=False):
        shape = array.shape[0]
        new_shape = shape//division_factor
        new_array = []
        for i in range(division_factor):
            for j in range(division_factor):
                piece = array[new_shape*i:new_shape*(i+1), new_shape*j:new_shape*(j+1), :]
                # plt.imshow(piece[:,:,8])
                # plt.show()
                if np.nanmean(piece[:,:,8])==-1.0:
                    continue
                new_array.append(piece)
        if concat==False:
            array = np.stack(new_array, axis=0)
        else:
            array = np.concatenate(new_array, axis=0)
        return array

    def remove_outliers(self, x, outlierConstant):
        upper_quartile = np.percentile(x, 75)
        lower_quartile = np.percentile(x, 25)
        # print(upper_quartile, lower_quartile)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        # print(quartileSet)

        result = x * (x >= quartileSet[0]) * (x <= quartileSet[1])

        return result

    def standardization(self, x):
        # scaler = preprocessing.StandardScaler().fit(x)
        # x = scaler.transform(x)
        x = (x - x.mean()) / x.std()
        return x

    def normalization(self, x):
        return 255 * (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


    def generate_dataset_proj4(self, region_ids = ['na', 'sa', 'af', 'eu', 'au', 'sas', 'nas']):
        params_fetching = ParamsFetching()
        for region_id in region_ids:
            path = os.path.join('proj4_gedi_palsar', region_id.upper(), '*.tif')
            file_list = glob(path)
            dataset_list = []
            print('region_id:', region_id)
            index=0
            for file in file_list:
                array, _ = self.read_tiff(file)
                if array.shape[0]!=1280 or array.shape[1]!=1280 or array.shape[2]!=11:
                    continue

                output_array = np.zeros((array.shape[0], array.shape[1], 10)).astype(np.float32)
                print(index)
                agbd = params_fetching.get_agbd(array[:, :, 4:])
                for i in range(3):
                    output_array[:, :, i] = self.remove_outliers(array[:, :, i], 1)
                    output_array[:, :, i] = np.nan_to_num(output_array[:, :, i])
                output_array[:, :, 3:8] = array[:, :, 6:]
                output_array[:, :, 8] = agbd
                output_array[:, :, 9] = array[:, :, 5]
                if np.nanmean(output_array[:, :, 8])==-1:
                    continue
                output_array = self.slice_into_small_tiles(output_array, 20)
                dataset_list.append(output_array)
                # img = np.zeros((64,64,3))
                # for i in range(3):
                #     img[:,:,i] = (output_array[0,:,:,i]-output_array[0,:,:,i].min())/(output_array[0,:,:,i].max()-output_array[0,:,:,i].min())
                # plt.imshow(img)
                # plt.show()
                # plt.imshow(output_array[0,:,:,8])
                # plt.show()
                index += 25
                if index % 1000==0:
                    # break
                    print('{:.2f}% completed'.format(index*100/len(file_list)))
            dataset = np.concatenate(dataset_list, axis=0)
            np.save('dataset/proj4_train_'+region_id+'.npy', dataset)

    def evaluate_and_plot(self, test_array_path='proj4_gedi_palsar/NA/class_DBT_NA_00000000000-0000000000.tif'):
        params_fetching = ParamsFetching()
        fnf=2
        test_array, _ = self.read_tiff(test_array_path)
        # for i in range(3):
        #     test_array[:, :, i] = self.remove_outliers(test_array[:, :, i], 1)
        #     test_array[:, :, i] = np.nan_to_num(self.normalization(test_array[:, :, i]))
        test_array = self.slice_into_small_tiles(test_array, 5)
        agbd = params_fetching.get_agbd(test_array[:, :, :, 4:])
        x_scatter = agbd[np.logical_and(agbd!=-1, np.logical_or(test_array[:, :, :, 3]==1, test_array[:, :, :, 3]==2))]
        y_scatter = test_array[:,:,:,2][np.logical_and(agbd!=-1, np.logical_or(test_array[:, :, :, 3]==1, test_array[:, :, :, 3]==2))]
        # plt.axis('off')
        plt.scatter(x=x_scatter, y=y_scatter)
        plt.show()

