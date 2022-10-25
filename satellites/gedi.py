import datetime
import os
from glob import glob

import ee
import numpy as np
import rasterio
import yaml
from google.cloud import storage
from matplotlib import pyplot as plt

from ParamsFetching import ParamsFetching
from run_cnn_model_gedi import create_model

with open("config/sample.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class gedi:
    def download_to_gcloud(self, region_ids=['na'], year = 2019, custom_region=None):
        dataset = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH') \
            .filter(ee.Filter.date(str(year)+'-01-01', str(year + 1)+'-01-01'))
        sarHh_log = dataset.select('HH').first().pow(2).log10().multiply(10).subtract(83)
        sarHv_log = dataset.select('HV').first().pow(2).log10().multiply(10).subtract(83)
        sarhvhh = sarHv_log.subtract(sarHh_log).rename('HV-HH')
        composite = ee.Image([sarHh_log, sarHv_log, sarhvhh])
        lc = ee.ImageCollection("ESA/WorldCover/v100").first()
        l4b = ee.Image('LARSE/GEDI/GEDI04_B_002').select(['PS', 'MU'])
        if custom_region == None:
            for region_id in region_ids:
                roi_col = ee.FeatureCollection('users/zhaoyutim/GEDI_SAMPLE_'+region_id.upper())
                size = roi_col.size().getInfo()
                roi_col = roi_col.toList(size)
                for i in range(size):
                    roi = ee.Feature(roi_col.get(i).getInfo())
                    class_id = roi.args['metadata'].get('class')
                    date_pre = dataset.select('epoch').median().clip(roi).reduceRegion(
                        reducer=ee.Reducer.max(),
                        geometry=roi.geometry(),
                        scale=1000,
                        crs='EPSG:4326').getNumber('epoch')
                    date_pre = ee.Date(ee.Date.fromYMD(1970,1,1).advance(ee.Number(date_pre).divide(1000), 'second'))
                    gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')\
                        .filterDate(date_pre.difference(15, "day"), date_pre.advance(15, "day"))\
                        .map(self.qualityMask)\
                        .select(['rh40', 'rh50', 'rh60', 'rh70', 'rh98']).mosaic()
                    output = ee.Image([composite, lc, l4b, gedi])
                    dir = 'proj4_gedi_palsar' + '/' + region_id.upper() + str(year) + '/' + 'year'+ str(year)+ 'class_' + class_id + '_' + str(i)
                    image_task = ee.batch.Export.image.toCloudStorage(
                        image=output.toFloat(),
                        description='Image Export:' + 'GEDI_PALSAR_' + region_id.upper() + str(year)+'_CLASS_'+class_id,
                        fileNamePrefix=dir,
                        bucket='ai4wildfire',
                        scale=25,
                        maxPixels=1e11,
                        region=roi.geometry(),
                        fileDimensions=256*5
                    )
                    image_task.start()
                    print('Start with image task (id: {}).'.format(
                        'GEDI-PALSAR Image Export:' + 'GEDI_SAMPLE_'+region_id.upper() + str(year)+'_INDEX_'+str(i)+'_CLASS_'+class_id))
        else:
            region = ee.Geometry.Rectangle(custom_region)
            roi = ee.Feature(region)
            date_pre = dataset.select('epoch').median().clip(roi).reduceRegion(
                reducer=ee.Reducer.max(),
                geometry=roi.geometry(),
                scale=1000,
                crs='EPSG:4326').getNumber('epoch')
            date_pre = ee.Date(ee.Date.fromYMD(1970,1,1).advance(ee.Number(date_pre).divide(1000), 'second'))
            gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')\
                .filterDate(date_pre.difference(15, "day"), date_pre.advance(15, "day"))\
                .map(self.qualityMask)\
                .select(['rh40', 'rh50', 'rh60', 'rh70', 'rh98']).mosaic()
            output = ee.Image([composite, lc, l4b, gedi])
            dir = 'proj4_gedi_palsar' + '/' + 'custom_region' + str(year) + '/' + 'year'+ str(year)
            image_task = ee.batch.Export.image.toCloudStorage(
                image=output.toFloat(),
                description='Image Export:' + 'GEDI_PALSAR_' + 'custom_region' + str(year),
                fileNamePrefix=dir,
                bucket='ai4wildfire',
                scale=25,
                maxPixels=1e11,
                region=roi.geometry(),
                fileDimensions=256*5
            )
            image_task.start()
            print('Start with image task (id: {}).'.format(
                'GEDI-PALSAR Image Export:' + 'GEDI_SAMPLE_' + 'custom_region' + str(year)))

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

    def combine_images(self, array, desired_shape, scale_factor):
        curent_shape = array.shape[1]
        if array.shape[1] != desired_shape // scale_factor:
            raise Exception('Invalid shape')
        new_array = np.zeros((desired_shape, desired_shape, array.shape[3]))
        for i in range(scale_factor):
            for j in range(scale_factor):
                new_array[curent_shape*i:curent_shape*(i+1), curent_shape*j:curent_shape*(j+1), :] = array[i*scale_factor+j, :, :, :]
        return new_array

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


    def generate_dataset_proj4(self, region_ids = ['na', 'sa', 'af', 'eu', 'au', 'sas', 'nas'], year=2020, custom_region=None):
        params_fetching = ParamsFetching()
        if custom_region!=None:
            region_ids=['custom_region']
        for region_id in region_ids:
            if year == 2019:
                path = os.path.join('proj4_gedi_palsar', region_id.upper(), '*.tif')
            else:
                path = os.path.join('proj4_gedi_palsar', region_id.upper()+str(year), '*.tif')
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
                output_array[:, :, 4:9] = array[:, :, 6:]
                output_array[:, :, 9] = np.where(agbd!=-1, np.nan_to_num(array[:, :, 5]/100, nan=-1), -1)
                # output_array[:, :, 8] = agbd
                output_array[:, :, 3] = array[:, :, 3]
                if np.nanmean(output_array[:, :, 8])==-1:
                    continue
                output_array = self.slice_into_small_tiles(output_array, 20)
                dataset_list.append(output_array)
                index += 25
                if index % 1000==0:
                    # break
                    print('{:.2f}% completed'.format(index*100/len(file_list)/25))

            dataset = np.concatenate(dataset_list, axis=0)
            if year != 2019:
                np.save('dataset/proj4_train_'+region_id+str(year)+'.npy', dataset)
            else:
                np.save('dataset/proj4_train_' + region_id + '.npy', dataset)

    def evaluate_and_plot(self, test_array_path='dataset/proj4_train_na2020.npy', model_path='model/proj4_unet_pretrained_resnet18/', nchannels=9):
        import segmentation_models as sm
        region_id='custom_region'
        sm.set_framework('tf.keras')
        test_array= np.load(test_array_path)
        if not os.path.exists('dataset_pred/'+region_id+'agbd_resnet18_unet.npy'):
            model = create_model('unet', 'resnet18', 0.0003, nchannels=nchannels)
            model.load_weights(model_path)
            agbd_pred = model.predict(test_array[:, :, :, :nchannels])
            np.save('dataset_pred/'+region_id+'agbd_resnet18_unet.npy', agbd_pred)
        else:
            agbd_pred = np.load('dataset_pred/'+region_id+'agbd_resnet18_unet.npy')
        agbd = test_array[:,:,:,[9]]
        x_scatter = agbd[np.squeeze(agbd) != -1]
        y_scatter = agbd_pred[np.squeeze(agbd) != -1]
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_scatter.flatten(), y_scatter.flatten())
        plt.title('Correlation between predicted AGBD and GEDI AGBD. r-squared: {0:.2f}'.format(r_value ** 2))
        plt.scatter(x=x_scatter * 100, y=y_scatter * 100, c='g')
        plt.xlabel("agdb_groundtruth")
        plt.ylabel("agbd_predicted")
        plt.show()

    def inference(self, path='proj4_gedi_palsar/CUSTOM_REGION2020/*.tif', model_path='model/proj4_unet_pretrained_resnet18/'):
        file_list=glob(path)
        import segmentation_models as sm
        sm.set_framework('tf.keras')
        model = create_model('unet', 'resnet18', 0.0003)
        model.load_weights(model_path)
        for file_dir in file_list:
            array, pf = self.read_tiff(file_dir)
            if array.shape[0] != 1280 or array.shape[1] != 1280 or array.shape[2] != 11:
                print('invalid shape')
                continue
            input = self.slice_into_small_tiles(array, 20)
            agbd_pred = model.predict(input[:,:,:,:3])
            agbd_pred = self.combine_images(agbd_pred, 1280, 20)
            pf.data['count'] = 1
            self.write_tiff(file_dir.replace('proj4_gedi_palsar', 'recon'), agbd_pred.transpose((2,0,1)), pf)
            print('successfully reconstruct agbd predicted')

