import os

import ee
from google.cloud import storage

from satellites.palsar import palsar
from satellites.gedi import gedi
# ee.Authenticate()
import os
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:15236'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:15236'
ee.Initialize()

if __name__=='__main__':
    client = palsar()
    gedi = gedi()
    custom_regions=None
    rb = 0.5
    # custom_regions=[-122.5,39.5,-123.5,38.5] # xmin, ymin, xmax, ymax
    # ['na', 'sa', 'af', 'eu', 'au', 'sas', 'nas']
    # gedi.download_to_gcloud(region_ids=['na'], mode='test', year=2020, custom_region=custom_regions)
    # gedi.download_to_local_proj4('2023-01-04')
    # gedi.generate_dataset_proj4(['na'], year=2020, random_blind=True, custom_region=custom_regions, mode='test', rb=rb)
    # gedi.evaluate_and_plot_train(train_array_path='proj4_gedi_palsar_NA2020_year2020class_DBT_NA.tif')
    # gedi.evaluate_and_plot_rh('dataset/proj4_train_na2020test.npy', nchannels=2)
    gedi.evaluate_and_plot('dataset/proj4_train_na2020test'+str(rb)+'.npy', model_path='/Users/zhaoyu/Desktop/proj4_unet_pretrained_resnet18_nchannels_', nchannels=9)
    # gedi.inference(path='proj4_gedi_palsar/CUSTOM_REGION2020/*.tif', random_blind=True, overlap=32, nchannels=9)
    # client.download_to_gcloud('eva')
    # client.download_to_local('eva', '2022-06-20')