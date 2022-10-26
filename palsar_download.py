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
    # custom_regions=[-121, 40, -123, 38] # xmin, ymin, xmax, ymax
    # gedi.download_to_gcloud(region_ids=['na'], year=2020)
    # gedi.download_to_local_proj4('2022-10-23')
    gedi.generate_dataset_proj4(['na'], year=2020, custom_region=custom_regions)
    # gedi.evaluate_and_plot('dataset/proj4_train_custom_region2020.npy')
    # gedi.inference()
    # client.download_to_gcloud('eva')
    # client.download_to_local('eva', '2022-06-20')