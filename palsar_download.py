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
    # gedi.download_to_gcloud(['sa', 'af', 'eu', 'sas', 'nas', 'au'])
    # gedi.download_to_local_proj4('2022-09-20')
    gedi.generate_dataset_proj4(['au', 'sas', 'nas'])
    # client.download_to_gcloud('eva')
    # client.download_to_local('eva', '2022-06-20')