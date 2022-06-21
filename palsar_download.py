import os

import ee
from google.cloud import storage

from satellites.palsar import palsar
# ee.Authenticate()
ee.Initialize()

if __name__=='__main__':
    client = palsar()
    # client.download_to_gcloud('train')
    # client.download_to_local('train', '2022-06-18')