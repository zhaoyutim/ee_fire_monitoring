import os

import ee
from google.cloud import storage

from satellites.palsar import palsar

ee.Initialize()

if __name__=='__main__':
    client = palsar()
    client.download_to_gcloud()
    # client.download_to_gcloud_evaluate()
    # client.download_to_local()