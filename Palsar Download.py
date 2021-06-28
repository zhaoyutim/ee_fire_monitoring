import os

import ee
from google.cloud import storage

from satellites.palsar import Palsar

ee.Initialize()

if __name__=='__main__':
    client = Palsar()
    # client.download_to_gcloud()
    client.download_to_local()