import ee
from satellites.s2viirs import s2viirs
ee.Initialize()

if __name__=='__main__':
    s2viirs = s2viirs()
    s2viirs.download_to_gcloud('train')
    s2viirs.download_to_local()