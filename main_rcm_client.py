import ee

from dataset_gen import dataset_gen
from satellites.rcm import RCMClient
ee.Initialize()

if __name__=='__main__':
    # rcm = RCMClient()
    # for location in ['slave_lake', 'slave_lake3', 'slave_lake4', 'slave_lake5', 'donnie_creek', 'edmonton', 'fox_lake', 'rainbow_lake']:
    #     # rcm.download_to_gcloud(beammode='all', location=location, dataset='train')
    #     rcm.download_to_local()

    dataset_gen('rcm', nchannel=4)