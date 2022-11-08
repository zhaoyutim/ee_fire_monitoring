import argparse

from GEDIClient.GEDIClient import GEDIClient

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-id', type=int, help='Model to be executed')
    args = parser.parse_args()
    id = args.id
    gedi_client=GEDIClient()
    gedi_client.visualize_geojson('gedi_sample_json/custom_region_json.geojson')
    file_url_list = gedi_client.query_with_json('gedi_sample_json/custom_region_json.geojson', id)
    gedi_client.download('gedi_sample_json/custom_region_json.geojson', file_url_list, id)
    # gedi_client.concatcsv('subsets/*.csv')
    # gedi_client.csv_to_tiff('subsets/aca_gedi_l4a0.csv')

