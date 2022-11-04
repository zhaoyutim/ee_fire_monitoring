from GEDIClient.GEDIClient import GEDIClient

if __name__=='__main__':
    gedi_client=GEDIClient()
    # gedi_client.visualize_geojson('gedi_sample_json/na_json.geojson')
    # file_url_list = gedi_client.query_with_json('gedi_sample_json/na_json.geojson')
    # gedi_client.download('gedi_sample_json/na_json.geojson', file_url_list)
    gedi_client.concatcsv('subset/*.csv')