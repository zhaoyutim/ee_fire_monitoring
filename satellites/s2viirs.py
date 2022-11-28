import datetime
import os
from glob import glob

import pandas as pd
import yaml
import ee
from google.cloud import storage

with open("config/land_cover_id.yaml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("config/land_cover_id_eva.yaml", "r", encoding="utf8") as f:
    config_eva = yaml.load(f, Loader=yaml.FullLoader)


class s2viirs:
    def get_bbox(self, feature):
        return feature.bounds().buffer(2)

    def get_buffer(self, feature):
        return feature.buffer(1000)

    def get_ratio(self, img):
        b08 = img.select('B8')
        b11 = img.select('B11')
        b12 = img.select('B12')
        index = b08.subtract(b12).divide(b08.add(b12)).rename('index')
        return ee.Image.cat([b08, b11, b12, index])

    def download_to_gcloud(self, dataset='train'):
        elevation = ee.Image("USGS/GTOPO30").select('elevation')
        geoinfos=[]
        id_fire_list=[]
        lon_list=[]
        lat_list=[]
        elevation_list=[]
        land_cover_list = []
        if dataset=='train':
            land_covers = ['needle', 'broadleaf', 'shrublands', 'savannas', 'grasslands']
            fire_info = config
        else:
            land_covers = ['needle', 'broadleaf', 'shrublands', 'savannas', 'grasslands', 'mixed']
            fire_info = config_eva
        for land_cover in land_covers:
            id_list = ee.List(list(fire_info.get(land_cover).keys()))
            for y in range(2017,2021):
                polygons = ee.FeatureCollection('projects/ee-zhaoyutim/assets/globfire'+str(y)).filter(ee.Filter.inList('Id', id_list))
                polygons = polygons.map(self.get_buffer)
                polygons = polygons.filter(ee.Filter.eq('Type', 'FinalArea'))
                n = polygons.size().getInfo()
                for i in range(n):
                    poly = ee.Feature(polygons.toList(n).get(i))
                    bbox = poly.geometry().bounds().buffer(2)

                    centroid = bbox.centroid(maxError=1).getInfo()
                    ele_info = elevation.reduceRegion(ee.Reducer.mean(), bbox).getInfo()
                    id_fire = poly.getInfo().get('properties').get('Id')
                    print('polygon centroid =', centroid)
                    print(ele_info)


    def download_to_local(self, dataset='train', create_time='2022-06-18'):
        storage_client = storage.Client()

        bucket = storage_client.bucket('ai4wildfire')
        if dataset=='train':
            prefixs = ['palsar', 'palsar_s1']
            fire_info=config
        else:
            prefixs = ['palsar_eva', 'palsar_s1_eva']
            fire_info=config_eva
        for prefix in prefixs:
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                if blob.time_created.date() < datetime.datetime.strptime(create_time, '%Y-%m-%d').date():
                    continue
                filename = blob.name

                id = filename.split('/')[2][:8]
                land_cover = filename.split('/')[1]
                if int(id) in list(fire_info.get(land_cover).keys()):
                    path = os.path.dirname(filename)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    blob.download_to_filename(filename)
                    print(
                        "Blob {} downloaded to {}.".format(
                            filename, filename
                        )
                    )
