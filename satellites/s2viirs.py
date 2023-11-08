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
                    end_date = polygons.first().get('FDate')
                    end_date = ee.Date(ee.Date.fromYMD(1970,1,1).advance(ee.Number(end_date).divide(1000), 'second'))
                    start_date = polygons.first().get('IDate')
                    start_date = ee.Date(ee.Date.fromYMD(1970,1,1).advance(ee.Number(start_date).divide(1000), 'second'))

                    # centroid = bbox.centroid(maxError=1).getInfo()
                    # ele_info = elevation.reduceRegion(ee.Reducer.mean(), bbox).getInfo()
                    id_fire = str(poly.getInfo().get('properties').get('Id'))
                    # print('polygon centroid =', centroid)
                    # print(ele_info)
                    s2_pre = ee.ImageCollection('COPERNICUS/S2_SR')\
                        .filterDate(start_date.advance(-30, 'Day'), start_date)\
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
                        .filterBounds(polygons)\
                        .median().select('B12', 'B8A', 'B4')\


                    viirs_pre = ee.ImageCollection("NOAA/VIIRS/001/VNP09GA")\
                        .filterDate(start_date.advance(-10, 'Day'), start_date)\
                        .filterBounds(polygons)\
                        .median() \
                        .select('M11', 'I2', 'I1')

                    s2_post = ee.ImageCollection('COPERNICUS/S2_SR')\
                        .filterDate(end_date, end_date.advance(30, 'Day'))\
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
                        .filterBounds(polygons)\
                        .median().select('B12', 'B8A', 'B4')\


                    viirs_post = ee.ImageCollection("NOAA/VIIRS/001/VNP09GA")\
                        .filterDate(end_date, end_date.advance(10, 'Day'))\
                        .filterBounds(polygons)\
                        .median() \
                        .select('M11', 'I2', 'I1')
                    polygons_img = polygons.reduceToImage(properties=['Id'], reducer=ee.Reducer.first()).rename('polygons')
                    output_pre = ee.Image([s2_pre, viirs_pre, polygons_img])
                    output_post = ee.Image([s2_post, viirs_post, polygons_img])
                    if dataset=='train':
                        dir_pre = 's2_viirs_train_pre' + '/' + land_cover + '/' + id_fire
                        dir_post = 's2_viirs_train_post' + '/' + land_cover + '/' + id_fire
                    else:
                        dir_pre = 's2_viirs_test_pre' + '/' + land_cover + '/' + id_fire
                        dir_post = 's2_viirs_test_post' + '/' + land_cover + '/' + id_fire
                    image_task = ee.batch.Export.image.toCloudStorage(
                        image=output_pre.toFloat(),
                        description='Image Export:' + land_cover + '_' + id_fire + '_' +'pre',
                        fileNamePrefix=dir_pre,
                        bucket='ai4wildfire',
                        scale=20,
                        maxPixels=1e11,
                        region=bbox,
                    )
                    image_task.start()
                    print('Start with image task (id: {}).'.format(
                        'S2_VIIRS Image Export:' + land_cover + '_' + id_fire + '_' +'pre'))
                    image_task = ee.batch.Export.image.toCloudStorage(
                        image=output_post.toFloat(),
                        description='Image Export:' + land_cover + '_' + id_fire + '_' +'post',
                        fileNamePrefix=dir_post,
                        bucket='ai4wildfire',
                        scale=20,
                        maxPixels=1e11,
                        region=bbox,
                    )
                    image_task.start()
                    print('Start with image task (id: {}).'.format(
                        'S2_VIIRS Image Export:' + land_cover + '_' + id_fire + '_' +'post'))
    def download_to_local(self, dataset='train', create_time='2023-01-29'):
        storage_client = storage.Client()
        bucket = storage_client.bucket('ai4wildfire')
        if dataset=='train':
            land_covers = ['needle', 'broadleaf', 'shrublands', 'savannas', 'grasslands']
        else:
            land_covers = ['needle', 'broadleaf', 'shrublands', 'savannas', 'grasslands', 'mixed']
        for mode in ['pre', 'post']:
            for land_cover in land_covers:
                blobs = bucket.list_blobs(prefix='s2_viirs_train_'+mode+'/'+land_cover)
                for blob in blobs:
                    if blob.time_created.date() < datetime.datetime.strptime(create_time, '%Y-%m-%d').date():
                        continue
                    filename = blob.name
                    path = os.path.dirname(filename)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    blob.download_to_filename(filename)
                    print(
                        "Blob {} downloaded to {}.".format(
                            filename, filename
                        )
                    )
