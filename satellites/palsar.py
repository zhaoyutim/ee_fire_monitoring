import datetime
import os
from glob import glob

import yaml
import ee
from google.cloud import storage

with open("dataset_config/land_cover_id.yaml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("dataset_config/land_cover_id_eva.yaml", "r", encoding="utf8") as f:
    config_eva = yaml.load(f, Loader=yaml.FullLoader)


class palsar:
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
                    pre_fire_end = datetime.datetime.fromtimestamp(poly.get('IDate').getInfo() / 1000)
                    fire_year = pre_fire_end.year
                    pre_fire_start = pre_fire_end - datetime.timedelta(weeks=12)

                    pre_fire_median = ee.ImageCollection('COPERNICUS/S2')\
                        .filterDate(pre_fire_start.strftime('%Y-%m-%d'), pre_fire_end.strftime('%Y-%m-%d'))\
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 100))\
                        .filterBounds(bbox).median()
                    pre_fire_median = self.get_ratio(pre_fire_median)

                    pre_fire_median_s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                        .filterDate(pre_fire_start.strftime('%Y-%m-%d'), pre_fire_end.strftime('%Y-%m-%d')) \
                        .filterBounds(bbox).select(['VV', 'VH']).median()

                    pre_year = fire_info.get(land_cover).get(poly.getInfo().get('properties').get('Id')).get('pre')
                    post_year = fire_info.get(land_cover).get(poly.getInfo().get('properties').get('Id')).get('post')
                    # PARSAR dataset
                    post_img = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR').filter(ee.Filter.date(str(post_year), str(post_year + 1)))
                    pre_img = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR').filter(ee.Filter.date(str(pre_year), str(pre_year + 1)))

                    after_date = post_img.select('date').first().reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=bbox,
                            scale=25,
                            maxPixels=1e9
                        )
                    if post_year >= 2018:
                        after_date = ee.Date.fromYMD(2014, 6, 1).advance(ee.Number(after_date.get('date')).floor(), 'day').getInfo()
                    else:
                        after_date = ee.Date.fromYMD(1970, 1, 1).advance(ee.Number(after_date.get('date')).floor(), 'day').getInfo()
                    after_date = datetime.datetime.fromtimestamp(after_date.get('value') / 1000)
                    print(after_date, pre_fire_end, i, land_cover)
                    if after_date < pre_fire_end:
                        if post_year == fire_year:
                            continue
                        else:
                            break
                    else:
                        after_img = ee.ImageCollection('COPERNICUS/S2') \
                            .filterDate(pre_fire_end.strftime('%Y-%m-%d'), after_date.strftime('%Y-%m-%d')) \
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                            .filterBounds(bbox).median()

                        after_img = self.get_ratio(after_img)

                        after_img_s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                            .filterDate(pre_fire_end.strftime('%Y-%m-%d'), after_date.strftime('%Y-%m-%d')) \
                            .filterBounds(bbox).select(['VV', 'VH']).median()

                        dnbr = pre_fire_median.select('index').subtract(after_img.select('index')).rename('dnbr')

                        sarHh_log_pre = pre_img.select('HH').first().pow(2).log10().multiply(10).subtract(83).rename('HH_pre')
                        sarHv_log_pre = pre_img.select('HV').first().pow(2).log10().multiply(10).subtract(83).rename('HV_pre')
                        sarhvhh_pre = sarHv_log_pre.subtract(sarHh_log_pre)

                        sarHh_log = post_img.select('HH').first().pow(2).log10().multiply(10).subtract(83).rename('HH_post')
                        sarHv_log = post_img.select('HV').first().pow(2).log10().multiply(10).subtract(83).rename('HV_post')
                        sarHvhh = sarHv_log.subtract(sarHh_log)

                        logrt_HH = sarHh_log_pre.subtract(sarHh_log)
                        logrt_HV = sarHv_log_pre.subtract(sarHv_log)
                        logrtHvhh = sarHvhh.subtract(sarhvhh_pre).rename('HV-HH')

                        landAreaImg = polygons.reduceToImage(properties=['Id'], reducer=ee.Reducer.first()).rename('polygons')

                        composite = ee.Image([sarHh_log_pre, sarHv_log_pre, sarHh_log, sarHv_log, landAreaImg.gt(0), dnbr, logrt_HH, logrt_HV, logrtHvhh])
                        composite_s1 = ee.Image([pre_fire_median_s1, after_img_s1])
                        id = str(poly.get('Id').getInfo())
                        if dataset=='train':
                            dir = 'palsar_s1' + '/' + land_cover + '/' + id + '_' + str(post_year) + '/' + str(post_year)
                        else:
                            dir = 'palsar_s1_eva' + '/' + land_cover + '/' + id + '_' + str(post_year) + '/' + str(post_year)
                        image_task = ee.batch.Export.image.toCloudStorage(
                            image=composite_s1.toFloat(),
                            description='Image Export:' + land_cover + '_' + id + '_' + str(post_year)+'S1',
                            fileNamePrefix=dir,
                            bucket='ai4wildfire',
                            scale=25,
                            maxPixels=1e11,
                            region=bbox,
                        )
                        image_task.start()
                        if dataset=='train':
                            print('Start with image task (id: {}).'.format('S1 Image Export:' + land_cover + '_' + id + '_' + str(post_year)))
                            dir = 'palsar' + '/' + land_cover + '/' + id + '_' + str(post_year) + '/' + str(post_year)
                        else:
                            print('Start with image task (id: {}).'.format('S1 Image Eva Export:' + land_cover + '_' + id + '_' + str(post_year)))
                            dir = 'palsar_eva' + '/' + land_cover + '/' + id + '_' + str(post_year) + '/' + str(post_year)
                        image_task = ee.batch.Export.image.toCloudStorage(
                            image=composite.toFloat(),
                            description='Image Export:' + land_cover + '_' + id + '_' + str(post_year),
                            fileNamePrefix=dir,
                            bucket='ai4wildfire',
                            scale=25,
                            maxPixels=1e11,
                            region=bbox,
                        )
                        image_task.start()
                        if dataset=='train':
                            print('Start with image task (id: {}).'.format('Palsar Image Export:' + land_cover + '_' + id + '_' + str(post_year)))
                        else:
                            print('Start with image task (id: {}).'.format('Palsar Image Eva Export:' + land_cover + '_' + id + '_' + str(post_year)))

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