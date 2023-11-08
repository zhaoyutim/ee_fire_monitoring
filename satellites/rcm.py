import datetime
import os
from glob import glob

import pandas as pd
import yaml
import ee
from google.cloud import storage

with open("config/rcm_config.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class RCMClient:

    def download_to_gcloud(self, beammode, location, dataset='train'):
        def get_mask(img):
            return img.updateMask(img.gt(0))

        def get_log(img):
            return ee.Image(
                [img.select('b1').multiply(100), img.select('b2').multiply(100), img.select('b3').multiply(100)]).set(
                'date', ee.String(img.get('system:index')).slice(36, 51)).copyProperties(img, img.propertyNames())
        roi = config[location]['roi']
        th = config[location]['th']
        pre = config[location]['pre'].strftime('%Y-%m-%d')
        start = config[location]['start'].strftime('%Y-%m-%d')
        end = config[location]['end'].strftime('%Y-%m-%d')
        rcm_col = ee.ImageCollection("projects/rcm-data/assets/rcm-data").sort('system:time_start').map(get_mask)\
            .map(get_log).filterBounds(ee.Geometry.Rectangle(config[location]['roi']))
        if beammode in ['SC30MCPA', 'SC30MCPC', 'SC30MCPB', 'SC30MCPD']:
            rcm_col = rcm_col.filter(ee.Filter.stringContains("system:index", beammode))
        pre = rcm_col.filterDate(pre, start).median()
        rcm_col = rcm_col.filterDate(start, end)

        def get_logrt(img):
            return img.subtract(pre)

        def get_extraband(img):
            return img.addBands(img.select('b1').add(img.select('b3')).rename('b4'))

        def get_rt_mask(img):
            return img.addBands(img.select('b1').gt(img.select('b3')).And(img.select('b4').gt(th)).rename('b5'))

        rcm_log_ratio = rcm_col.map(get_logrt).map(get_extraband).map(get_rt_mask)

        img_list = rcm_log_ratio.toList(rcm_log_ratio.size())
        size = rcm_log_ratio.size().getInfo()
        if dataset=='train':
            dir = 'rcm_train' + '/' + location + '/'
        else:
            dir = 'rcm_test' + '/' + location + '/'
        for i in range(size):
            img = ee.Image(img_list.get(i))
            imgid = img.get('system:index').getInfo()
            image_task = ee.batch.Export.image.toCloudStorage(
                image=img.toFloat(),
                description='Image Export:' + location + imgid,
                fileNamePrefix=dir+imgid,
                bucket='ai4wildfire',
                scale=30,
                maxPixels=1e11,
                region=roi,
                crs='EPSG:32610'
            )
            image_task.start()
            print('Start with image task (id: {}).'.format(imgid))


    def download_to_local(self, dataset='train', create_time='2022-06-18'):
        storage_client = storage.Client()

        bucket = storage_client.bucket('ai4wildfire')
        if dataset=='train':
            prefix = 'rcm_train'
        else:
            prefix = 'rcm_test'
        blobs = bucket.list_blobs(prefix=prefix)
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
