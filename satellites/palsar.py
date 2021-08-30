import datetime
import os
import yaml
import ee
from google.cloud import storage

with open("land_cover_id.yaml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("land_cover_id_eva.yaml", "r", encoding="utf8") as f:
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

    def download_to_gcloud(self):
        land_covers = ['grasslands']
        for land_cover in land_covers:
            id_list = ee.List(config.get(land_cover))
            polygons = ee.FeatureCollection('JRC/GWIS/GlobFire/v2/FinalPerimeters').filter(ee.Filter.inList('Id', id_list))
            # bbox = polygons.map(self.get_bbox)
            polygons = polygons.map(self.get_buffer)
            polygons = polygons.filterMetadata('area', 'not_less_than', 50000000).filterMetadata('InitialDate',
                                                                                                 'not_less_than',
                                                                                                 1490030000000)

            n = polygons.size().getInfo()
            for i in range(n):
                poly = ee.Feature(polygons.toList(n).get(i))
                bbox = poly.geometry().bounds().buffer(2)
                pre_fire_end = datetime.datetime.fromtimestamp(poly.get('InitialDate').getInfo() / 1000)
                fire_year = pre_fire_end.year
                pre_fire_start = pre_fire_end - datetime.timedelta(weeks=12)
                after_fire_end = pre_fire_end + datetime.timedelta(weeks=8)

                pre_fire_median = ee.ImageCollection('COPERNICUS/S2')\
                    .filterDate(pre_fire_start.strftime('%Y-%m-%d'), pre_fire_end.strftime('%Y-%m-%d'))\
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 100))\
                    .filterBounds(bbox).median()
                pre_fire_median = self.get_ratio(pre_fire_median)


                for year in range(fire_year, min(fire_year+2, 2021)):
                    # PARSAR dataset
                    dataset = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR').filter(ee.Filter.date(str(year), str(year + 1)))
                    pre_img = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR').filter(ee.Filter.date(str(year - 1), str(year)))

                    after_date = dataset.select('date').first().reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=bbox,
                            scale=25,
                            maxPixels=1e9
                        )
                    if year >= 2018:
                        after_date = ee.Date.fromYMD(2014, 6, 1).advance(ee.Number(after_date.get('date')).floor(), 'day').getInfo()
                    else:
                        after_date = ee.Date.fromYMD(1970, 1, 1).advance(ee.Number(after_date.get('date')).floor(), 'day').getInfo()
                    after_date = datetime.datetime.fromtimestamp(after_date.get('value') / 1000)
                    print(after_date, pre_fire_end, i, land_cover)
                    if after_date < pre_fire_end:
                        if year == fire_year:
                            continue
                        else:
                            break
                    else:
                        after_img = ee.ImageCollection('COPERNICUS/S2') \
                            .filterDate(pre_fire_end.strftime('%Y-%m-%d'), after_date.strftime('%Y-%m-%d')) \
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                            .filterBounds(bbox).mean()
                        after_img = self.get_ratio(after_img)

                        dnbr = pre_fire_median.select('index').subtract(after_img.select('index')).rename('dnbr')

                        sarHh_log_pre = pre_img.select('HH').first().pow(2).log10().multiply(10).subtract(83)
                        sarHv_log_pre = pre_img.select('HV').first().pow(2).log10().multiply(10).subtract(83)
                        sarhvhh_pre = sarHv_log_pre.subtract(sarHh_log_pre)

                        sarHh_log = dataset.select('HH').first().pow(2).log10().multiply(10).subtract(83)
                        sarHv_log = dataset.select('HV').first().pow(2).log10().multiply(10).subtract(83)
                        sarHvhh = sarHv_log.subtract(sarHh_log)

                        logrt_HH = sarHh_log_pre.subtract(sarHh_log)
                        logrt_HV = sarHv_log_pre.subtract(sarHv_log)
                        logrtHvhh = sarHvhh.subtract(sarhvhh_pre).rename('HV-HH')

                        landAreaImg = polygons.reduceToImage(properties=['area'], reducer=ee.Reducer.first()).rename('polygons')

                        composite = ee.Image([logrt_HH, logrt_HV, logrtHvhh, landAreaImg.gt(0), dnbr])

                        id = str(poly.get('Id').getInfo())
                        dir = 'palsar' + '/' + land_cover + '/' + id + '_' + str(year) + '/' + str(year)
                        image_task = ee.batch.Export.image.toDrive(
                            image=composite.toFloat(),
                            description='Image Export:' + land_cover + '_' + id + '_' + str(year),
                            folder='yu_dataset_proj2/'+dir,
                            scale=25,
                            maxPixels=1e9,
                            region=bbox,
                        )
                        image_task.start()
                        print('Start with image task (id: {}).'.format('Image Export:' + land_cover + '_' + id + '_' + str(year)))

    def download_to_gcloud_evaluate(self):
        land_cover = 'evaluation'
        id_list = ee.List(config_eva.get('evaluation'))
        polygons = ee.FeatureCollection('JRC/GWIS/GlobFire/v2/FinalPerimeters').filter(ee.Filter.inList('Id', id_list))
        # bbox = polygons.map(self.get_bbox)
        polygons = polygons.filterMetadata('area', 'not_less_than', 50000000).filterMetadata('InitialDate',
                                                                                             'not_less_than',
                                                                                             1490030000000)
        n = polygons.size().getInfo()
        for i in range(n):
            poly = ee.Feature(polygons.toList(n).get(i))
            bbox = poly.geometry().bounds().buffer(2)
            pre_fire_end = datetime.datetime.fromtimestamp(poly.get('InitialDate').getInfo() / 1000)
            fire_year = pre_fire_end.year
            pre_fire_start = pre_fire_end - datetime.timedelta(weeks=12)
            after_fire_end = pre_fire_end + datetime.timedelta(weeks=8)

            pre_fire_median = ee.ImageCollection('COPERNICUS/S2')\
                .filterDate(pre_fire_start.strftime('%Y-%m-%d'), pre_fire_end.strftime('%Y-%m-%d'))\
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 100))\
                .filterBounds(bbox).median()
            pre_fire_median = self.get_ratio(pre_fire_median)


            for year in range(fire_year, min(fire_year+2, 2021)):
                # PARSAR dataset
                dataset = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR').filter(ee.Filter.date(str(year), str(year + 1)))
                pre_img = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR').filter(ee.Filter.date(str(year - 1), str(year)))

                after_date = dataset.select('date').first().reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=bbox,
                        scale=25,
                        maxPixels=1e9
                    )
                if year >= 2018:
                    after_date = ee.Date.fromYMD(2014, 6, 1).advance(ee.Number(after_date.get('date')).floor(), 'day').getInfo()
                else:
                    after_date = ee.Date.fromYMD(1970, 1, 1).advance(ee.Number(after_date.get('date')).floor(), 'day').getInfo()
                after_date = datetime.datetime.fromtimestamp(after_date.get('value') / 1000)
                print(after_date, pre_fire_end, i, land_cover)
                if after_date < pre_fire_end:
                    if year == fire_year:
                        continue
                    else:
                        break
                else:
                    after_img = ee.ImageCollection('COPERNICUS/S2') \
                        .filterDate(pre_fire_end.strftime('%Y-%m-%d'), after_date.strftime('%Y-%m-%d')) \
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                        .filterBounds(bbox).mean()
                    after_img = self.get_ratio(after_img)

                    dnbr = pre_fire_median.select('index').subtract(after_img.select('index')).rename('dnbr')

                    sarHh_log_pre = pre_img.select('HH').first().pow(2).log10().multiply(10).subtract(83)
                    sarHv_log_pre = pre_img.select('HV').first().pow(2).log10().multiply(10).subtract(83)
                    sarhvhh_pre = sarHv_log_pre.subtract(sarHh_log_pre)

                    sarHh_log = dataset.select('HH').first().pow(2).log10().multiply(10).subtract(83)
                    sarHv_log = dataset.select('HV').first().pow(2).log10().multiply(10).subtract(83)
                    sarHvhh = sarHv_log.subtract(sarHh_log)

                    logrt_HH = sarHh_log_pre.subtract(sarHh_log)
                    logrt_HV = sarHv_log_pre.subtract(sarHv_log)
                    logrtHvhh = sarHvhh.subtract(sarhvhh_pre).rename('HV-HH')

                    landAreaImg = polygons.reduceToImage(properties=['area'], reducer=ee.Reducer.first()).rename('polygons')

                    composite = ee.Image([logrt_HH, logrt_HV, logrtHvhh, landAreaImg.gt(0), dnbr])

                    id = str(poly.get('Id').getInfo())
                    dir = 'palsar' + '/' + land_cover + '/' + id + '_' + str(year) + '/' + str(year)
                    image_task = ee.batch.Export.image.toCloudStorage(
                        image=composite.toFloat(),
                        description='Image Export:' + land_cover + '_' + id + '_' + str(year),
                        fileNamePrefix=dir,
                        bucket='zhaoyutimtest',
                        scale=25,
                        maxPixels=1e9,
                        region=bbox,
                    )
                    image_task.start()
                    print('Start with image task (id: {}).'.format('Image Export:' + land_cover + '_' + id + '_' + str(year)))





    def download_to_local(self):
        storage_client = storage.Client()

        bucket = storage_client.bucket('zhaoyutimtest')
        blobs = bucket.list_blobs(prefix='palsar')
        for blob in blobs:
            if blob.time_created.date() < datetime.date.today()-datetime.timedelta(days=2):
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