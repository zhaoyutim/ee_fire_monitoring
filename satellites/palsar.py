import os
import yaml
import ee
from google.cloud import storage

with open("land_cover_id.yaml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class Palsar:
    def get_bbox(self, feature):
        return feature.bounds().buffer(2)

    def download_to_gcloud(self):
        land_covers = ['evergreen_needle', 'evergreen_broadleaf', 'open_shrublands', 'woody_savannas', 'savannas', 'grasslands']
        for land_cover in land_covers:
            for year in range(2017, 2021):
                id = ee.List(config.get(land_cover))
                polygons = ee.FeatureCollection('JRC/GWIS/GlobFire/v2/FinalPerimeters').filter(ee.Filter.inList('Id', id))
                bbox = polygons.map(self.get_bbox)
                polygons = polygons.filterMetadata('area', 'not_less_than', 50000000).filterMetadata('InitialDate',
                                                                                                     'not_less_than',
                                                                                                     1490030000000)

                # PARSAR dataset
                dataset = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR').filter(ee.Filter.date(str(year), str(year + 1)))
                pre_img = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR').filter(ee.Filter.date(str(year - 1), str(year)))

                sarHh_log_pre = pre_img.select('HH').first().pow(2).log10().multiply(10).subtract(83);
                sarHv_log_pre = pre_img.select('HV').first().pow(2).log10().multiply(10).subtract(83);
                sarhvhh_pre = sarHv_log_pre.subtract(sarHh_log_pre)

                sarHh_log = dataset.select('HH').first().pow(2).log10().multiply(10).subtract(83);
                sarHv_log = dataset.select('HV').first().pow(2).log10().multiply(10).subtract(83);
                sarHvhh = sarHv_log.subtract(sarHh_log)

                logrt_HH = sarHh_log_pre.subtract(sarHh_log)
                logrt_HV = sarHv_log_pre.subtract(sarHv_log)
                logrtHvhh = sarHvhh.subtract(sarhvhh_pre).rename('HV-HH')

                landAreaImg = polygons.reduceToImage(properties=['area'], reducer=ee.Reducer.first()).rename('polygons')

                composite = ee.Image([logrt_HH, logrt_HV, logrtHvhh, landAreaImg.gt(0)])

                n = polygons.size().getInfo()
                for i in range(n):
                    poly = ee.Feature(polygons.toList(n).get(i))
                    geometry = poly.geometry().bounds()

                    id = str(poly.get('Id').getInfo())
                    ini_date = str(ee.Date(poly.get('InitialDate')).get('year').getInfo())
                    dir = 'palsar' + '/' + land_cover + '/' + id + '_' + ini_date + '/' + str(year)
                    image_task = ee.batch.Export.image.toCloudStorage(
                        image=composite.toFloat(),
                        description='Image Export:' + land_cover + '_' + id + '_' + ini_date,
                        fileNamePrefix=dir,
                        bucket='zhaoyutimtest',
                        scale=25,
                        maxPixels=1e9,
                        region=geometry,
                    )
                    image_task.start()
                    print('Start with image task (id: {}).'.format('Image Export:' + land_cover + '_' + id + '_' + ini_date))

    def download_to_local(self):
        storage_client = storage.Client()

        bucket = storage_client.bucket('zhaoyutimtest')
        blobs = bucket.list_blobs(prefix='palsar')
        for blob in blobs:
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