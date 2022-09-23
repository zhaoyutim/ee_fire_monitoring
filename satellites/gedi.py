import yaml
import ee
with open("config/sample.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
class gedi:
    def download_to_gcloud(self, dataset='train'):
        width = 1
        centroid = config.get('dbt_na')
        roi = ee.Geometry.Rectangle(
                [centroid.get('lon')-width, centroid.get('lat')-width,
                 centroid.get('lon')+width, centroid.get('lat')+width])
        year = 2019
        dataset = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR')\
            .filter(ee.Filter.date(str(year), str(year + 1)))
        date_pre = dataset.select('date').median().clip(roi).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=roi,
            scale=100,
            crs='EPSG:4326').getNumber('date')
        date_pre = ee.Date(ee.Date.fromYMD(2014, 5, 24).advance(date_pre, 'day'))
        print(date_pre.getInfo())
        sarHh_log = dataset.select('HH').first().pow(2).log10().multiply(10).subtract(83)
        sarHv_log = dataset.select('HV').first().pow(2).log10().multiply(10).subtract(83)
        sarhvhh = sarHv_log.subtract(sarHh_log).rename('HV-HH')
        composite = ee.Image([sarHh_log, sarHv_log, sarhvhh])
        l4b = ee.Image('LARSE/GEDI/GEDI04_B_002').select('PS')
        gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')\
            .filterDate(date_pre.difference(15, "day"), date_pre.advance(15, "day"))\
            .map(self.qualityMask)\
            .select(['rh40', 'rh50', 'rh60', 'rh70', 'rh98']).mosaic()
        output = ee.Image([composite, l4b, gedi])
        dir = 'gedi_eva' + '/' + 'dnt_na'
        image_task = ee.batch.Export.image.toCloudStorage(
            image=output.toFloat(),
            description='Image Export:' + 'example gedi',
            fileNamePrefix=dir,
            bucket='ai4wildfire',
            scale=25,
            maxPixels=1e11,
            region=roi,
        )
        image_task.start()
        print('Start with image task (id: {}).'.format(
            'Palsar Image Export:' + 'gedi dnt_na'))

    def qualityMask(self, img):
        return img.updateMask(img.select('quality_flag').eq(1)).updateMask(img.select('degrade_flag').eq(0))
