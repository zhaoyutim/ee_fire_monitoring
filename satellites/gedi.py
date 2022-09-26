import yaml
import ee
with open("config/sample.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
class gedi:
    def download_to_gcloud(self, region_ids=['na'], dataset='train'):
        year = 2019
        dataset = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/SAR') \
            .filter(ee.Filter.date(str(year), str(year + 1)))
        sarHh_log = dataset.select('HH').first().pow(2).log10().multiply(10).subtract(83)
        sarHv_log = dataset.select('HV').first().pow(2).log10().multiply(10).subtract(83)
        sarhvhh = sarHv_log.subtract(sarHh_log).rename('HV-HH')
        composite = ee.Image([sarHh_log, sarHv_log, sarhvhh])
        l4b = ee.Image('LARSE/GEDI/GEDI04_B_002').select('PS')
        for region_id in region_ids:
            roi_col = ee.FeatureCollection('users/zhaoyutim/GEDI_SAMPLE_'+region_id.upper())
            size = roi_col.size().getInfo()
            roi_col = roi_col.toList(size)
            for i in range(size):
                roi = ee.Feature(roi_col.get(i).getInfo())
                class_id = roi.args['metadata'].get('class')
                date_pre = dataset.select('date').median().clip(roi).reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=roi.geometry(),
                    scale=100,
                    crs='EPSG:4326').getNumber('date')
                date_pre = ee.Date(ee.Date.fromYMD(2014, 5, 24).advance(date_pre, 'day'))
                gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')\
                    .filterDate(date_pre.difference(15, "day"), date_pre.advance(15, "day"))\
                    .map(self.qualityMask)\
                    .select(['rh40', 'rh50', 'rh60', 'rh70', 'rh98']).mosaic()
                output = ee.Image([composite, l4b, gedi])
                dir = 'proj4_gedi_palsar' + '/' + region_id.upper() + '/' + 'class_' + class_id + '_' + str(i)
                image_task = ee.batch.Export.image.toCloudStorage(
                    image=output.toFloat(),
                    description='Image Export:' + 'GEDI_PALSAR_' + region_id.upper()+'_CLASS_'+class_id,
                    fileNamePrefix=dir,
                    bucket='ai4wildfire',
                    scale=25,
                    maxPixels=1e11,
                    region=roi.geometry(),
                    fileDimensions=256
                )
                image_task.start()
                print('Start with image task (id: {}).'.format(
                    'GEDI-PALSAR Image Export:' + 'GEDI_SAMPLE_'+region_id.upper()+'_INDEX_'+str(i)+'_CLASS_'+class_id))

    def qualityMask(self, img):
        return img.updateMask(img.select('quality_flag').eq(1)).updateMask(img.select('degrade_flag').eq(0))
