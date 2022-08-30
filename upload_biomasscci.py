import os
import subprocess
from glob import glob

import ee

if __name__=='__main__':
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:15236'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:15236'
    year='2018'
    path = os.path.join('/Users/zhaoyu/v3.0/geotiff', year, '*.tif')
    file_list = glob(path)
    for file in file_list:
        cloud_file = file.replace('.0', '').split('/')[-1]
        print('upload to gcloud')
        upload_cmd = 'gsutil cp ' + file + ' gs://ai4wildfire/' + 'biomasscci/' + year + '/' + cloud_file
        os.system(upload_cmd)

        print('upload to gee')
        cmd = 'earthengine upload image --time_start '+year+'-01-01' + ' --asset_id=projects/ee-zhaoyutim/assets/biomasscci'+year+'/' + \
              cloud_file.split('/')[-1][
              :-4] + ' --pyramiding_policy=sample gs://ai4wildfire/' + 'biomasscci/' + year + '/' + cloud_file
        subprocess.call(cmd.split())
