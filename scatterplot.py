import matplotlib.pyplot as plt
import rasterio

def read_tiff(self, file_path):
    with rasterio.open(file_path, 'r') as reader:
        profile = reader.profile
        tif_as_array = reader.read()
    return tif_as_array, profile

def main():
    folder = 'palsar'


if __name__=='__main__':
    main()