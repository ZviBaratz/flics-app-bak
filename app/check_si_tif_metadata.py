import pathlib
import pprint

import tifffile
img_path_angoli = pathlib.Path(r'd:\git\flics\flics_data\imges\Angoli_vena_conventional_Series012t5_6gradi.tif')
img_path_wfa = r'd:\git\flics\flics_data\imges\198_WFA-FITC_RGECO_X25_mag5_910nm_1024px_Plane2_20190916_00001.tif'
folder = pathlib.Path(r'Z:\David\reut_flics_07072020')
folder1 = pathlib.Path(r'd:\git\flics\flics_data')
file = pathlib.Path('fov1_Mag_1_256_px_uni_30hz_00001.tif')
file2= r'Z:\Reut\TiRS\fov1_Mag_1_256_px_uni_30hz_00001.tif'

with tifffile.TiffFile(str(file2)) as f:
    meta = f.scanimage_metadata

pprint.pprint(meta)
