import pathlib

import numpy as np

from app.main import *
from app.schema import *


def init_db(db_path):
    session = connect_to_db(db_path, True)
    return session


def create_mock_data_dict() -> dict:
    return {
        'path': "d:/git/flics/flics_data/198_WFA-FITC_RGECO_X25_mag5_910nm_1024px_Plane2_20190916_00001.tif", #get_full_path(image_select.value),
        'frame_rate': 300, #frame_rate_input.value,
        'line_rate': 900, #line_rate_input.value,
        'x_pixel_to_micron': 30, #calc_x_pixel_to_micron(), #x_pixel_to_micron_input.value,  #float(x_pixel_to_micron_input.value) * 1e-6,
        'y_pixel_to_micron': 40, #calc_y_pixel_to_micron(), #_pixel_to_micron(), #get_pixel_to_micron(y_pixel_to_micron_input.value) #float(y_pixel_to_micron_input.value), #make sure
        '_2d_projection': np.zeros((1024, 1024)), #calculate_2d_projection()
        'corr_calc_config': int(0),
        'corr_analysis_state': int(0),
        'correlation_results': np.array([0, 1, 2, 3]),
        'fitting_params': 0,
        'fitting_analysis_state': 0,
        'fitting_results': np.array([34, 34, 2322]),
        'vessel_diameters': np.array([[404040, 32323], [30232, 23232]]),
        'corr_calc_state': 0,
        'fitting_state': 0,
        'roi_coordinates':  np.array([100, 120, 100, 120], dtype=np.uint16),
        'vector_loc': np.array([[1, 2], [3, 4], [10, 20], [30, 40]], dtype=np.uint16),
        'delete_row' : False
    }


if __name__ == '__main__':
    db_path1 = r'd:\git\flics\flics_data\db.db'
    db_path2 = r'd:\git\new_data_folder\db.db'
    image_full_path = r"d:\git\flics\flics_data\198_WFA-FITC_RGECO_X25_mag5_910nm_1024px_Plane2_20190916_00001.tif"
    session = init_db(str(db_path1))
    #datadict = create_mock_data_dict()
    #data = Data(**datadict)
   # session.add(data)
    #session.commit()


#curr_roi = np.array([802.34289734, 746.94272402, 247.11348909, 286.34833082], dtype='float')

str_for_engine = r'sqlite:///%s' %(db_path1)
print('str_for_engine=', str_for_engine)
engine = create_engine(str_for_engine, echo=True)
session = sessionmaker(bind=engine)()
Base.metadata.create_all(engine)
print('connection created')

#x = check_col_image_exist_in_db(db_path, image_full_path, 'roi_coordinates', curr_roi, 'float')

y=1

