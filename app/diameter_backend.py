from schema import get_row_to_procc_from_db
import matlab.engine
import matlab
import numpy as np


def connect_matlab_eng():
    eng = matlab.engine.start_matlab()
    eng.cd(r'/data/Reut/TiRS/')
    return eng


def quit_matlab_eng(eng):
    try:
        eng.ls()
        eng.quit()
    except:
        print('could not stop matlab engine')


def run_parallel(eng, data):
    file_name = data.path #'vessel_in_plane.tif'
    magnification = data.magnification # 1;  %zoom_factor in python app
    axis_len_micr = data.axis_len_microns # 165; %get_fov() in python app
    frameRate = data.frame_rate
    vesselType = data.vessel_type
    roi = data.roi_coordinates # [89.6297   64.5569; 68.1385   95.8353; 176.0102  181.5496; 195.2522  149.4796];
    vector = data.vector_loc #[101.8950  163.7026; 157.5802   72.4490];
    res = eng.DiamCalcSurfaceVessel(file_name, magnification, axis_len_micr, frameRate, vesselType, roi, vector)
    res_arr = np.array(res._data)
    #to be replaced:
    print('res is:', res_arr)
    import matplotlib.pyplot as plt
    t = np.linspace(start=0, stop=4799, num=4800)
    plt.plot(t, res_arr)
    plt.ylabel('diameter')
    plt.show()


def run_perpendicular(data):
    pass


def diameter_main(db_path):
    eng = connect_matlab_eng()
    is_flics = False
    data = get_row_to_procc_from_db(db_path, is_flics)
    if data.vessel_orientation == "parallel":
        run_parallel(eng, data)
    else:
        run_perpendicular(eng, data)
    quit_matlab_eng(eng)
