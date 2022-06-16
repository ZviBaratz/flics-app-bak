import numpy as np

img_path_angoli = r'd:\git\flics\flics_data\imges\Angoli_vena_conventional_Series012t5_6gradi.tif'

def create_angoli_data_dict(roi_index:int, angle, roi_par, vector_loc, beam_waist_xy, beam_waist_z, rbc_radius ,pixel_dwell_time) -> dict:
    print('creating angoli data_dict')
    return {
        'path': img_path_angoli,
        'frame_rate': None,
        'data_channel': 0,
        'num_of_data_channels': 1,
        'line_rate': None,
        'x_pixel_to_micron': None,
        'y_pixel_to_micron': None,
        'n_frames': 1,
        'min_distance': 0,
        'max_distance': 300,
        'distance_step': 20,
        'distance_step_limit': 50,
        'vector_angle': angle,
        '_2d_projection': None,
        'corr_calc_config': 0,
        'corr_analysis_state': 0,
        'correlation_results': np.zeros((50,516)), #dict() keys: 0-50, values: array of xcor results
        'fitting_params': 0,
        'fitting_analysis_state': 0,
        'fitting_results': 5, #np.zeros((50,516)), #np.array([34, 34, 2322]),
        'vessel_diameters': np.zeros(50), #np.array([[404040, 32323], [30232, 23232]]),
        'corr_calc_state': 0,
        'fitting_state': 1, #1= ready for backend processing . 2= processing. 3= processing done. 0 = processing error
        'roi_coordinates': roi_par, #['x_start', 'x_end', 'y_start', 'y_end']
        'vector_loc': vector_loc,
        'beam_waist_xy': beam_waist_xy,
        'beam_waist_z': beam_waist_z,
        'rbc_radius': rbc_radius,
        'pixel_dwell_time': pixel_dwell_time,
    }
