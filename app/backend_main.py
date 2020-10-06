from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from schema import *
from app.analysis.flics_edit import *
from diameter_backend import diameter_main

db_path = r'Z:\David\reut_flics_07072020' #r'd:\git\flics\flics_data\db.db'
img_path_angoli = r'd:\git\flics\flics_data\imges\Angoli_vena_conventional_Series012t5_6gradi.tif'


class Backend(object):
    def __init__(self, db_path :str, autorun : bool = True):
        self.db_path = db_path
        #todo: try:   #create session - if success and autorun -> run main_server. else error
        if autorun:
            self.main_server()

    def xcorr_global_fit_on_frame(self, data_to_xcor: dict) -> dict:
        """
        recieves a dict fer frame per roi and runs cross-correlation and global fit
        returns blood flow velocity in that given frame
        :param data_to_xcor: data regarding a single frame in an roi
        :type data_to_xcor : dict
        :return: frame number, RBC flow velocity in that frame, xcorr results
        """
        print('running xcorr_global_fit_on_frame', data_to_xcor['frame'])
        flics_analysis = Analysis(data_to_xcor['meta_data'].data_channel, data_to_xcor['meta_data'].num_of_data_channels,
                                  None, data_to_xcor['meta_data'].path,
                                  data_to_xcor['frame'], data_to_xcor['meta_data'].roi_coordinates, None,
                                  data_to_xcor['meta_data'].min_distance,
                                  data_to_xcor['meta_data'].max_distance, data_to_xcor['meta_data'].distance_step,
                                  data_to_xcor['meta_data'].distance_step_limit, True)

        #temp: for comparing to the article:
        if data_to_xcor['meta_data'].path == img_path_angoli:
            print('running globalfit for angoli img')
            global_fit = GlobalFit(flics_analysis.results, data_to_xcor['meta_data'].vector_angle,
                                   0.03651, 0.2, 2e-6, 6.2666,0.001)
        else:
            global_fit = GlobalFit(flics_analysis.results, data_to_xcor['meta_data'].vector_angle,
                                   data_to_xcor['meta_data'].x_pixel_to_micron, data_to_xcor['meta_data'].beam_waist_xy,
                                   data_to_xcor['meta_data'].beam_waist_z, data_to_xcor['meta_data'].rbc_radius,
                                   data_to_xcor['meta_data'].pixel_dwell_time)
        v = global_fit.results
        print(f'global fit results for frame number:{data_to_xcor["frame"]}, velocity is: {v}')
        return [data_to_xcor['frame'], v, flics_analysis.results]

    def prepare_data_to_xcor(self) -> list:
        """
        reads single row in the DB, each row is a roi to process
        creates a dictionary per frame with all the data in that row
        returns a list of all dictionaries, all the frames of the chosen roi

        :return: list of dictionaries
        """
        is_flics = True
        db_row = get_row_to_procc_from_db(self.db_path, is_flics)
        if not db_row:
            return None
        data_to_xcorr = []
        for i_frame in range(db_row.n_frames+1):
            data_to_xcorr.append({'frame': i_frame, 'meta_data': db_row})
        return data_to_xcorr

    def run_xcorr_global_fit(self):
        data_to_xcor = self.prepare_data_to_xcor()
        if not data_to_xcor:
            time.sleep(10) #improve in the future - determening for how long to sleep
        else:
            fitting_results = np.zeros(len(data_to_xcor))
            xcor_results_dictlist = [dict() for x in range(len(data_to_xcor))] #np.zeros(len(data_to_xcor))
            with PoolExecutor(max_workers=1) as executor:
                for res_arr in executor.map(self.xcorr_global_fit_on_frame, data_to_xcor):
                    print(f'frame is :{res_arr[0]}, velocity is: {res_arr[1]} ')
                    fitting_results[res_arr[0]] = res_arr[1]
                    xcor_results_dictlist[res_arr[0]] = res_arr[2]
            add_res_to_db(self.db_path, data_to_xcor[0]['meta_data'].path, data_to_xcor[0]['meta_data'].roi_coordinates, fitting_results, xcor_results_dictlist)

    def run_diameter_calc(self):
        pass

    def main_server(self):
        while True:
            diameter_main(db_path)
            #self.run_xcorr_global_fit()
            #self.run_diameter_calc()

Backend(db_path)
