from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from file_handling import *
from schema import *
from app.analysis.flics_edit import *

db_path = r'd:\git\flics\flics_data\db.db'


class Backend(object):
    def __init__(self, db_path :str, autorun : bool = True):
        self.db_path = db_path

        #try:   #create session - if success - and autorun: run main_server

        if autorun:
            self.results = self.main_server()

    def xcorr_global_fit_on_frame(self, data_to_xcor) -> dict:
        print('running xcorr_global_fit_on_frame', data_to_xcor['frame'])
        flics_analysis = Analysis(None, data_to_xcor['meta_data'].path, data_to_xcor['frame'],
                                data_to_xcor['meta_data'].roi_coordinates, None, data_to_xcor['meta_data'].min_distance,
                                  data_to_xcor['meta_data'].max_distance, data_to_xcor['meta_data'].distance_step,
                                  data_to_xcor['meta_data'].distance_step_limit, True)
        global_fitting = GlobalFit(flics_analysis.results, data_to_xcor['meta_data'].vector_angle,
                                   data_to_xcor['meta_data'].x_pixel_to_micron, data_to_xcor['meta_data'].beam_waist_xy,
                                   data_to_xcor['meta_data'].beam_waist_z, data_to_xcor['meta_data'].rbc_radius,
                                   data_to_xcor['meta_data'].pixel_dwell_time)
        v = global_fitting.run()
        return {data_to_xcor['frame']: v}
        print('global fit results for frame number:', data_to_xcor['frame'], 'velocity is:', v)

    def prepare_data_to_xcor(self) -> list:
        """
        reads single row in the DB, each row is a roi to process
        creates a dictionary per frame with all the data needed for processing
        returns a list of all dictionaries, all the frames of the chosen roi

        :return: list of dictionaries
        """
        db_row = get_row_to_procc_from_db(self.db_path)
        if not db_row:
            return None
        data_to_xcorr = []
        for i_frame in range(db_row.n_frames +1):
            data_to_xcorr.append({'frame': i_frame, 'meta_data': db_row})
        return data_to_xcorr

    def run_xcorr_global_fit(self):
        data_to_xcorr = self.prepare_data_to_xcor()
        if not data_to_xcorr:
            time.sleep(10)
        else:
            fitting_results = []
            with PoolExecutor(max_workers=1) as executor:
                for frame_res in executor.map(self.xcorr_global_fit_on_frame, data_to_xcorr):
                    fitting_results.append(frame_res)
                    print(frame_res)
            add_res_to_db()

    def main_server(self):
        while True:
            self.run_xcorr_global_fit()


#Backend(db_path)
