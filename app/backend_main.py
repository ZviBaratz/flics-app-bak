from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from file_handling import *
from schema import *
from app.analysis.flics_edit import *

db_path = r'd:\git\latest\flics\flics_data\db.db'


def xcorr_global_fit_on_frame(data_to_xcor) -> dict:
    print('running xcorr_global_fit_on_frame', data_to_xcor["frame"])
    flics_analysis = Analysis(data_to_xcor["image"], None, None, data_to_xcor['meta_data'].min_distance,
                              data_to_xcor['meta_data'].max_distance, data_to_xcor['meta_data'].distance_step,
                              data_to_xcor['meta_data'].distance_step_limit, True)
    global_fitting = GlobalFit(flics_analysis.results, 0.02472, 0.03651, 0.2, 2e-6, 6.2666, 0.001)
    v = global_fitting.run()
    return {data_to_xcor['frame'] : v}
    #print('global fit results for frame number:', i_frame, 'velocity is:', v)


def prepare_data_to_xcor() -> list:
    """
    reads single row in the DB, each row is a roi to process
    creates a dictionary per frame with all the data needed for processing
    returns a list of all dictionaries, all the frames of the chosen roi

    :return: list of dictionaries
    """
    db_row = get_row_to_procc_from_db(db_path)
    if not db_row:
        return None
    data_to_xcorr = []
    for i_frame in range(db_row.n_frames +1):
        data_array = crop_img(db_row.roi_coordinates, i_frame, db_row.path)
        data_to_xcorr.append({'frame': i_frame, 'meta_data': db_row, "image": data_array})
    return data_to_xcorr


def run_xcorr_global_fit():
    data_to_xcorr = prepare_data_to_xcor()
    if not data_to_xcorr:
        return None
    fitting_results = []
    with PoolExecutor(max_workers=4) as executor:
        for frame_res in executor.map(xcorr_global_fit_on_frame, data_to_xcorr):
            fitting_results.append(frame_res)
            print(frame_res)
    add_res_to_db()

run_xcorr_global_fit()
