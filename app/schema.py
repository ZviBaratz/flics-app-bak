from sqlalchemy import ForeignKey, create_engine, Column, Integer, String, Float, BLOB, Binary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import numpy as np


Base = declarative_base()


class Data(Base):
    __tablename__ = 'data'
    id = Column(Integer, primary_key=True)
    path = Column(String)
    data_channel = Column(Integer)
    num_of_data_channels = Column(Integer)
    frame_rate = Column(Float)
    line_rate = Column(Float)
    magnification = Column(Float)
    vessel_orientation = Column(String)
    vessel_type = Column(String)
    axis_len_microns = Column(Float)
    x_pixel_to_micron = Column(Float)
    y_pixel_to_micron = Column(Float)
    n_frames = Column(Integer)
    _2d_projection = Column(BLOB)
    min_distance = Column(Integer)
    max_distance = Column(Integer)
    distance_step = Column(Integer)
    distance_step_limit = Column(Integer)
    vector_angle = Column(Float)
    corr_calc_config = Column(Integer)
    corr_analysis_state = Column(Integer)
    correlation_results = Column(BLOB)
    fitting_params = Column(Integer)
    fitting_results = Column(BLOB)
    corr_calc_state = Column(Integer)
    fitting_state = Column(Integer)
    diameter_analysis_state = Column(Integer)
    diameter_results = Column(BLOB)
    roi_coordinates = Column(BLOB)
    vector_loc = Column(BLOB)
    beam_waist_xy = Column(Float)
    beam_waist_z = Column(Float)
    rbc_radius = Column(Float)
    pixel_dwell_time = Column(Float)


class CorrelationAnalysisConfiguration(Base):
    __tablename__ = 'correlation_analysis'

    id = Column(Integer, primary_key=True)
    data_id = Column(Integer, ForeignKey('data.id'))

    data = relationship("Data", back_populates="correlation_analysis")


Data.correlation_analysis = relationship("CorrelationAnalysisConfiguration", order_by=CorrelationAnalysisConfiguration.id, back_populates="data")


def connect_to_db(db_path: str, delete_table: bool):
    str_for_engine = r'sqlite:///%s' %(db_path)
    engine = create_engine(str_for_engine, echo=True)
    if delete_table:
        Base.metadata.drop_all(engine)
        #delete_table(engine)
    session = sessionmaker(bind=engine)()
    Base.metadata.create_all(engine)
    print('connection created')
    return session


def delete_table(engine):
    Base.metadata.drop_all(engine)
    print('table deleted')


def read_db(db_path: str) -> list:
    # this function returns all information stored in a db
    session = connect_to_db(db_path, False)
    return session.query(Data).all()


def read_db_image(db_path: str, image_full_path: str) -> list:
    # returns a list of all rows which have this image path
    session = connect_to_db(db_path, False)
    list = []
    for row in session.query(Data).filter(Data.path == image_full_path).all():
        list.append(row)
    #print('read_db_image(). returns all rows with the given image path as a list, the list is:',list)
    return list


def get_all_field_in_image_db(db_path: str, image_full_path: str, column_name: str, column_dtype: str) -> list:
    # input: 1. image path, 2. desired field(column) 3. column's data type
    # return: list of the column from all rows that have the image path
    #('get_column_image function. column_name is=', column_name)
    list = []
    session = connect_to_db(db_path, False)
    for row in session.query(Data).filter(Data.path == image_full_path).all():
        data = np.frombuffer(getattr(row, column_name), dtype=getattr(np, column_dtype))
        list.append(data)
    print('roi list is:', list)
    return list


def get_field_in_roi_image_db(db_path: str, image_full_path: str, roi: np.array, column_name: str, column_dtype: str):
    #todo: not working
    # input: 1. image path, 2. roi in image 3. desired field(column) 3. field's data type
    # return: field's value as saved in table (per a specific image and roi)
    print('roi for get_field_in_roi_image_db is:', roi)
    print('get_field_in_roi_image_db, db_path', db_path, 'image_full_path', image_full_path, 'column_name=',column_name, 'column_dtype=', column_dtype)
    session = connect_to_db(db_path, False)
    for row in session.query(Data).filter(Data.path == image_full_path).all():
        roi_db = np.frombuffer(getattr(row, 'roi_coordinates'), dtype=getattr(np, 'float'))
        roi_db_parsed = np.frombuffer(roi_db, dtype=np.float)
        if np.allclose(roi_db_parsed, roi):
            print(f'column_name is {column_name}, column_dtype is {column_dtype}')
            result = np.frombuffer(getattr(row, column_name), dtype=getattr(np, 'int')) #this line isn't working
            print('get_field_in_roi_image_db, result=', result)
            return result


def get_fitting_results_db(db_path: str, image_full_path: str, roi: np.array) : #todo: find out how to call this return value
    # input: 1. image path, 2. roi in image
    # return: fitting_state and fitting results
    print('get_fitting_results_db, db_path', db_path, 'image_full_path', image_full_path)
    session = connect_to_db(db_path, False)
    for row in session.query(Data).filter(Data.path == image_full_path).all():
        roi_db = np.frombuffer(getattr(row, 'roi_coordinates'), dtype=getattr(np, 'float'))
        roi_db_parsed = np.frombuffer(roi_db, dtype=np.float)
        if np.allclose(roi_db_parsed, roi):
            fitting_res = row.fitting_results
            fitting_state = row.fitting_state
            return fitting_state, fitting_res


def get_xcor_results_db(db_path: str, image_full_path: str, roi: np.array):
    # input: 1. image path, 2. roi in image
    # return: xcor_state and xcor results
    print('get_xcor_results_db, db_path', db_path, 'image_full_path', image_full_path)
    session = connect_to_db(db_path, False)
    for row in session.query(Data).filter(Data.path == image_full_path).all():
        roi_parsed = np.frombuffer(roi, dtype=np.float)
        roi_db = np.frombuffer(getattr(row, 'roi_coordinates'), dtype=getattr(np, 'float'))
        if np.allclose(roi_db, roi_parsed):
            xcor_state = row.corr_analysis_state
            xcor_res = np.frombuffer(row.correlation_results, dtype=getattr(np, 'float'))
            return xcor_state, xcor_res


def check_col_image_exist_in_db(db_path: str, image_full_path: str, column_name: str, column_val_tocomp: np.array, column_dtype: str) -> bool:
    #print('check_col_image_exist_in_db func called, db_path = ', db_path)
    col_val_db = np.array(get_all_field_in_image_db(db_path, image_full_path, column_name, column_dtype))
    for db_val in col_val_db:
        if np.allclose(db_val, column_val_tocomp):
            print('check_col_image_exist_in_db func match found in db for:', column_val_tocomp)
            return True
    print('check_col_image_exist_in_db func NO match found in db for:', column_val_tocomp)
    return False


def delete_row_in_db(session, image_full_path: str, roi_val: np.array):
    # input: image, one of the roi values that was marked previously in this image
    # delete this row from the db
    # add in the future - return value: true if succeeded, false if failed
    for row in session.query(Data).filter(Data.path == image_full_path).all():
        data = row.roi_coordinates
        data_parsed = np.frombuffer(row.roi_coordinates, dtype=np.float)
        if np.allclose(data_parsed, roi_val):
            print('delete_row_in_db , roi to delete is - roi_coordinates ==', roi_val)
            #session.query(Data).filter(Data.roi_coordinates == data).update({'delete_row': True})
            session.query(Data).filter(Data.roi_coordinates == data).delete(synchronize_session=False)
            session.commit()
    #session.query(Data).filter(Data.path == image_full_path, Data.delete_row == '1').delete(synchronize_session=False)
    #session.commit()


def data_for_xcorr_globfit():
    pass


# backend :
def get_row_to_procc_from_db_old(db_path):
    """
   # checks fitting_state field in row, returns data if row has not been processed yet.
    #:param db_path:
    #:return:
    """
    session = connect_to_db(db_path, False)
    row = session.query(Data).filter(Data.fitting_state == 0).first()
    if row:
        session.query(Data).filter(Data.id == row.id).update({'fitting_state': 1})
        session.commit()
    return row


def get_data_for_flics_from_db(session):
    row = session.query(Data).filter(Data.fitting_state == 0).first()
    if row:
        session.query(Data).filter(Data.id == row.id).update({'fitting_state': 1})
        session.commit()
    return row


def get_vasc_data_from_db(session):
    row = session.query(Data).filter(Data.diameter_analysis_state == 0).first()
    if row:
        session.query(Data).filter(Data.id == row.id).update({'diameter_analysis_state': 1})
        session.commit()
    return row


def get_row_to_procc_from_db(db_path, is_flics):
    """
    returns relevant row that has not yet been processed to process in backend,
    according to: is_flics (if true calc velocity) and
                  is_parallel (if true calc diameter of parallel to the surface vasc)
                  if both false - calc diameter of perpendicular vasc
    :param db_path:
    :return:
    """
    session = connect_to_db(db_path, False)
    if is_flics:
        data = get_data_for_flics_from_db(session)
    else:
        data = get_vasc_data_from_db(session)
    return data


def from_dict_to_arr(arr_of_dict : list) -> np.array:
    """
    receives array of dictionaries [{},{},{}], each dictionary is the xcorr results of a single frame.
    returns array where each row is a dictionary turned into an array: [[dict1],[dict2]...]  and [dictx]= [key,value,key,value....]
    the length of value is: step_limit+1
    the number of keys in each dictionary is: ((max_distance - min_distance) / distance_step) + 1
    :param dict_input
    :return: matrix with each row having the xcorr results per frame
    """
    res_array = np.zeros((len(arr_of_dict), len(arr_of_dict[0]), len(arr_of_dict[0]['0'])))
    i = 0
    for dict in arr_of_dict:
        res_array[i] = list(dict.values())
        i += 1
    return res_array


def add_res_to_db(db_path: str, img_path: str, roi_img: np.array, fitting_results: np.array, correlation_results: list):
    """
    receives 1. an array of RBC velocities ordered by frame number. (the velocity per frame x is located at position x-1 in the array)
             2. cross-correlation graphs per each frame.
    :return:None
    """
    session = connect_to_db(db_path, False)
    session.query(Data).filter(Data.path == img_path, Data.roi_coordinates == roi_img).update({'fitting_state': 2})
    session.query(Data).filter(Data.path == img_path, Data.roi_coordinates == roi_img).update({'corr_analysis_state': 2})
    correlation_results_arr = from_dict_to_arr(correlation_results)
    session.query(Data).filter(Data.path == img_path, Data.roi_coordinates == roi_img).update({'correlation_results': correlation_results_arr})
    session.query(Data).filter(Data.path == img_path, Data.roi_coordinates == roi_img).update({'fitting_results': fitting_results})
    session.commit()
    print(f'done DB updating')

"""
if __name__ == '__main__':
    db_path = r'd:\git\latest\flics\flics_data\db.db'
    image_full_path = r'd:\git\latest\flics\flics_data\\198_WFA-FITC_RGECO_X25_mag5_910nm_1024px_Plane2_20190916_00001.tif'
    column_name= 'fitting_state'
    column_dtype= 'int'
    roi = np.array([294.76671214, 658.67737617, 318.51568895, 442.77376171])
    get_field_in_roi_image_db(db_path, image_full_path, roi, 'fitting_state', 'int')


#connect_to_db(r'd:\git\flics\flics_data\db.db', True)
"""

"""

arr1 = [{'0': [10, 11], '1':[12, 13]}, {'0': [10, 11], '1':[12, 13]}, {'0': [10, 11], '1':[12, 13]}]
arr2 = np.array([[[10,11], [12,13]], [[10,11], [12,13]],[[10,11], [12,13]]])
img_path_wfa = r'd:\git\flics\flics_data\imges\198_WFA-FITC_RGECO_X25_mag5_910nm_1024px_Plane2_20190916_00001.tif'

add_res_to_db('d:\\git\\flics\\flics_data\\db.db',
 'd:\\git\\flics\\flics_data\\imges\\fov5.tif',
 b'J\xe4o\xdf\xc9\xcc{@\xe0T&l?ez@\x8c\x81n\x0b\xd5SZ@\x9cW@\xe3n\x9dW@', np.array([500., 500., 500.]), arr1)
"""
"""
state, xcor_results = get_xcor_results_db('d:\\git\\flics\\flics_data\\db.db',
 'd:\\git\\flics\\flics_data\\imges\\fov5.tif',
 b'J\xe4o\xdf\xc9\xcc{@\xe0T&l?ez@\x8c\x81n\x0b\xd5SZ@\x9cW@\xe3n\x9dW@')

c = 3
"""
