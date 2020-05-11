from sqlalchemy import ForeignKey, create_engine, Column, Integer, String, Float, BLOB, Binary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import numpy as np


Base = declarative_base()


class Data(Base):
    __tablename__ = 'data'
    id = Column(Integer, primary_key=True)
    path = Column(String)
    frame_rate = Column(Float)
    line_rate = Column(Float)
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
    fitting_analysis_state = Column(Integer)
    fitting_results = Column(Integer) #todo: should be blob!
    vessel_diameters = Column(Binary)
    corr_calc_state = Column(Integer)
    fitting_state = Column(Integer)
    roi_coordinates = Column(BLOB)
    vector_loc = Column(BLOB)
    beam_waist_xy= Column(Float)
    beam_waist_z = Column(Float)
    rbc_radius = Column(Float)
    pixel_dwell_time= Column(Float)


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
        print('get_all_field_in_image_db. row.', column_name, 'is=', data)
        list.append(data)
    print('get_all_field_in_image_db, roi list is:', list)
    return list


def get_field_in_roi_image_db(db_path: str, image_full_path: str, roi: np.array, column_name: str, column_dtype: str):
    # input: 1. image path, 2. roi in image 3. desired field(column) 3. field's data type
    # return: field's value as saved in table (per a specific image and roi)
    print('roi for get_field_in_roi_image_db is:', roi)
    print('get_field_in_roi_image_db, db_path', db_path, 'image_full_path', image_full_path, 'column_name=',column_name, 'column_dtype=', column_dtype)
    session = connect_to_db(db_path, False)
    for row in session.query(Data).filter(Data.path == image_full_path).all():
        roi_db = np.frombuffer(getattr(row, 'roi_coordinates'), dtype=getattr(np, 'float'))
        roi_db_parsed = np.frombuffer(roi_db, dtype=np.float)
        if np.allclose(roi_db_parsed, roi):
            result = np.frombuffer(getattr(row, column_name), dtype=getattr(np, column_dtype)) #dtype = getattr(np, column_dtype))
            print('get_field_in_roi_image_db, result=',result)
            return result


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


def add_res_to_db():
    """

    :return:
    """
#    fitting_results.sort(key=operator.itemgetter('frame'))
    pass

# backend :
def get_row_to_procc_from_db(db_path):
    session = connect_to_db(db_path, False)
    row = session.query(Data).filter(Data.fitting_state == 1).first()
    if row:
        session.query(Data).filter(Data.id == row.id).update({'fitting_state': 1})
        session.commit()
    return row

"""
if __name__ == '__main__':
    db_path = r'd:\git\latest\flics\flics_data\db.db'
    image_full_path = r'd:\git\latest\flics\flics_data\\198_WFA-FITC_RGECO_X25_mag5_910nm_1024px_Plane2_20190916_00001.tif'
    column_name= 'fitting_state'
    column_dtype= 'int'
    roi = np.array([294.76671214, 658.67737617, 318.51568895, 442.77376171])
    get_field_in_roi_image_db(db_path, image_full_path, roi, 'fitting_state', 'int')

"""
