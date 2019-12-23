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
    _2d_projection = Column(BLOB)
    corr_calc_config = Column(Integer)
    corr_analysis_state = Column(Integer)
    correlation_results = Column(BLOB)
    fitting_params = Column(Integer)
    fitting_analysis_state = Column(Integer)
    fitting_results = Column(BLOB)
    vessel_diameters = Column(Binary)
    corr_calc_state = Column(Integer)
    fitting_state = Column(Integer)
    roi_coordinates = Column(BLOB)
    vector_loc = Column(BLOB)
    distance = Column(BLOB)
    step = Column(Integer)
    delete_row = Column(Boolean)

class CorrelationAnalysisConfiguration(Base):
    __tablename__ = 'correlation_analysis'

    id = Column(Integer, primary_key=True)
    data_id = Column(Integer, ForeignKey('data.id'))

    data = relationship("Data", back_populates="correlation_analysis")


Data.correlation_analysis = relationship("CorrelationAnalysisConfiguration", order_by=CorrelationAnalysisConfiguration.id, back_populates="data")


def connect_to_db(db_path : str, delete_table: bool):
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


def read_db_image(db_path: str, image__full_path: str) -> list:
    # returns a list of all rows which have this image path
    session = connect_to_db(db_path, False)
    list = []
    for row in session.query(Data).filter(Data.path == image__full_path).all():
        list.append(row)
    print('read_db_image(). returns all rows with the given image path as a list, the list is:',list)
    return list


def get_column_image_db(db_path: str, image_full_path: str, column_name: str, column_dtype: str) -> list:
    # input: 1. image path, 2. desired field(column) 3. columns data type
    # return: list of the column from all rows that have the image path
    print('get_column_image function. column_name is=', column_name)
    list = []
    session = connect_to_db(db_path, False)
    for row in session.query(Data).filter(Data.path == image_full_path).all():
        data = np.frombuffer(getattr(row, column_name), dtype=getattr(np, column_dtype))
        print('row.', column_name, 'is=', data)
        list.append(data)
    return list


def check_col_image_exist_in_db(db_path : str, image_full_path: str, column_name: str, column_val_tocomp: np.array, column_dtype: str) -> bool:
    print('check_col_image_exist_in_db func called, db_path = ', db_path)
    col_val_db = np.array(get_column_image_db(db_path, image_full_path, column_name, column_dtype))
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
            print('match found, row.roi_coordinates ==', roi_val)
            session.query(Data).filter(Data.roi_coordinates == data).update({'delete_row': True})
            session.commit()
    session.query(Data).filter(Data.path == image_full_path, Data.delete_row == '1').delete(synchronize_session=False)
    session.commit()


if __name__ == '__main__':
    connect_to_db('db_path')
