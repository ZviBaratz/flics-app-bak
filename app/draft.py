from PIL import Image
import PIL.Image
import tifffile
import numpy as np
from file_handling import *
from schema import *
import math

from bokeh.plotting import figure, output_file, show

lines_distance = 0.1
num_of_lines = 5
pixel_size = 0.005

x_user_start = -3 #np.random.random()
y_user_start = 9 #np.random.random()
x_user_end = 4 #np.random.random()
y_user_end = 15 #np.random.random()
x = [x_user_start, x_user_end]
y = [y_user_start, y_user_end]

m_user = (y[1]-y[0])/(x[1]-x[0])
b_user = y[0]-m_user*x[0]


def calc_m(vec):
    if (vec[2] - vec[0]) == 0:
        return 'inf'
    return (vec[3] - vec[1]) / (vec[2] - vec[0])


def calc_b(x, y, m):
    b = y-m*x
    return b


def check_is_per(vec1, vec2):
    m1 = calc_m(vec1)
    m2 = calc_m(vec2)
    if m1 == 'inf' and m2 == 0 or m1 == 0 and m2 == 'inf':
        return True
    if -1 < m1*m2 < -0.98:
        print(f'vectors are perpendicular')
        return True
    print(f'vectors are NOT perpendicular')
    return False


def create_perpen(x_start, y_start, x_end, y_end):
    a = x_end - x_start
    b = y_end - y_start
    x_per_start = x_start - b
    y_per_start = y_start + a
    x_per_end = x_start + b
    y_per_end = y_start - a
    #check_is_per((x_start, y_start, x_end, y_end), (x_per_start, y_per_start, x_per_end, y_per_end))
    p.line([x_per_start, x_per_end], [y_per_start, y_per_end], legend_label="Temp.", line_color="orange",line_width=2)
    return x_per_start, y_per_start, x_per_end, y_per_end


# output to static HTML file
output_file("lines.html")

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.line(x, y, legend_label="Temp.", line_width=2)


def find_delta_x(m):
    delta = pixel_size / math.sqrt(m*m+1)
    return delta


def go_over_line(x_start, y_start, x_end, y_end):
    m_perpen = calc_m((x_start, y_start, x_end, y_end))
    b_perpen = calc_b(x_start, y_start, m_perpen)
    delta_x = find_delta_x(m_perpen)
    x_current = min(x_start, x_end)
    x_end = max(x_start, x_end)
    while x_current < x_end:
        x_current += x_current+delta_x
        y_current = m_perpen*x_current + b_perpen
        p.circle(x_current, y_current, color='black')


def handle_perpendic(x_start_per, x_end_per, b_per):
    m_perpen = -1/m_user
    l3 = (x_end_per - x_start_per)/12
    delta_x_3 = calc_delta(l3, m_perpen)
    x_current = x_start_per
    while x_current < x_end_per:
        x_current += delta_x_3
        y_current = m_perpen*x_current + b_per
        p.circle(x_current, y_current, color='black')



   # go_over_line(x_per_start, y_per_start, x_per_end, y_per_end)

def calc_delta(l, m):
    return l/math.sqrt(m*m +1)


# main :
L1 = 5
L2 = 0.1
delta_x_1 = calc_delta(L1, 1/m_user)
delta_x_2 = calc_delta(L2, m_user)
x_start_per = x_user_start - delta_x_1
x_end_per = x_user_start + delta_x_1
for i in range(num_of_lines):
    b_perpen = (x_user_start + i*delta_x_2) * (m_user + 1/m_user) + b_user
    handle_perpendic(x_start_per+i*delta_x_2, x_end_per+i*delta_x_2, b_perpen) #m_perpen = -1/m_user


# show the results
show(p)

"""
db_path = 'd:\git\\flics\\flics_data\db.db'

img_path_wfa = r'd:\git\flics\flics_data\imges\198_WFA-FITC_RGECO_X25_mag5_910nm_1024px_Plane2_20190916_00001.tif'
img_path_fov7 = r'd:\git\flics\flics_data\imges\fov7.tiff'
img_path_angoli = r'd:\git\flics\flics_data\imges\Angoli_vena_conventional_Series012t5_6gradi.tif'
img_path_fov5 = r'd:\git\flics\flics_data\imges\fov5.tif'

from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from schema import *
from app.analysis.flics_edit import *


db_path = r'd:\git\flics\flics_data\db.db'
img_path_angoli = r'd:\git\flics\flics_data\imges\Angoli_vena_conventional_Series012t5_6gradi.tif'

arr2 = np.array([[[10,11], [12,13]], [[10,11], [12,13]],[[10,11], [12,13]]])
keys = np.array((0,1))
filtered_results = dict(list(zip(keys, list(arr2[0]))))
x =1
"""
"""
dict1 = {'1':12}
dict2= {'2':13}
dict3 = {'3':15}

arr = [{'0':[10,11]},{'1':[12,13]},{'2':[13,14]},{'3':[15,16]}]
len(arr)
x=3
res = np.zeros((len(arr), len(arr[0]['0'])))
res.append(arr[0].keys(), arr[0].values())
x=4
res = np.zeros((len(arr), len(arr[0])*len(arr[0]['0'])))
"""
"""
def get_field_in_roi_image_db_draft(db_path: str, image_full_path: str, roi: np.array, column_name: str, column_dtype: str):
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
            result = np.frombuffer(getattr(row, column_name), dtype='int')
                                                                    #'getattr(np, 'float'))
            print('get_field_in_roi_image_db, result=', result)
            return result

res = get_field_in_roi_image_db_draft(db_path, img_path_wfa, [506.41167179, 790.38981387, 518.83778618, 844.655968], 'fitting_results', 'int')

x=3
"""
"""
def comp_load_to_crop():
    image = PIL.Image.open(img_path_angoli)
    x = image.size[0]
    y = image.size[1]
    data = [[0 for i in range(y)] for j in range(x)]
    a = list(image.getdata())
    for i in range(y):
        for j in range(x):
            data[j][i] = float(a[i * x + j])

    print('data is:', data)

    image_crop = crop_img(None, 0, img_path_angoli, 0, 1)
    print('img_crop is:', image_crop)
    T=1

comp_load_to_crop()

"""
"""
dict1 = {'a':1}
dict2 = {'b':2}
list = [dict1, dict2]
a = list[0]['a']
"""
