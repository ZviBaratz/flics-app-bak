from angoli_meta_data import create_angoli_data_dict
import math
import pathlib
from functools import partial
from schema import *
from file_handling import *
from bokeh.plotting import curdoc
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import (
    Button,
    Paragraph,
    Select,
    Slider,
    TextInput,
    Toggle,
)
from bokeh.palettes import Category20_20
from figures.image_plot import create_image_figure
from figures.results_plot import (
    create_results_plot,
    create_cross_correlation_plot,
)
defualt_base_dir_z = r'Z:\David\reut_flics_07072020'
defualt_base_dir_mycomp = r'd:\git\flics\flics_data\\'
base_dir_input = TextInput(value=defualt_base_dir_z, title='Base Directory')

config = {
    'null': '0',
    'beam_waist_xy': '0.5e-6',
    'beam_waist_z': '2e-6',
    'rbc_radius': '4e-6',
    'tau': '0.001',
    's': 1,
    'min_distance': '0',
    'max_distance': '300',
    'distance_step': '20',
    'distance_step_limit': '50',
}


def update_base_dir_select(attr, old, new: str):
    """
    called when user changes basedir
    adds the image's file path to a bokeh

    :param attr: part of function signature , not used
    :param old: part of function signature , not used
    :param new: new base_dir value
    :type new: str
    :return: None
    """
    print(f'update_base_dir_select to {base_dir_input.value} if isdir true. isdir = {os.path.isdir(new)}')
    if os.path.isdir(new):
        image_paths = get_file_paths(base_dir_input.value)
        relative_paths = [path[len(new):] for path in image_paths]
        if relative_paths:
            image_select.options = relative_paths
            image_select.value = image_select.options[0]
            print(f'image_select,value = {image_select.value}')
            return
    image_select.options = [config['null']]


def get_db_path() -> pathlib.Path:
    """
    Asserts that the user supplied a valid directory,
    and if so returns the possible name of the database
    which should be located inside that directory.
    Note: The database might not exist yet.
    :return: pathlib.Path - path to database file
    """
    data_folder = pathlib.Path(base_dir_input.value)
    if data_folder.is_dir():
        print('db_path is:', data_folder / 'db.db')
        return data_folder / 'db.db'


base_dir_input.on_change('value', update_base_dir_select)

raw_source = ColumnDataSource(data=dict(image=[], meta=[]))

image_select = Select(
    title='Image', value=config['null'], options=[config['null']])


menu = ["1", "2", "3", "4"]
data_channel = Select(title="Data channel:", value="1", options=menu)
num_of_channels = Select(title="Number of Data Channels:", value="1", options=menu)

orient_options = ["perpendicular", "parallel"]
vessel_orient = Select(title="Vessel's Orientation:", options=orient_options)

vessel_types = ['A', 'D', 'C'] #vain?
vessel_type = Select(title="Vessel's Type:", options=vessel_types)


def update_vessel_type(attr, old, new):
    vessel_type.value = new


def update_vessel_orient(attr, old, new):
    vessel_orient.value = new


def update_data_channel(attr, old, new):
    data_channel.value = new


def update_num_of_channels(attr, old, new):
    num_of_channels.value = new


data_channel.on_change('value', update_data_channel)
num_of_channels.on_change('value', update_num_of_channels)
vessel_orient.on_change('value', update_vessel_orient)
vessel_type.on_change('value', update_vessel_type)

show_image_button = Button(label='Show Image', button_type="success")


def get_full_path(relative_path: str) -> str:
    """

    :param relative_path:
    :return:
    """
    image_paths = get_file_paths(base_dir_input.value)
    print(f'get_full_path - image paths are:{image_paths}')
    try:
        return [f for f in image_paths if f.endswith(relative_path)][0]
    except IndexError:
        print('Failed to find appropriate image path')


def get_current_file_path() -> str:
    return get_full_path(image_select.value)


def get_current_file_name() -> str:
    return os.path.basename(image_select.value).split('.')[0]


def get_current_metadata() -> dict:
    try:
        return raw_source.data['meta'][0]
    except IndexError:
        return None


def update_time_slider() -> None:
    image = raw_source.data['image'][0]
    if len(image.shape) is 3:
        time_slider.end = image.shape[0] - 1
        time_slider.disabled = False
    else:
        time_slider.end = 1
        time_slider.disabled = True
    time_slider.value = 0


def update_plot_axes(width: int, height: int) -> None:
    plot.x_range.end = width
    plot.y_range.end = height


def get_frame_in_image() -> np.ndarray:
    image = raw_source.data['image'][0]
    if image.ndim == 2:
        image = image[np.newaxis, :]  #increase img dim by 1
    elif image.ndim != 3:
        raise ValueError(f"Image dimension mismatch. Number of dimensions: {image.ndim}")
    return image[time_slider.value, :, :]


def update_plot() -> None:
    image_frame = get_frame_in_image()
    width, height = image_frame.shape[1], image_frame.shape[0]
    image_source.data = dict(image=[image_frame], dw=[width], dh=[height])
    #print('update_plot func. frame is:', frame, 'image_source.data = ', image_source.data) # ,'image_source.data[height]=', image_source.data[height])
    update_plot_axes(width, height)
    db_path = get_db_path()
    if os.path.isfile(db_path):
        draw_existing_rois()
        draw_existing_vectors()
        return
    roi_source.data = dict(x=[], y=[], width=[], height=[])
    vector_source.data = dict(xs=[], ys=[])


def draw_existing_rois() -> None:
    image_full_path = get_full_path(image_select.value)
    print('draw_existing_roi func . image_full_path is:', image_full_path)
    roi_data = get_all_field_in_image_db(get_db_path(), image_full_path, 'roi_coordinates', 'float')
    x = []
    y = []
    width = []
    height = []
    for roi in roi_data:
        x.append(roi[0])
        y.append(roi[1])
        width.append(roi[2])
        height.append(roi[3])
    roi_source.data = dict(x=x, y=y, width=width, height=height)


def draw_existing_vectors() -> None:
    image_full_path = get_full_path(image_select.value)
    vector_data = get_all_field_in_image_db(get_db_path(), image_full_path, 'vector_loc', 'float')
    xs = []
    ys = []
    for vector in vector_data:
        xs.append([vector[0], vector[1]])
        ys.append([vector[2], vector[3]])
    vector_source.data = dict(xs=xs, ys=ys)


def get_frame_rate() -> float:
    meta = get_current_metadata()
    if meta:
        print('get_frame_rate func for x_pixel_to_micron. meta!=none')
        return meta['FrameData']['SI.hRoiManager.scanFrameRate']


def get_line_rate() -> float:
    meta = get_current_metadata()
    if meta:
        line_period = meta['FrameData']['SI.hRoiManager.linePeriod']
        return 1 / line_period


def get_number_of_rows() -> float:
    meta = get_current_metadata()
    if meta:
        return meta['FrameData']['SI.hRoiManager.linesPerFrame']


def get_number_of_columns() -> float:
    meta = get_current_metadata()
    if meta:
        return meta['FrameData']['SI.hRoiManager.pixelsPerLine']


def get_fov() -> float:
    try:
        return int(fov_input.value)
    except ValueError:
        print('Invalid input! FOV must be an integer.')


def get_zoom_factor() -> float:
    meta = get_current_metadata()
    if meta:
        return meta['FrameData']['SI.hRoiManager.scanZoomFactor']


def calc_x_pixel_to_micron() -> float:
    print('calc_x_pixel_to_micron, get_num_cols=', get_number_of_columns(), 'get_fov=', get_fov(), 'get_zoom_factor=', get_zoom_factor())
    try:
        return get_number_of_columns() / (get_fov() * get_zoom_factor())
    except TypeError:
        pass


def calc_y_pixel_to_micron() -> float:
    try:
        return get_number_of_rows() / (get_fov() * get_zoom_factor())
    except TypeError:
        pass


def get_image_shape() -> str:
    #image shape in pixels
    print('get_image_shape func, get_num_of_rows=', get_number_of_rows(), 'get_num_of_cols=', get_number_of_columns())
    return f'{get_number_of_rows()}x{get_number_of_columns()}'


PARAM_GETTERS = {
    'zoom_factor': get_zoom_factor,
    'image_shape': get_image_shape,
    'frame_rate': get_frame_rate,
    'line_rate': get_line_rate,
    'x_pixel_to_micron': calc_x_pixel_to_micron,
    'y_pixel_to_micron': calc_y_pixel_to_micron,
}


def get_string(param: str):
    try:
        value = PARAM_GETTERS[param]()
        if value:
            return str(value)
    except KeyError:
        return 'INVALID'
    return config['null']


def update_parameter_widgets() -> None:
    line_rate_input.value = get_string('line_rate')
    frame_rate_input.value = get_string('frame_rate')
    image_shape_input.value = get_string('image_shape')
    zoom_factor_input.value = get_string('zoom_factor')
    x_pixel_to_micron_input.value = get_string('x_pixel_to_micron')
    y_pixel_to_micron_input.value = get_string('y_pixel_to_micron')


def show_image():# read from folder data and metadata of image and, if exists, info stored in the DB
    print(f'show_image called')
    message_paragraph.style = {'color': 'orange'}
    message_paragraph.text = 'Loading...'
    path = get_full_path(image_select.value)
    raw_source.data = {
        'image': [read_image_file(path, int(data_channel.value) - 1, int(num_of_channels.value))],
        'meta': [read_metadata(path)]
    }
    update_plot()
    update_time_slider()
    update_parameter_widgets()
    message_paragraph.style = {'color': 'green'}
    message_paragraph.text = 'Image successfully loaded!'
    print('show_image done')


show_image_button.on_click(show_image)


def update_frame(attr, old, new):
    message_paragraph.style = {'color': 'orange'}
    message_paragraph.text = 'Loading frame...'
    image = raw_source.data['image'][0]
    if len(image.shape) is 3:
        try:
            image_source.data['image'] = [image[new, :, :]]
        except IndexError:
            print('Failed to update image frame!')
    message_paragraph.style = {'color': 'green'}
    message_paragraph.text = f'Frame {new} successfully loaded!'


time_slider = Slider(start=0, end=1, value=0, step=1, title='Time')
time_slider.on_change('value', update_frame)


def calculate_2d_projection() -> np.ndarray:
    image = raw_source.data['image'][0]
    if len(image.shape) is 3:
        return np.max(image, axis=0)


image_source = ColumnDataSource(
    data=dict(
        image=[],
        x=[],
        y=[],
        dw=[],
        dh=[],
    ))

roi_source = ColumnDataSource(data={
    'x': [],
    'y': [],
    'width': [],
    'height': [],
})
data_xcorr = {}
results_plot_source = ColumnDataSource(data=dict(x=[], y=[]))
cross_correlation_plot_source = ColumnDataSource(data=dict(x=[]))

results_plot = create_results_plot(results_plot_source)
cross_correlation_plot = create_cross_correlation_plot(cross_correlation_plot_source)


def get_roi_vector(roi_index: int):
    n_vectors = len(vector_source.data['xs'])
    for vector_index in range(n_vectors):
        x0, x1, y0, y1 = get_vector_params(vector_index)
        x_start, x_end, y_start, y_end = get_roi_coordinates(get_roi_params(roi_index))
        if not any([x0 < x_start, x1 > x_end, y0 < y_start, y1 > y_end]):
            return vector_index


def change_selected_roi(attr: str, old: str, new: str):
    print(f'change_selected_roi func. new is: {new}, new[0] is {new[0]}')
    roi = get_roi_params(new[0])
    update_res_graph(roi)
    update_cross_correlation_plot(roi, time_slider.value)
    """
    if new:
        roi_index = new[0]
        roi_data = get_roi_data(roi_index, time_slider.value)
        height, width = roi_data.shape[0], roi_data.shape[1]
        roi_shape_input.value = f'{height}x{width}'
        vector_index = get_roi_vector(roi_index)
        if type(vector_index) is int:
            vector_angle_input.value = f'{get_vector_angle_input(vector_index)}'
            vector_source.selected.indices = [vector_index]
            #run_roi_button.disabled = False
        else:
            vector_source.selected.indices = []
            vector_angle_input.value = config['null']
            #run_roi_button.disabled = True
    else:
        roi_shape_input.value = config['null']
        vector_angle_input.value = config['null']
        vector_source.selected.indices = []
        #run_roi_button.disabled = True
    """

roi_source.selected.on_change('indices', change_selected_roi)


def get_roi_data(roi_index: int, frame: int):
    x_start, x_end, y_start, y_end = get_roi_coordinates(get_roi_params(roi_index))
    return get_current_image(get_full_path(image_select.value), int(data_channel.value) - 1, int(num_of_channels.value))[frame, y_start:y_end, x_start:x_end]


vector_source = ColumnDataSource(data={
    'xs': [],
    'ys': [],
})

plot = create_image_figure(image_source, roi_source, vector_source)

save_button = Button(label='Save ROIs', button_type="success")
toggle_2d = Toggle(label='2D Projection')
run_diameter_calc = Button(label='Calc Diameter', button_type="success")


def show_2d_projection(attr, old, new):
    if new is False:
        time_slider.disabled = False
        update_plot()
    else:
        time_slider.disabled = True
        image_source.data['image'] = [calculate_2d_projection()]


toggle_2d.on_change('active', show_2d_projection)

fov_input = TextInput(value='870', title='FOV (μm)')


def update_pixel_to_micron(attr, old, new):
    x_pixel_to_micron_input.value = str(calc_x_pixel_to_micron())
    y_pixel_to_micron_input.value = str(calc_y_pixel_to_micron())


fov_input.on_change('value', update_pixel_to_micron)

zoom_factor_input = TextInput(value=config['null'], title='Zoom Factor')
zoom_factor_input.disabled = False
image_shape_input = TextInput(value=config['null'], title='Image Shape')
image_shape_input.disabled = False
x_pixel_to_micron_input = TextInput(
    value=config['null'], title='Pixel to Micron (X)')
x_pixel_to_micron_input.disabled = False
y_pixel_to_micron_input = TextInput(
    value=config['null'], title='Pixel to Micron (Y)')
y_pixel_to_micron_input.disabled = False
line_rate_input = TextInput(value=config['null'], title='Line Rate (Hz)')
frame_rate_input = TextInput(value=config['null'], title='Frame Rate (Hz)')


def validate_numbers(widget, attr, old, new):
    try:
        float(new)
    except ValueError:
        widget.value = old


min_distance_input = TextInput(
    value=config['min_distance'],
    title='Starting Distance',
)

max_distance_input = TextInput(
    value=config['max_distance'],
    title='Maximum Distance',
)

distance_step_input = TextInput(
    value=config['distance_step'],
    title='Step',
)

distance_step_limit_input = TextInput(
    value=config['distance_step_limit'],
    title='Step Limit',
)

roi_shape_input = TextInput(
    value=config['null'],
    title='ROI Shape',
)
roi_shape_input.disabled = True

vector_angle_input = TextInput(
    value=config['null'],
    title='Vector Angle (Radians)',
)
vector_angle_input.disabled = True

beam_waist_xy_input = TextInput(
    value=config['beam_waist_xy'],
    title='Beam Waist XY (m)',
)
beam_waist_xy_input.on_change(
    'value',
    partial(validate_numbers, beam_waist_xy_input),
)

beam_waist_z_input = TextInput(
    value=config['beam_waist_z'],
    title='Beam Waist Z (m)',
)
beam_waist_z_input.on_change(
    'value',
    partial(validate_numbers, beam_waist_z_input),
)

rbc_radius_input = TextInput(
    value=config['rbc_radius'],
    title='RBC Radius (m)',
)
rbc_radius_input.on_change(
    'value',
    partial(validate_numbers, rbc_radius_input),
)

tau_input = TextInput(
    value=config['tau'],
    title='Pixel Dwell Time [τ] (s)',
)
tau_input.on_change(
    'value',
    partial(validate_numbers, tau_input),
)

distance_sources = dict()


def update_cross_correlation_plot(roi, frame_num):
    xcor_state, xcor_results = get_xcor_results_db(get_db_path(), get_full_path(image_select.value), roi)
    print(f'update_cross_correlation_plot, xcor_state= {xcor_state}')
    if int(xcor_state) != 2:
        return
    start = int(min_distance_input.value)
    stop = int(max_distance_input.value)
    step = int(distance_step_input.value)
    keys = np.arange(start, stop+1, step)
    #keys = np.array([1, 2])
    print('start = {start}, stop = {stop}, step = {step} . keys = {keys}, ')
    xcor_results = xcor_results.reshape(len(xcor_results)/(step+1)*(((stop-start)/step)+1), len(keys), int(distance_step_limit_input.value)+1)
    #xcor_results = xcor_results.reshape(3,2,2)
    print(f'update_cross_correlation_plot func called with roi = {roi}, frame number: {frame_num}, xcor_res={xcor_results}')
    n_columns = len(keys)
    #n_columns = 2
    cross_correlation_plot.x_range.end = n_columns
    filtered_results = dict(zip(keys, xcor_results[frame_num]))
    print(f'filtered res = {filtered_results}')
    for distance in filtered_results:
        current_results = list(filtered_results[distance])
        if distance not in distance_sources:
            distance_index = list(keys).index(distance)
            color = Category20_20[distance_index % 20]
            distance_sources[distance] = ColumnDataSource(
                data=dict(
                    x=range(len(current_results)),
                    y=current_results,
                ))
            cross_correlation_plot.line(
                x='x',
                y='y',
                color=color,
                legend=str(distance),
                source=distance_sources[distance],
                name=f'distance_{distance}_line',
            )
        else:
            distance_sources[distance].data['y'] = current_results
            line = curdoc().get_model_by_name(f'distance_{distance}_line')
            line.visible = True
            line.glyph.line_alpha = 1
    to_hide = [
        distance for distance in distance_sources.keys()
        if distance not in filtered_results
    ]
    for distance in to_hide:
        distance_sources[distance].data.y = [0] * n_columns
        line = curdoc().get_model_by_name(f'distance_{distance}_line')
        line.visible = False
        line.glyph.line_alpha = 0
    cross_correlation_plot.legend.click_policy = 'hide'


def update_res_graph_roi(roi, clear_res_graph):
    print('update_res_graph_roi')
    if clear_res_graph:
    #clear plot somehow!!
        return
    n_frames = time_slider.end
    analysis_status_paragraph.text = 'Updating plot with results...'
    fitting_state, fitting_results = get_fitting_results_db(get_db_path(), get_full_path(image_select.value), roi)
    print('fitting_results=', fitting_results)
    x = np.linspace(1, n_frames+2, n_frames+1)
    y = np.linspace(1, n_frames+2, n_frames+1)
    data = {
        'x_values': x,
        'y_values': y} #fitting_results}
    source = ColumnDataSource(data=data)
    results_plot.circle(x='x_values', y='y_values', source=source)


def is_res_ready(roi: np.array) -> bool:
    print('is_res_ready called. roi = ', roi)
    db_path = get_db_path()
    if os.path.isfile(db_path):
        fitting_state = get_fitting_results_db(db_path, get_full_path(image_select.value), roi)
        print('is_res_ready. fitting_state=', fitting_state)
        if fitting_state[0] == 2:
            print('is_res_ready returns True. fitting_state=0')
            return True
    print('is_res_ready returns False. fitting_state!=1')
    return False


def update_res_graph(roi):
    print(f'update_res_graph function. roi is: {roi}')
    if is_res_ready(roi):
        print(f'update_res_graph. roi={roi} , result is ready')
        update_res_graph_roi(roi, False)
        print('update_res_graph done')
        return
    update_res_graph_roi(None, True) #clear results graph
    analysis_status_paragraph.text = 'Results being processed, Try again later'
    print('update_res_graph done')


message_paragraph = Paragraph(text='')
analysis_status_paragraph = Paragraph(text='')


def get_roi_params(index: int) -> np.array:
    values = roi_source.data
    return np.array([
        values['x'][index],
        values['y'][index],
        values['width'][index],
        values['height'][index]
    ])


def get_vector_params(index: int) -> np.array:
    values = vector_source.data
    return np.array([
        values['xs'][index][0],
        values['xs'][index][1],
        values['ys'][index][0],
        values['ys'][index][1]
    ])


def get_vector_data_by_index() -> np.ndarray:
    vector_data = np.full((50, 4), np.nan)
    n_vectors = len(vector_source.data['xs'])
    for index in range(n_vectors):
        vector_data[index, :] = get_vector_params(index)
    return vector_data


def get_vector_angle_input(index: int) -> float:
    x0, x1, y0, y1 = get_vector_params(index)
    return math.atan2(y1 - y0, x1 - x0)


def create_data_dict(roi_index: int) -> dict:
    print('create_data_dict, data_channel =' ,(int(data_channel.value) -1),
        'num_of_data_channels=', int(num_of_channels.value))
    num_frames = int(time_slider.end)
    num_keys = int((int(max_distance_input.value)-int(min_distance_input.value))/int(distance_step_input.value))+1
    xcor_len = num_frames*num_keys*int(distance_step_limit_input.value)+1 #number of frames * number of keys * num of values in each xcor
    return {
        'path': get_full_path(image_select.value),
        'frame_rate': float(frame_rate_input.value),
        'data_channel' : (int(data_channel.value) - 1),
        'num_of_data_channels' : int(num_of_channels.value),
        'line_rate': float(line_rate_input.value),
        'magnification': float(get_zoom_factor()),
        'vessel_orientation': str(vessel_orient.value),
        'vessel_type': str(vessel_type.value),
        'axis_len_microns': float(get_fov()),
        'x_pixel_to_micron': float(calc_x_pixel_to_micron()),
        'y_pixel_to_micron': float(calc_y_pixel_to_micron()),
        'n_frames': num_frames,
        'min_distance': int(min_distance_input.value),
        'max_distance': int(max_distance_input.value),
        'distance_step': int(distance_step_input.value),
        'distance_step_limit': int(distance_step_limit_input.value),
        'vector_angle': get_vector_angle_input(roi_index),
        '_2d_projection': calculate_2d_projection(),
        'corr_calc_config': 0,
        'corr_analysis_state': 0,
        'correlation_results': np.zeros((1,xcor_len)),
        'fitting_params': 0,
        'fitting_results': np.array((1, num_frames)),
        #'vessel_diameters': np.array((1, num_frames)),
        'corr_calc_state': 0,
        'fitting_state': 0, #0= ready for backend processing. 1= processing. 2= processing done. 3= processing error
        'diameter_analysis_state': 0, #0= ready for backend processing. 1= processing. 2= processing done. 3= processing error
        'diameter_results': np.array((1, num_frames)),
        'roi_coordinates': get_roi_params(roi_index), #['x_start', 'x_end', 'y_start', 'y_end']
        'vector_loc': get_vector_params(roi_index), #['x_start', 'x_end', 'y_start', 'y_end']
        'beam_waist_xy': beam_waist_xy_input.value,
        'beam_waist_z': beam_waist_z_input.value,
        'rbc_radius': rbc_radius_input.value,
        'pixel_dwell_time': tau_input.value,
    }


def create_coords_dict(path: str) -> dict:
    pass

    image = raw_source.data['image'][0]
    return {
        #'pix_x': np.arange(image.shape[2]),
        #'pix_y': np.arange(image.shape[1]),
        #'time': np.arange(image.shape[0]),
        #'distance': np.arange(0, 301, 20)
    }


def save():
    print('save function called')
    session = connect_to_db(get_db_path(), False)
    add_image_to_db(session)


def diameter_calc():
    print('diameter calc')

def add_image_to_db(session):
    #this function is called when user presses 'save'
    #we add each roi in the image as a line to the db
    n_rois_current = len(roi_source.data['x'])
    print('add_image_to_db func, n_rois_current=',n_rois_current)
    image_full_path = get_full_path(image_select.value)
    handle_rois_added_by_user(n_rois_current, session, image_full_path)
    handle_rois_deleted_by_user(session, n_rois_current, image_full_path)


def handle_rois_added_by_user(n_rois_current: int, session, image_full_path: str):
    """"
    go over list of rois per image received from user
    check for every roi if exists in DB
    add only non existing rois to db
    """
    print('handle_rois_added_by_user')
    for roi_index in range(n_rois_current):
        roi_exist = check_col_image_exist_in_db(get_db_path(), image_full_path, 'roi_coordinates', get_roi_params(roi_index), 'float')
        if not roi_exist:
            add_new_roi_to_db(roi_index, session)


def handle_rois_deleted_by_user(session, n_rois_current: int, image_full_path: str):
    """
    #go over list of rois in DB, check for each roi if exists
    #if roi exists in db but not in image_source then delete it from the db
    :param session:
    :param n_rois_current:
    :param image_full_path:
    :return:
    """
    roi_db_list = get_all_field_in_image_db(get_db_path(), image_full_path, 'roi_coordinates', 'float')
    for roi_db in roi_db_list:
        roi_db_parsed = np.frombuffer(roi_db, dtype=np.float)
        delete_roi_from_db = True
        for roi_index in range(n_rois_current):
            if np.allclose(roi_db_parsed, get_roi_params(roi_index)):
                delete_roi_from_db = False
        if delete_roi_from_db:
            delete_row_in_db(session, image_full_path, np.array(roi_db))


def add_new_roi_to_db(roi_index: int, session):
    data_vars, coords = get_parameters(roi_index)
    data = Data(**data_vars)
    session.add(data)
    session.commit()


img_path_angoli = r'd:\git\flics\flics_data\imges\Angoli_vena_conventional_Series012t5_6gradi.tif'


def get_parameters(roi_index: int):
    image_path = get_full_path(image_select.value)
    if image_path == img_path_angoli:
        data_vars = create_angoli_data_dict(roi_index, get_vector_angle_input(roi_index), get_roi_params(roi_index), get_vector_params(roi_index), 0.2, beam_waist_z_input.value,
            rbc_radius_input.value, tau_input.value)
    else:
        data_vars = create_data_dict(roi_index)
    coords = create_coords_dict(image_path)
    return data_vars, coords


save_button.on_click(save)
run_diameter_calc.on_click(diameter_calc)

main_layout = row(
    widgetbox(
        base_dir_input,
        image_select,
        data_channel,
        num_of_channels,
        show_image_button,
        message_paragraph,
        time_slider,
        image_shape_input,
        zoom_factor_input,
        x_pixel_to_micron_input,
        y_pixel_to_micron_input,
        fov_input,
        line_rate_input,
        frame_rate_input,
        toggle_2d,
        save_button,
        run_diameter_calc,
    ),
    column(
        row(roi_shape_input, vector_angle_input),
        plot,
    ),
    column(
        row(beam_waist_xy_input, beam_waist_z_input),
        row(rbc_radius_input, tau_input),
        results_plot,
        row(min_distance_input, max_distance_input),
        row(distance_step_input, distance_step_limit_input),
        cross_correlation_plot,
        analysis_status_paragraph,
        #run_roi_button,
    ),
    name='main_layout',
)

doc = curdoc()
doc.add_root(main_layout)
