import math
import numpy as np
import os
import pathlib
import tifffile
import xarray as xr

from app.schema import connect_to_db, Data, get_column_image_db, check_col_image_exist_in_db, delete_row_in_db

from app.analysis.flics import Analysis
from app.analysis.global_fit import GlobalFit
from bokeh.io import curdoc
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
from app.figures.image_plot import create_image_figure
from app.figures.results_plot import (
    create_results_plot,
    create_cross_correlation_plot,
)
from functools import partial

base_dir_input = TextInput(value=r'd:\git\flics\flics_data\\', title='Base Directory')
IMAGE_EXT = '.tif'

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
}


def get_file_paths(ext: str = IMAGE_EXT) -> list:
    """
    Returns a list of files ending with 'ext' within a directory tree.
    
    :param ext: The target file extension, defaults to IMAGE_EXT.
    :type ext: str
    :return: A list of file paths.
    :rtype: list
    """
    image_paths = []
    for root, dirs, files in os.walk(base_dir_input.value):
        for f in files:
            if f.endswith(ext):
                image_paths.append(os.path.join(root, f))
    return image_paths


def update_image_select(attr, old, new): #called when user changes basedir or image
    if os.path.isdir(new):
        image_paths = get_file_paths()
        relative_paths = [path[len(new):] for path in image_paths]
        if relative_paths:
            image_select.options = relative_paths
            image_select.value = image_select.options[0]
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
        return data_folder / 'db.db'


base_dir_input.on_change('value', update_image_select)

raw_source = ColumnDataSource(data=dict(image=[], meta=[]))

image_select = Select(
    title='Image', value=config['null'], options=[config['null']])


def get_full_path(relative_path: str) -> str:
    image_paths = get_file_paths()
    try:
        return [f for f in image_paths if f.endswith(relative_path)][0]
    except IndexError:
        print('Failed to find appropriate image path')


def get_current_file_path() -> str:
    return get_full_path(image_select.value)


def get_current_file_name() -> str:
    return os.path.basename(image_select.value).split('.')[0]


def get_current_metset_path() -> str:
    location = os.path.dirname(get_current_file_path())
    return os.path.join(location, get_current_file_name() + '.nc')


def read_image_file(path: str) -> np.ndarray:
    with tifffile.TiffFile(path, movie=True) as f:
        try:
            data = f.asarray(slice(1, None, 2))  # read the second channel only
        except ValueError:
            data = f.asarray()
        return data


def get_current_image() -> np.ndarray:
    img = raw_source.data['image'][0]
    if img.ndim == 2:
        return img[np.newaxis, :]
    elif img.ndim == 3:
        return img
    else:
        raise ValueError(f"Image dimension mismatch. Number of dimensions: {img.ndim}")


def get_current_metadata() -> dict:
    try:
        return raw_source.data['meta'][0]
    except IndexError:
        return None


def update_time_slider() -> None:
    image = get_current_image()
    if len(image.shape) is 3:
        time_slider.end = image.shape[0] - 1
        time_slider.disabled = False
    else:
        time_slider.end = 1
        time_slider.disabled = True
    time_slider.value = 0


def get_current_frame() -> np.ndarray:
    image = get_current_image()
    if len(image.shape) is 3:
        return image[time_slider.value, :, :]
    return image


def update_plot_axes(width: int, height: int) -> None:
    plot.x_range.end = width
    plot.y_range.end = height


def update_plot() -> None:
    frame = get_current_frame()
    width, height = frame.shape[1], frame.shape[0]
    image_source.data = dict(image=[frame], dw=[width], dh=[height])
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
    roi_data = get_column_image_db(get_db_path(), image_full_path, 'roi_coordinates', 'float')
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
    vector_data = get_column_image_db(get_db_path(), image_full_path, 'vector_loc', 'float')
    xs = []
    ys = []
    for vector in vector_data:
        xs.append([vector[0], vector[1]])
        ys.append([vector[2], vector[3]])
    vector_source.data = dict(xs=xs, ys=ys)


def read_metadata(path: str) -> dict:
    with tifffile.TiffFile(path, movie=True) as f:
        return f.scanimage_metadata


def get_frame_rate() -> float:
    meta = get_current_metadata()
    if meta:
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


def select_image(attr, old, new): # read from db prev data and metadata of image if exists
    message_paragraph.style = {'color': 'orange'}
    message_paragraph.text = 'Loading...'
    path = get_full_path(new)
    raw_source.data = {
        'image': [read_image_file(path)],
        'meta': [read_metadata(path)]
    }
    update_time_slider()
    update_plot()
    update_parameter_widgets()
    n_frames = time_slider.end + 1
    results_plot_source.data = dict(x=range(n_frames), y=[0] * n_frames)
    message_paragraph.style = {'color': 'green'}
    message_paragraph.text = 'Image successfully loaded!'


image_select.on_change('value', select_image)

time_slider = Slider(start=0, end=1, value=0, step=1, title='Time')


def update_frame(attr, old, new):
    image = get_current_image()
    if len(image.shape) is 3:
        try:
            image_source.data['image'] = [image[new, :, :]]
        except IndexError:
            print('Failed to update image frame!')


def calculate_2d_projection() -> np.ndarray:
    image = get_current_image()
    if len(image.shape) is 3:
        return np.max(image, axis=0)


time_slider.on_change('value', update_frame)

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

results_plot_source = ColumnDataSource(data=dict(x=[], y=[]))

results_plot = create_results_plot(results_plot_source)

cross_correlation_plot_source = ColumnDataSource(
    data=dict(
        xs=[],
        ys=[],
        color=[],
        alpha=[],
    ))

cross_correlation_plot = create_cross_correlation_plot(
    cross_correlation_plot_source)


def get_roi_vector(roi_index: int):
    n_vectors = len(vector_source.data['xs'])
    for vector_index in range(n_vectors):
        x0, x1, y0, y1 = get_vector_params(vector_index)
        x_start, x_end, y_start, y_end = get_roi_coordinates(roi_index)
        if not any([x0 < x_start, x1 > x_end, y0 < y_start, y1 > y_end]):
            return vector_index


def change_selected_roi(attr, old, new):
    if new:
        roi_index = new[0]
        roi_data = get_roi_data(roi_index, time_slider.value)
        height, width = roi_data.shape[0], roi_data.shape[1]
        roi_shape_input.value = f'{height}x{width}'
        vector_index = get_roi_vector(roi_index)
        if type(vector_index) is int:
            vector_angle_input.value = f'{get_vector_angle_input(vector_index)}'
            vector_source.selected.indices = [vector_index]
            run_roi_button.disabled = False
        else:
            vector_source.selected.indices = []
            vector_angle_input.value = config['null']
            run_roi_button.disabled = True
    else:
        roi_shape_input.value = config['null']
        vector_angle_input.value = config['null']
        vector_source.selected.indices = []
        run_roi_button.disabled = True


roi_source.selected.on_change('indices', change_selected_roi)


def get_roi_coordinates(roi_index: int):
    width = round(roi_source.data['width'][roi_index])
    height = round(roi_source.data['height'][roi_index])
    x_center = round(roi_source.data['x'][roi_index])
    y_center = round(roi_source.data['y'][roi_index])
    x_start = int(x_center - width // 2)
    x_end = int(x_center + width // 2)
    x_start, x_end = sorted([x_start, x_end])
    y_start = int(y_center - height // 2)
    y_end = int(y_center + height // 2)
    y_start, y_end = sorted([y_start, y_end])
    return x_start, x_end, y_start, y_end


def get_roi_data(roi_index: int, frame: int):
    x_start, x_end, y_start, y_end = get_roi_coordinates(roi_index)
    return get_current_image()[frame, y_start:y_end, x_start:x_end]


vector_source = ColumnDataSource(data={
    'xs': [],
    'ys': [],
})

plot = create_image_figure(image_source, roi_source, vector_source)

save_button = Button(label='Save', button_type="success")

toggle_2d = Toggle(label='2D Projection')


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
zoom_factor_input.disabled = True
image_shape_input = TextInput(value=config['null'], title='Image Shape')
image_shape_input.disabled = True
x_pixel_to_micron_input = TextInput(
    value=config['null'], title='Pixel to Micron (X)')
x_pixel_to_micron_input.disabled = True
y_pixel_to_micron_input = TextInput(
    value=config['null'], title='Pixel to Micron (Y)')
y_pixel_to_micron_input.disabled = True
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


def update_cross_correlation_plot(results: dict):
    n_columns = len(list(results.values())[0])
    cross_correlation_plot.x_range.end = n_columns
    filtered_results = {
        key: value
        for key, value in results.items()
        if isinstance(value, (np.ndarray, list))
    }
    for distance in filtered_results:
        current_results = filtered_results[distance]
        if distance not in distance_sources:
            distance_index = list(results.keys()).index(distance)
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


def run_flics_on_roi():
    roi_index = roi_source.selected.indices[0]
    n_frames = time_slider.end
    cross_correlation_results = []
    fitting_results = []
    for i_frame in range(n_frames + 1):
        time_slider.value = i_frame
        analysis_status_paragraph.text = f'Retrieving ROI[{roi_index}] data for frame #{i_frame}...'
        data = get_roi_data(roi_index, i_frame)
        analysis_status_paragraph.text = 'Calculating column cross-correlation...'
        flics_analysis = Analysis(
            image=data,
            min_distance=int(min_distance_input.value),
            max_distance=int(max_distance_input.value) + 1,
            distance_step=int(distance_step_input.value),
        )
        update_cross_correlation_plot(flics_analysis.results)
        cross_correlation_results.append(list(flics_analysis.results.values()))
        analysis_status_paragraph.text = 'Calculating global fit...'
        global_fitting = GlobalFit(flics_analysis.results)
        angle = float(vector_angle_input.value)
        dx = float(x_pixel_to_micron_input.value) * 1e-6
        frame_results = global_fitting.run(
            angle=angle,
            pixel_to_micron_x=dx,
            beam_waist_xy=float(beam_waist_xy_input.value),
            beam_waist_z=float(beam_waist_z_input.value),
            rbc_radius=float(rbc_radius_input.value),
            tau=float(tau_input.value),
        )
        v = frame_results.params['v_']
        fitting_results.append(v)
        analysis_status_paragraph.text = 'Updating plot with results...'
        results_plot_source.data = dict(
            x=range(i_frame + 1),
            y=fitting_results,
        )
        results_plot.x_range.end = i_frame
        results_plot.y_range.start = min(fitting_results)
        results_plot.y_range.end = max(fitting_results)

    import pickle
    with open('cross_corr.bin', 'wb') as cross_corr_file:
        pickle.dump(cross_correlation_results, cross_corr_file)
    with open('fitting.bin', 'wb') as fitting_file:
        pickle.dump(fitting_results, fitting_file)
    db_path = get_db_path()
    with xr.open_dataset(db_path) as db: ### reut
        path = get_full_path(image_select.value)
        fitting_results = (['time'], fitting_results)
        cross_correlation_results = (['time', 'distance'],
                                     cross_correlation_results)
        # TODO: Needs to be associated with an ROI
        db.loc[{
            'path': path
        }].assign(cross_correlation_results=cross_correlation_results)
        db.loc[{'path': path}].assign(fitting_results=fitting_results)
        os.remove(db_path)
        db.to_netcdf(db_path, mode='w')


run_roi_button = Button(label='Run ROI', button_type='primary')
run_roi_button.disabled = True
run_roi_button.on_click(run_flics_on_roi)

run_flics_button = Button(label='Run FLICS', button_type='primary')

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
    return {
        'path': get_full_path(image_select.value),
        'frame_rate': float(frame_rate_input.value),
        'line_rate': float(line_rate_input.value),
        'x_pixel_to_micron': float(calc_x_pixel_to_micron()),
        'y_pixel_to_micron': float(calc_y_pixel_to_micron()),
        '_2d_projection': calculate_2d_projection(),
        'corr_calc_config': 0,
        'corr_analysis_state': 0,
        'correlation_results':  np.array([0, 1, 2, 3]),
        'fitting_params': 0,
        'fitting_analysis_state': 0,
        'fitting_results': np.array([34, 34, 2322]),
        'vessel_diameters': np.array([[404040, 32323], [30232, 23232]]),
        'corr_calc_state': 0,
        'fitting_state': 0,
        'roi_coordinates': get_roi_params(roi_index),
        'vector_loc': get_vector_params(roi_index), #['x_start', 'x_end', 'y_start', 'y_end']
        'distance': np.arange(0, 301, 20),
        'step': 0,
        'delete_row': False
    }


def create_coords_dict(path: str) -> dict:
    image = get_current_image()
    return {
        #'pix_x': np.arange(image.shape[2]),
        #'pix_y': np.arange(image.shape[1]),
        #'time': np.arange(image.shape[0]),
        #'distance': np.arange(0, 301, 20)
    }


def save():
    session = connect_to_db(get_db_path(), False)
    add_image_to_db(session)


def add_image_to_db(session):
    #this function is called after an image and its rois are choosen
    #we add each roi as a line to the db
    n_rois_current = len(roi_source.data['x'])
    image_full_path = get_full_path(image_select.value)
    handle_rois_added_by_user(n_rois_current, session, image_full_path)
    handle_rois_deleted_by_user(session, n_rois_current, image_full_path)


def handle_rois_added_by_user(n_rois_current: int, session, image_full_path: str):
    """"
    go over list of rois per image recieved from user
    check for every roi if exists in DB
    add only non existing rois to db
    """
    for roi_index in range(n_rois_current):
        roi_exist = check_col_image_exist_in_db(get_db_path(), image_full_path, 'roi_coordinates', get_roi_params(roi_index), 'float')
        if not roi_exist:
            add_new_roi_to_db(roi_index, session)


def handle_rois_deleted_by_user(session, n_rois_current: int, image_full_path: str):
    #go over list of rois in DB, check for each roi if exists
    #if roi exists in db but not in image_source then delete it from the db
    roi_db_list = get_column_image_db(get_db_path(), image_full_path, 'roi_coordinates', 'float')
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


def get_parameters(roi_index: int):
    image_path = get_full_path(image_select.value)
    data_vars = create_data_dict(roi_index)
    coords = create_coords_dict(image_path)
    return data_vars, coords


save_button.on_click(save)

main_layout = row(
    widgetbox(
        base_dir_input,
        image_select,
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
    ),
    column(
        row(roi_shape_input, vector_angle_input),
        plot,
    ),
    column(
        row(beam_waist_xy_input, beam_waist_z_input),
        row(rbc_radius_input, tau_input),
        results_plot,
        row(min_distance_input, max_distance_input, distance_step_input),
        cross_correlation_plot,
        analysis_status_paragraph,
        run_roi_button,
    ),
    name='main_layout',
)

curdoc().add_root(main_layout)
