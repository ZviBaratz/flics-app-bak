import numpy as np
import os
import tifffile
import xarray as xr

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, )
from bokeh.models.widgets import (
    Button,
    Select,
    Slider,
    TextInput,
    Toggle,
)
from figures.image_plot import create_image_figure

base_dir_input = TextInput(value='Enter path...', title='Base Directory')
IMAGE_EXT = '.tif'


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


def update_image_select(attr, old, new):
    if os.path.isdir(new):
        image_paths = get_file_paths()
        relative_paths = [path[len(new):] for path in image_paths]
        if relative_paths:
            image_select.options = relative_paths
            return
    image_select.options = ['---']


base_dir_input.on_change('value', update_image_select)

raw_source = ColumnDataSource(data=dict(image=[], meta=[]))

image_select = Select(title='Image', value='---', options=['---'])


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


def read_current_metaset() -> xr.Dataset:
    return xr.open_dataset(get_current_metset_path())


def read_image_file(path: str) -> np.ndarray:
    with tifffile.TiffFile(path, movie=True) as f:
        try:
            data = f.asarray(slice(1, None, 2))  # read the second channel only
        except ValueError:
            data = f.asarray()
        return data


def get_current_image() -> np.ndarray:
    return raw_source.data['image'][0]


def has_metaset() -> bool:
    return os.path.isfile(get_current_metset_path())


def get_current_metadata() -> dict:
    return raw_source.data['meta'][0]


def update_time_slider() -> None:
    image = get_current_image()
    if len(image.shape) is 3:
        time_slider.end = image.shape[0] - 1
        time_slider.disabled = False
    else:
        time_slider.end = 1
        time_slider.disabled = True


def get_current_frame() -> np.ndarray:
    image = get_current_image()
    if len(image.shape) is 3:
        return image[time_slider.value, :, :]
    return image


def update_plot_axes(width: int, height: int) -> None:
    plot.x_range.end = width
    plot.y_range.end = height


def draw_existing_rois() -> None:
    ds = read_current_metaset()
    x = ds['roi_x'].values.tolist()
    y = ds['roi_y'].values.tolist()
    width = ds['roi_width'].values.tolist()
    height = ds['roi_height'].values.tolist()
    roi_source.data = dict(x=x, y=y, width=width, height=height)


def draw_existing_vectors() -> None:
    ds = read_current_metaset()
    x_starts = ds['vector_x_start'].values.tolist()
    x_ends = ds['vector_x_end'].values.tolist()
    y_starts = ds['vector_y_start'].values.tolist()
    y_ends = ds['vector_y_end'].values.tolist()
    xs = [list(t) for t in list(zip(x_starts, x_ends))]
    ys = [list(t) for t in list(zip(y_starts, y_ends))]
    vector_source.data = dict(xs=xs, ys=ys)


def update_plot() -> None:
    frame = get_current_frame()
    width, height = frame.shape[1], frame.shape[0]
    image_source.data = dict(image=[frame], dw=[width], dh=[height])
    update_plot_axes(width, height)
    if has_metaset():
        draw_existing_rois()
        draw_existing_vectors()
    else:
        roi_source.data = dict(x=[], y=[], width=[], height=[])
        vector_source.data = dict(xs=[], ys=[])


def read_metadata(path: str) -> dict:
    with tifffile.TiffFile(path, movie=True) as f:
        return f.scanimage_metadata


def get_fps() -> float:
    meta = get_current_metadata()
    return meta['FrameData']['SI.hRoiManager.scanFrameRate']


def get_line_rate() -> float:
    meta = get_current_metadata()
    line_period = meta['FrameData']['SI.hRoiManager.linePeriod']
    return 1 / line_period


def select_image(attr, old, new):
    path = get_full_path(new)
    raw_source.data = {
        'image': [read_image_file(path)],
        'meta': [read_metadata(path)]
    }
    update_time_slider()
    update_plot()


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

vector_source = ColumnDataSource(data={
    'xs': [],
    'ys': [],
})

plot = create_image_figure(image_source, roi_source, vector_source)


def save():
    image = get_current_image()
    path = get_full_path(image_select.value)
    data_vars = {
        '2d_projection': (['x', 'y'], calculate_2d_projection()),
    }
    coords = {
        'path': path,
        'x': np.arange(image.shape[2]),
        'y': np.arange(image.shape[1]),
        'time': np.arange(image.shape[0]),
        'roi': np.arange(len(roi_source.data['x'])),
        'roi_x': roi_source.data['x'],
        'roi_y': roi_source.data['y'],
        'roi_width': roi_source.data['width'],
        'roi_height': roi_source.data['height'],
        'vector': np.arange(len(vector_source.data['xs'])),
        'vector_x_start': [v[0] for v in vector_source.data['xs']],
        'vector_x_end': [v[1] for v in vector_source.data['xs']],
        'vector_y_start': [v[0] for v in vector_source.data['ys']],
        'vector_y_end': [v[1] for v in vector_source.data['ys']],
        'fps': get_fps(),
        'line_rate': get_line_rate(),
    }
    ds = xr.Dataset(data_vars, coords)
    dest = get_current_metset_path()
    ds.to_netcdf(dest)


save_button = Button(label='Save', button_type="success")
save_button.on_click(save)

toggle_2d = Toggle(label='2D Projection')


def show_2d_projection(attr, old, new):
    if new is False:
        time_slider.disabled = False
        update_plot()
    else:
        time_slider.disabled = True
        image_source.data['image'] = [calculate_2d_projection()]


toggle_2d.on_change('active', show_2d_projection)

main_layout = row(
    column(
        base_dir_input,
        image_select,
        time_slider,
        toggle_2d,
        save_button,
    ),
    plot,
    name='main_layout',
)

curdoc().add_root(main_layout)
