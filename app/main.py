import numpy as np
import os
import tifffile

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, )
from bokeh.models.widgets import (
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

raw_source = ColumnDataSource(data=dict(image=[]))

image_select = Select(title='Image', value='---', options=['---'])


def get_full_path(relative_path: str) -> str:
    image_paths = get_file_paths()
    try:
        return [f for f in image_paths if f.endswith(relative_path)][0]
    except IndexError:
        print('Failed to find appropriate image path')


def read_image_file(path: str) -> np.ndarray:
    with tifffile.TiffFile(path, movie=True) as f:
        try:
            data = f.asarray(slice(1, None, 2))  # read the second channel only
        except ValueError:
            data = f.asarray()
        return data


def get_current_image() -> np.ndarray:
    return raw_source.data['image'][0]


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


def update_plot() -> None:
    frame = get_current_frame()
    width, height = frame.shape[1], frame.shape[0]
    image_source.data = dict(image=[frame], dw=[width], dh=[height])
    update_plot_axes(width, height)


def read_metadata(path: str) -> dict:
    with tifffile.TiffFile(path, movie=True) as f:
        return f.scanimage_metadata


def select_image(attr, old, new):
    path = get_full_path(new)
    image = read_image_file(path)
    raw_source.data['image'] = [image]
    update_time_slider()
    update_plot()
    meta = read_metadata(path)
    n_frames = image.shape[0]
    linerate = 1 / meta['FrameData']['SI.hRoiManager.linePeriod']
    fps = meta['FrameData']['SI.hRoiManager.scanFrameRate']


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
        return np.mean(image, axis=0)


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

plot = create_image_figure(image_source, roi_source)

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
    ),
    plot,
    name='main_layout',
)

curdoc().add_root(main_layout)
