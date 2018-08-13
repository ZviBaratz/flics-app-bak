import bokeh.plotting as bp
import json
import numpy as np

from bokeh.core.json_encoder import serialize_json
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    BoxEditTool,
)
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.io import curdoc
from data_access.data_access_object import DataAccessObject

dao = DataAccessObject()
data = dict(
    images=[image.name for image in dao.images],
    shapes=[image.data.shape for image in dao.images],
)
data_source = ColumnDataSource(data=data)


def create_data_table(dao: DataAccessObject):
    columns = [
        TableColumn(field='images', title='Image'),
        TableColumn(field='shapes', title='Shape'),
    ]
    return DataTable(
        source=data_source,
        columns=columns,
        width=400,
        height=800,
    )


data_table = create_data_table(dao)


def get_selection_index():
    return data_table.source.selected.indices[0]


def get_selection_name():
    selection_index = get_selection_index()
    return data_table.source.data['images'][selection_index]


def get_selection_image():
    return dao.get_image(get_selection_name())


data_source.selected.indices = [0]

selected_image = get_selection_image()
image_source = ColumnDataSource(data=dict(image=[selected_image.data]))

roi_source = ColumnDataSource(data={})
roi_source.data = dao.get_roi_source(selected_image)


def create_image_figure(image: np.ndarray):
    image = image_source.data['image'][0]

    # Create figure
    plot = bp.figure(
        plot_width=image.shape[1],
        plot_height=image.shape[0],
        x_range=[0, image.shape[1]],
        y_range=[0, image.shape[0]],
        title='Selected Image',
        name='image_figure',
    )

    # Plot image
    plot.image(
        image='image',
        x=0,
        y=0,
        dw=image.shape[1],
        dh=image.shape[0],
        source=image_source,
        palette='Spectral11',
        name='image_plot',
    )

    # Create hover tool
    hover = HoverTool(tooltips=[('x', '$x'), ('y', '$y'), ('value', '@image')])

    # Create ROI selection renderer
    r1 = plot.rect(
        x='x',
        y='y',
        width='width',
        height='height',
        source=roi_source,
        fill_alpha=0.5,
        fill_color='#DAF7A6',
        # dilate=True,
        name='rois',
    )

    # Add tools
    plot.tools = [
        hover,
        BoxEditTool(renderers=[r1]),
    ]

    return plot


plot = create_image_figure(selected_image.data)


def update_image_plot(attr, old, new):
    image = get_selection_image()
    image_source.data['image'] = [image.get_data()]
    roi_source.data = dao.get_roi_source(image)


data_source.selected.on_change('indices', update_image_plot)


def create_roi(attr, old, new):
    image = get_selection_image()
    jsoned = json.loads(serialize_json(roi_source.data))
    with open(image.rois_file_path, 'w') as roi_file:
        json.dump(jsoned, roi_file)


roi_source.on_change('data', create_roi)

layout = row(column(data_table), plot, name='main_layout')
curdoc().add_root(layout)
