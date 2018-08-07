import bokeh.plotting as bp
import numpy as np

from bokeh.layouts import column, row
# from bokeh.layouts import widgetbox
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    BoxSelectTool,
    BoxEditTool,
    LassoSelectTool,
    PolySelectTool,
)
from bokeh.models.widgets import DataTable, TableColumn, Div
from bokeh.io import curdoc
from data_access.data_access_object import DataAccessObject
# from data_access.data_object import DataObject

dao = DataAccessObject()
data = dict(
    images=[image.name for image in dao.images],
    shapes=[image.data.shape for image in dao.images],
)
data_source = ColumnDataSource(data)
roi_source = ColumnDataSource(data=dict(
    x=[],
    y=[],
    width=[],
    height=[],
))

sources_dict = {
    'data': data_source,
    'selected_image': None,
    'roi': roi_source,
}


def plot_image(image: np.ndarray):

    # Create data source
    source = ColumnDataSource(data=dict(image=[image]))
    sources_dict['selected_image'] = source

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
        source=source,
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
        source=sources_dict['roi'],
        fill_alpha=0.5,
        fill_color='#DAF7A6',
        dilate=True,
        name='rois',
    )

    # Add tools
    plot.add_tools(
        hover,
        BoxSelectTool(),
        BoxEditTool(renderers=[r1]),
        LassoSelectTool(),
        PolySelectTool(),
    )

    return plot


def create_data_table(dao: DataAccessObject):
    # Basic data table to choose image
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


def update_image_plot(attr, old, new):
    selected_image = data_table.source.data['images'][
        data_table.source.selected.indices[0]]
    image = [image for image in dao.images if image.name == selected_image][0]
    sources_dict['selected_image'].data.update(image.get_data())


def create_roi(attr, old, new):
    print(sources_dict['roi'].data)


coord = Div(text='Coordinates:\n')
sources_dict['data'].on_change('selected', update_image_plot)
plot = plot_image(dao.images[0].get_data())
sources_dict['roi'].on_change('selected', create_roi)
data_table = create_data_table(dao)
layout = row(column(data_table, coord), plot, name='main_layout')
curdoc().add_root(layout)
