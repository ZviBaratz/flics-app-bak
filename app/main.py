import bokeh.plotting as bp
import json
import numpy as np

from bokeh.core.json_encoder import serialize_json
from bokeh.layouts import column, row, widgetbox
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    BoxEditTool,
    NumberFormatter,
)
from bokeh.models.widgets import DataTable, TableColumn, Div
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
        height=300,
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
data = selected_image.data
image_source = ColumnDataSource(
    data=dict(
        image=[data],
        x=[0],
        y=[0],
        dw=[data.shape[1]],
        dh=[data.shape[0]],
    ))

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
        dw='dw',
        dh='dh',
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
    data = image.get_data()
    image_source.data['image'] = [data]
    image_source.data['dw'] = [data.shape[1]]
    image_source.data['dh'] = [data.shape[0]]
    plot.x_range.end = data.shape[1]
    plot.y_range.end = data.shape[0]
    roi_source.data = dao.get_roi_source(image)


data_source.selected.on_change('indices', update_image_plot)


def get_roi_properties(index: int) -> tuple:
    x = roi_source.data['x'][index]
    y = roi_source.data['y'][index]
    width = roi_source.data['width'][index]
    height = roi_source.data['height'][index]
    return x, y, width, height


def get_roi_data(index: int) -> np.ndarray:
    x, y, width, height = get_roi_properties(index)
    image_data = image_source.data['image'][0]
    x_start = int(x - 0.5 * width)
    x_end = int(x + 0.5 * width)
    y_start = int(y - 0.5 * height)
    y_end = int(y + 0.5 * height)
    return image_data[y_start:y_end, x_start:x_end]


def create_roi(attr, old, new):
    image = get_selection_image()
    jsoned = json.loads(serialize_json(roi_source.data))
    with open(image.rois_file_path, 'w') as roi_file:
        json.dump(jsoned, roi_file)
    roi_table_source.data = roi_source.data


def select_roi_in_plot(attr, old, new):
    print(f'Selected indices {new} in ROI table.')
    current_roi_index = roi_source.selected.indices
    if current_roi_index != new:
        print('Updating selected ROI in the image plot...')
        try:
            roi_source.selected.indices = new
        except Exception as e:
            print('Failed to update ROI source with the following exception:')
            print(e.args)


def select_roi_in_table(attr, old, new):
    print(f'Selected ROI indices {new} in image plot.')
    current_table_index = roi_table_source.selected.indices
    if current_table_index != new:
        print('Updating selected table index...')
        try:
            roi_table_source.selected.indices = new
        except Exception as e:
            print(
                'Failed to update ROI table source with the following exception:'
            )
            print(e.args)


roi_table_source = ColumnDataSource()

# roi_table_source.selected.on_change('indices', select_roi_in_plot)
# roi_source.selected.on_change('indices', select_roi_in_table)


def create_roi_table():
    int_formatter = NumberFormatter(format='0')
    columns = [
        TableColumn(
            field='x',
            title='x',
            formatter=int_formatter,
        ),
        TableColumn(
            field='y',
            title='y',
            formatter=int_formatter,
        ),
        TableColumn(
            field='width',
            title='Width',
            formatter=int_formatter,
        ),
        TableColumn(
            field='height',
            title='Height',
            formatter=int_formatter,
        ),
    ]
    roi_table_source.data = {
        key: [int(value) for value in values]
        for key, values in roi_source.data.items()
    }
    return DataTable(
        source=roi_source,
        columns=columns,
        width=400,
        height=150,
        editable=True,
    )


roi_table = create_roi_table()

roi_source.on_change('data', create_roi)

data_table_title = Div(text='<h4>Images</h4>')
roi_table_title = Div(text='<h4>ROIs</h4>')

main_layout = row(
    column(
        data_table_title,
        data_table,
        roi_table_title,
        roi_table,
    ),
    plot,
    name='main_layout',
)
curdoc().add_root(main_layout)
