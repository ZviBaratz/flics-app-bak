import json
import numpy as np

from bokeh.core.json_encoder import serialize_json
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    NumberFormatter,
)
from bokeh.models.widgets import DataTable, TableColumn, Div
from data_access.data_access_object import DataAccessObject
from data_access.image_file import ImageFile
from figures.image_plot import create_image_figure, update_image_figure
from widgets.datatables import (
    create_data_table,
    get_selection_image,
    get_selection_rois,
)

dao = DataAccessObject()
data = dict(
    images=[image.name for image in dao.images],
    shapes=[image.data.shape for image in dao.images],
)
data_source = ColumnDataSource(data=data)
data_table = create_data_table(data_source)

# selected_image = get_selection_image(data_source, dao)
# data = selected_image.data
# image_source = ColumnDataSource(
#     data=dict(
#         image=[data],
#         x=[0],
#         y=[0],
#         dw=[data.shape[1]],
#         dh=[data.shape[0]],
#     ))
image_source = ColumnDataSource(
    data=dict(
        image=[],
        x=[],
        y=[],
        dw=[],
        dh=[],
    ))

roi_source = ColumnDataSource(data={})

# roi_source.data = dao.get_roi_source(selected_image)


def create_image_data_source(data: np.ndarray) -> dict:
    return dict(image=[data], dw=[data.shape[1]], dh=[data.shape[0]])


def update_image_source(data: np.ndarray) -> None:
    image_source.data = create_image_data_source(data)


def update_roi_source():
    roi_source.data = get_selection_rois(data_source, dao)


def update_image_plot(attr, old, new):
    image = get_selection_image(data_source, dao)
    data = image.data
    update_image_source(data)
    update_image_figure(plot, data)
    update_roi_source()


data_source.selected.indices = [0]
image = get_selection_image(data_source, dao)
update_image_source(image.data)
update_roi_source()
plot = create_image_figure(image_source, roi_source)
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
    image = get_selection_image(data_source, dao)
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
    # roi_table_source.data = {
    #     key: [int(value) for value in values]
    #     for key, values in roi_source.data.items()
    # }
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


def handle_vector(attr, old, new):
    print('\nCreated new vector!')
    print(f'x: {new["xs"][-1]}, y: {new["ys"][-1]}')


lines = curdoc().get_model_by_name('vectors')
lines.data_source.on_change('data', handle_vector)
