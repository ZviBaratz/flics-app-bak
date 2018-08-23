from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn
from data_access.data_access_object import DataAccessObject
from data_access.image_file import ImageFile


def create_data_table(source: ColumnDataSource) -> DataTable:
    columns = [
        TableColumn(field='images', title='Image'),
        TableColumn(field='shapes', title='Shape'),
    ]
    return DataTable(
        source=source,
        columns=columns,
        width=400,
        height=300,
    )


def get_selection_image(
        source: ColumnDataSource,
        dao: DataAccessObject,
) -> ImageFile:
    try:
        return dao.get_image(source.selected.indices[0])
    except IndexError:
        return None


def get_selection_rois(source: ColumnDataSource, dao: DataAccessObject):
    image = get_selection_image(source, dao)
    return image.get_roi_dict()
