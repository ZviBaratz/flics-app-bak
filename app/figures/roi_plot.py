from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure, figure


def create_roi_figure(data_source: ColumnDataSource, ) -> Figure:
    plot = figure(
        plot_width=400,
        plot_height=400,
        x_range=[0, 400],
        y_range=[0, 400],
        title='Selected ROI',
        name='roi_figure',
    )

    plot.image(
        image='image',
        x=0,
        y=0,
        dw='dw',
        dh='dh',
        source=data_source,
        palette='Spectral11',
        name='roi_plot',
    )

    return plot