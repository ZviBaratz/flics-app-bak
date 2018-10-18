from numpy import ndarray
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    BoxEditTool,
    PolyDrawTool,
    PolyEditTool,
)
from bokeh.plotting import Figure, figure


def create_image_figure(
        image_source: ColumnDataSource,
        roi_source: ColumnDataSource,
        vector_source: ColumnDataSource,
) -> Figure:

    try:
        image = image_source.data['image'][0]
        width = image.shape[1]
        height = image.shape[0]
    except IndexError:
        width = 800
        height = 800

    plot = figure(
        plot_width=min(width, 800),
        plot_height=min(height, 800),
        x_range=[0, width],
        y_range=[0, height],
        title='Selected Image',
        name='image_figure',
    )

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

    hover = HoverTool(tooltips=[('x', '$x'), ('y', '$y'), ('value', '@image')])
    r1 = plot.rect(
        x='x',
        y='y',
        width='width',
        height='height',
        source=roi_source,
        fill_alpha=0.5,
        fill_color='#DAF7A6',
        name='rois',
    )

    lines = plot.multi_line(
        xs='xs',
        ys='ys',
        source=vector_source,
        line_color='red',
        line_width=2,
        name='vectors',
    )

    circles = plot.circle(
        x=[],
        y=[],
        size=10,
        color='yellow',
    )

    plot.tools = [
        hover,
        BoxEditTool(renderers=[r1]),
        PolyDrawTool(renderers=[lines]),
        PolyEditTool(renderers=[lines], vertex_renderer=circles),
    ]

    return plot


def update_image_figure(fig: Figure, data: ndarray):
    fig.x_range.end = data.shape[1]
    fig.y_range.end = data.shape[0]