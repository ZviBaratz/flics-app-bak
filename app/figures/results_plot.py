from bokeh.models import ColumnDataSource
from bokeh.plotting import Figure, figure


def create_results_plot(data_source: ColumnDataSource) -> Figure:
    plot = figure(
        plot_width=650,
        plot_height=300,
        x_range=[0, 100],
        y_range=[0, 1000],
        title='Results',
        name='results_figure',
    )

    plot.xaxis.axis_label = 'Frame'
    plot.yaxis.axis_label = 'v'

    plot.line(
        x='x',
        y='y',
        source=data_source,
        name='results_plot',
    )

    return plot


def create_cross_correlation_plot(data_source: ColumnDataSource) -> Figure:
    plot = figure(
        plot_width=650,
        plot_height=300,
        x_range=[0, 150],
        y_range=[0, 1000],
        title='Cross Correlations',
        name='cross_correlation_figure',
    )

    plot.xaxis.axis_label = 'ROI Column Index'
    plot.yaxis.axis_label = 'Cross-Correlation'

    return plot
