from functools import partial
from random import random
from threading import Thread
import time
import pandas as pd

from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure, show

from tornado import gen

new_data = {
            'x' : [0, 1.5, 2.5, 3.5, 4.5],
            'y' : [0, 10, 20, 30, 40],
            #'z': [0, 1*random(), 2*random(), 3*random(), 4*random()],
        }
df = pd.DataFrame(new_data)
for c in df.keys():
    d = df[c]

"""
data = {'x_values': [1, 2, 3, 4, 5],
        'y_values': [6, 7, 2, 3, 6]}
# this must only be modified from a Bokeh session callback
source = ColumnDataSource(data=dict(x=[0], y=[0]))
source.add([11, 22, 33, 44, 55], 'y_1_values')

p = figure()
p.circle(x='x_values', y='y_values', source=source)
"""
new_data = {
            'x' : [0, 2, 2.1, 3, 4],
            'y' : [0, 1, 2, 3, 4],
            #'z': [0, 1*random(), 2*random(), 3*random(), 4*random()],
        }
df = pd.DataFrame(new_data)

doc = curdoc()
source = ColumnDataSource(data=dict(x=[0], y=[0]))

@gen.coroutine
def update(x,y):
    for c in df.keys():
        d = df[c]
        print('d is:', d)
    source.stream(x, y)

def blocking_task():
    while True:
        # do some blocking computation
        time.sleep(0.1)
        #new_data = {
         #   'x' : [0, 1.5, 2.5, 3.5, 4.5],
          #  'y' : [0, 10, 20, 30, 40],
        #}
        #df = pd.DataFrame(new_data)
        x = 3
        y = 3

        doc.add_next_tick_callback(partial(update, x, y))

p = figure(x_range = [0,10], y_range = [0,50])
l = p.circle(x='x', y='y', source=source)
doc.add_root(p)
show(p)

thread = Thread(target=blocking_task)
thread.start()
