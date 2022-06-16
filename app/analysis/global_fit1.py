import numpy as np
import symfit as sf
from app.analysis.global_fit import *


class GlobalFit:
    def __init__(self, data: dict):
        self.data = data

    def create_distance_strings(self, prefix: str):
        return ', '.join(f'{prefix}_{distance}' for distance in self.data.keys())

    def run(
            self,
            angle=0.02472,
            pixel_to_micron_x=9.81e-8,
            beam_waist_xy=0.5e-6,
            beam_waist_z=2e-6,
            rbc_radius=4e-6,
            s=1,
            tau=0.001,
    ):
        # create parameters for symfit
        distances = self.data.keys()
        v = sf.parameters('v_', value=500, min=0, max=1000)[0]
        d = sf.parameters('D_', value=50, min=0, max=100)[0]
        y0_p = sf.parameters(
            self.create_distance_strings('y0'),
            min=0,
            max=1,
        )
        b_p = sf.parameters(
            self.create_distance_strings('b'),
            value=50,
            min=0,
            max=100,
        )
        # create variables for symfit
        x = sf.variables('x')[0]
        y_var = sf.variables(self.create_distance_strings('y'))

        # create model
        # pixel_to_micron_x
        model = sf.Model({
            y:
            y0 + b * sf.exp(-(dst * pixel_to_micron_x - v * (sf.cos(angle)) * x)**2 / (beam_waist_xy + 0.5 * rbc_radius**2 + 4 * d * x)) * sf.exp(-(x**2) * (v * sf.sin(angle) - pixel_to_micron_x / tau)**2 / (beam_waist_xy + 0.5 * rbc_radius**2 + angle * d * x)) / (4 * d * x + beam_waist_xy + 0.5 * rbc_radius**2)
            for y, y0, b, dst in zip(y_var, y0_p, b_p, distances)
        })
        # dependent variables dict
        data = {y.name: self.data[dst] for y, dst in zip(y_var, distances)}
        # independent variable x
        n_data_points = len(list(self.data.values())[0])
        max_time = n_data_points * tau
        x_data = np.linspace(0, max_time, n_data_points)
        # fit
        fit = sf.Fit(model, x=x_data, **data)
        res = fit.execute()
        return res
