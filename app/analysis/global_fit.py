import numpy as np
import symfit as sf


class GlobalFit(object):
    def __init__(self, data: dict, angle,
                 pixel_to_micron_x=0.03651,
                 beam_waist_xy=0.2,
                 beam_waist_z=2e-6,
                 rbc_radius=6.2666,
                 tau_line=0.001,
            ):
        self.data = data
        self.angle = angle
        self.pixel_to_micron_x = pixel_to_micron_x
        self.beam_waist_xy = beam_waist_xy
        self.beam_waist_z = beam_waist_z
        self.rbc_radius = rbc_radius
        self.tau_line = tau_line
        self.s = 1  # 1 = one- photon . 2 = two-photon. in the article S = 1

        #self.results = self.run()

    def create_distance_strings(self, prefix: str):
        return ', '.join(f'{prefix}_{distance}' for distance in self.data.keys())

    def run(self):
        print('global_fit starting')
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
            y: y0 + b * sf.exp(-(dst * self.pixel_to_micron_x - v * (sf.cos(self.angle)) * x)**2 /
            (self.beam_waist_xy**2 + 0.5 * self.rbc_radius**2 * self.s + 4 * d * x * self.s)) *
            sf.exp(-(x**2) * (self.pixel_to_micron_x / self.tau_line - v * sf.sin(self.angle))**2 /
            (self.beam_waist_xy**2 + 0.5 * self.s * self.rbc_radius**2 + 4 * self.s * d * x))
            for y, y0, b, dst in zip(y_var, y0_p, b_p, distances)
        })
        # dependent variables dict
        data = {y.name: self.data[dst] for y, dst in zip(y_var, distances)}
        # independent variable x
        n_data_points = len(list(self.data.values())[0])
        max_time = n_data_points * self.tau_line
        x_data = np.linspace(0, max_time, n_data_points)
        # fit
        fit = sf.Fit(model, x=x_data, **data)
        self.results = fit.execute()
        print('global_fit done')
        return self.results.params['v_']

