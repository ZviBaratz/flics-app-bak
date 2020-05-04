import numpy as np
import symfit as sf


class GlobalFit:
    def run(self):
        # create parameters for symfit
        a_0, b_0, a_1, b_1 = sf.parameters('a_0, b_0, a_1, b_1')
        x_0, y_0, x_1, y_1 = sf.variables('x_0, y_0, x_1, y_1')
        # create model
        model = sf.Model({
            y_0: a_0 * x_0 + b_0,
            y_1: a_1 * x_1 + b_1,
        })
        x_data0 = np.linspace(0, 60, 30)
        x_data1 = np.linspace(0, 60, 30)
        y_data0 = 10 * x_data0 + 20
        y_data1 = 30 * x_data1 + 3

        # fit
        fit = sf.Fit(model, x_0=x_data0, y_0=y_data0, x_1=x_data1, y_1=y_data1)
        res = fit.execute()
        print(res)


global_fit = GlobalFit()
global_fit_res = global_fit.run()
print(global_fit_res)
