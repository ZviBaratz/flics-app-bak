import symfit as sf
import glob
import numpy as np
import re
import pandas as pd
import time


class GlobalFit2:
    def __init__(self, data='', test=False, tst_pth=r'flics_data'):
        # if test is False and data is not isinstance(dict):
        #     raise TypeError('data input is not of type dict')
        # elif test is True:
        #     self.data = self.txt_res_to_dict(tst_pth)
        # else:
        self.data = data

    def do_glob_fit(self):
        """this method performs global fit on the CCPS"""
        # create parameters for symfit
        dist = self.data.keys()
        v = sf.parameters('v_', value=500, min=0, max=1000)[0]
        d = sf.parameters('D_', value=50, min=0, max=100)[0]
        y0_p = sf.parameters(
            ', '.join('y0_{}'.format(key) for key in self.data.keys()),
            min=0,
            max=1)
        b_p = sf.parameters(
            ', '.join('b_{}'.format(key) for key in self.data.keys()),
            value=50,
            min=0,
            max=100)
        # create variables for symfit
        x = sf.variables('x')[0]
        y_var = sf.variables(', '.join(
            'y_{}'.format(key) for key in self.data.keys()))
        # get fixed & shared params
        dx, a, w2, a2, tau, s, wz2 = self.get_params()
        # create model
        model = sf.Model({
            y: y0 + b * sf.exp(-(dst * dx - v * (sf.cos(a)) * x)**2 /
                               (w2 + 0.5 * a2 + 4 * d * x)) *
            sf.exp(-(x**2) * (v * sf.sin(a) - dx / tau)**2 /
                   (w2 + 0.5 * a2 + a * d * x)) / (4 * d * x + w2 + 0.5 * a2)
            for y, y0, b, dst in zip(y_var, y0_p, b_p, dist)
        })
        # dependent variables dict
        data = {y.name: self.data[dst] for y, dst in zip(y_var, dist)}
        # independent variable x
        max_time = len(self.data[20]) * tau
        x_data = np.linspace(0, max_time, len(self.data[20]))
        # fit
        fit = sf.Fit(model, x=x_data, **data)
        res = fit.execute()
        return res

    def txt_res_to_dict(self, pth):
        data_dict = dict()
        files_name_list = glob.glob(pth + r'\*.txt')
        # column correlation number to extract
        p = re.compile('\d+')
        # extract keys
        keys = np.array([p.findall(name) for name in files_name_list]).astype(
            np.int)
        # string format to extract
        p = re.compile('\d+\.\d+')
        # read all lines to a list
        data = np.array(([
            p.findall(open(name, 'r').read()) for name in files_name_list
        ])).astype(np.float16)
        # convert to a relevant dictionary
        for idx, key in enumerate(keys):
            data_dict[key[0]] = data[idx]
        return data_dict

    def get_params(self):
        # read params from csv file - for easy editing
        # path default set to working folder
        p_df = pd.read_csv('parameters.csv')
        p_df = p_df.rename(index=p_df.iloc[:, 0])
        # return the fixed & shared params as tuple
        return tuple(p_df.loc['dx':'wz2', 'Value'].astype(np.float32))


if __name__ == '__main__':
    s = time.time()
    gf = GlobalFit2(test=True)
    print(time.time() - s)
    res = gf.do_glob_fit()