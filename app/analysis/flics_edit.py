import PIL.Image
import scipy
import scipy.fftpack
import numpy as np
from app.analysis.global_fit import *
from file_handling import *
import time


class Analysis(object):
    def __init__(self,
                 data_channel: int,
                 num_of_data_channels : int,
                 image: np.ndarray = None,
                 image_path: str = None,
                 frame_number : int = None,
                 roi_coordinates : np.array= None,
                 threshold: float = None,
                 min_distance: int = 0,
                 max_distance: int = 320,
                 columnstep: int = 20,
                 columnlimit: int = 50,
                 autorun: bool = False):
        """
        Runs a flow image correlation spectroscopy (FLICS) analysis for a given
        2D numpy array or .tif image file.

        :param image: Numpy array representing an image or ROI to analyse.
        :type image: np.ndarray
        :param image_path: .tif image file path.
        :type image_path: str
        :param threshold: The threshold is to highlight the fluorescent diagonal lines with respect to black stripes (the intensity value of a pixel is put to zero every time it is below the chosen threshold)

        :param threshold: float, optional
        :param max_distance: Maximum column-pair distance to be calculated,
        defaults to 320. should be equal to (or less than) half of the y dimension (in pixels) of the image
        (due to the fact that we compute the correlation functions with FFTs)
        :param max_distance: int, optional
        :param columnlimit: Cross-correlation functions are computed for all the possibile distances from 0 to columnlimit, with step equal to columnstep
        :param columnlimit: int, defaults to 50
        :param columnstep: Step value for column-pair distances iteration,
        defaults to 20.
        :param columnstep: int, optional
        :param autorun: Run analysis on instantiation, defaults to True.
        :param autorun: bool, optional
        """
        self.frame_number = frame_number
        self.roi_coordinates = roi_coordinates
        self.threshold = threshold
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.columnstep = columnstep
        self.columnlimit = columnlimit

        # Set class parameters
        if not isinstance(image, np.ndarray) and not image_path:
            raise ValueError(
                'An instance of the FLICS analysis must be initialized with an image\'s data or path.'
            )
        elif not isinstance(image, np.ndarray) and image_path:
            self.image = crop_img(roi_coordinates, frame_number, image_path, data_channel, num_of_data_channels)
        else:
            self.image = image

        # Create a range of the distances to be calculated
        self.distances = range(self.min_distance, self.max_distance,
                               self.columnstep)
        if autorun:
            self.results = self.main_ccf()
            #self.check_results()

    def correlation(self, t1, m1, t2, m2, y):
        fft1 = scipy.fftpack.fft([(t1[i])-m1 for i in range(y)])
        fft2 = scipy.fftpack.fft([(t2[i])-m2 for i in range(y)])
        fft1 = np.ma.conjugate(fft1)
        crosscorr = np.real(scipy.fftpack.ifft(fft1*fft2)/(m1*m2*y))
        return crosscorr

    def main_ccf(self):
        """
        Returns the analysis results for all distances.
        :return: A dictionary of mean column-pair cross-correlations by
        distance.
        :rtype: dict
        """
        print("main ccf called")
        x = self.image.shape[0]
        y = self.image.shape[1]

        #data thresholding (if needed)
        if self.threshold:
            self.image[self.image < self.threshold] = 0.0
        self.image = self.image.T

        #computation of the average value for each image column
        mean = np.mean(self.image, axis=1)

        #computation of the cross-correlation function
        # the distance between the columns to cross-correlate varies
        # from 0 to max_distance, with step equal to columnstep
        ccf_results = {}
        for i in self.distances:
            output = [0 for k in range(y)]
            #For each distance, ALL the pairs of columns at that distance are exploited.
            #The average cross-correlation function is provided (convenient for statistical reasons)
            for j in range(x-i):
                corr = self.correlation(self.image[j+i], mean[j+i], self.image[j], mean[j], y)

                output = [output[l]+corr[l]/(x-i) for l in range(y)]

            res =[]
            res.append(output[0])
            for k in range(self.columnlimit):
                res.append(output[y-k-1])   #only for lefttoright option
            ccf_results[i] = res
        return ccf_results

    def check_results(self):
        print(self.results)
        match_indxs = []
        no_match_indxs = []
        for index in self.results:
            filename = r'D:\git\flics\flics_data\correlation%sleftright.txt' %(index)
            f = open(filename, 'r')
            val_anals_arr = np.array(self.results[index])
            for i in range(51):
                val_file = np.array(float(f.readline().split(',')[1]))
                val_anals = val_anals_arr[i]
                if np.allclose(val_file, val_anals):
                    if index not in match_indxs:
                        match_indxs.append(index)
                    print('match, index = ', index, 'i=', i)
                else:
                    if index not in no_match_indxs:
                        no_match_indxs.append(index)
                    print('no match for index:', index, 'i =', i, 'val_file=', val_file,'val_anals=', val_anals, 'val_file/val_anals=', val_file/val_anals)
        print('matching indexes:', match_indxs, 'not matching:', no_match_indxs)


def xcorr_globalfit():
    start_time_analysis = time.time()
    analysis = Analysis(None, r'd:\git\flics\flics_data\Angoli_vena_conventional_Series012t5_6gradi.tif', None, 0, 320, 20, 50, True)
    elap_time_analysis = time.time() - start_time_analysis
    print('elapsed time analysis func =', elap_time_analysis)

    start_time_globalfit = time.time()
    global_fit = GlobalFit(analysis.results, 0.02472, 0.03651, 0.2, 2e-6, 6.2666, 0.001)
    global_fit_res = global_fit.run()
    elap_time_globalfit = time.time() - start_time_globalfit
    print('elapsed time globalfit func =', elap_time_globalfit)

    return global_fit_res

def run_xcor(db_path):
    db_row = get_row_to_procc_from_db(db_path)
