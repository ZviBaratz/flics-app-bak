# import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import scipy
import scipy.fftpack
from PIL.TiffImagePlugin import TiffImageFile
from app.analysis.global_fit import *

# from .tests.old_results_parser import parse_old_results


class Analysis:
    results = None

    def __init__(self,
                 image: np.ndarray = None,
                 image_path: str = None,
                 threshold: float = None,
                 min_distance: int = 0,
                 max_distance: int = 320,
                 distance_step: int = 20,
                 autorun: bool = True):
        """
        Runs a flow image correlation spectroscopy (FLICS) analysis for a given
        2D numpy array or .tif image file.
        
        :param image: Numpy array representing an image or ROI to analyse.
        :type image: np.ndarray
        :param image_path: .tif image file path.
        :type image_path: str
        :param threshold: Threshold to be applied to the image, defaults to None.
        :param threshold: float, optional
        :param max_distance: Maximum column-pair distance to be calculated,
        defaults to 320.
        :param max_distance: int, optional
        :param distance_step: Step value for column-pair distances iteration,
        defaults to 20.
        :param distance_step: int, optional
        :param autorun: Run analysis on instantiation, defaults to True.
        :param autorun: bool, optional
        """

        # Set class parameters
        if not isinstance(image, np.ndarray) and not image_path:
            raise ValueError(
                'An instance of the FLICS analysis must be initialized with an image\'s data or path.'
            )
        elif not isinstance(image, np.ndarray) and image_path:
            self.image = self.load_image(image_path)
        else:
            self.image = image
        self.threshold = threshold
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.distance_step = distance_step

        # Create a range of the distances to be calculated
        self.distances = range(self.min_distance, self.max_distance,
                               self.distance_step)

        # Load image data and calculate column means and element-wise deviation
        self.image_data = self.get_image_data()
        self.column_means = self.image_data.mean(axis=0)
        self.deviation_from_column_mean = self.image_data - self.column_means

        if autorun:
            self.run()

    def load_image(self, path: str):
        """
        Returns the loaded image data.
        
        :param path: .tif image file path.
        :type path: str
        :raises FileNotFoundError: Failed to read image from file.
        :return: Loaded image data.
        :rtype: PIL.TiffImagePlugin.TiffImageFile
        """
        try:
            return PIL.Image.open(path)
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(f'Failed to load image file from {path}')

    def get_image_data(self) -> np.ndarray:
        """
        Returns image data as numpy array and applies the analysis threshold
        if defined.
        
        :return: Thresholded image data.
        :rtype: np.ndarray
        """
        data = np.array(self.image, dtype=float)
        if isinstance(self.threshold, float):
            data[data < self.threshold] = 0
        return data

    def calc_fft(self, i_column: int) -> np.ndarray:
        """
        Returns the discrete Fourier transform of a column's element-wise
        deviation from the (column) mean.
        
        :param i_column: Index of the chosen image column.
        :type i_column: int
        :return: A complex array with the tranform results.
        :rtype: np.ndarray
        """
        return scipy.fftpack.fft(self.deviation_from_column_mean[:, i_column])

    def calc_cross_corr(self, i_column: int, distance: int) -> np.ndarray:
        """
        Returns the cross-correlation function results for a given column and
        distance.
        
        :param i_column: Index of the chosen image column.
        :type i_column: int
        :param distance: Distance between the columns to cross-correlate.
        :type distance: int
        :return: Cross-correlation between the two columns.
        :rtype: np.ndarray
        """
        # TODO: This function should be refactored further with a better
        # understanding of the underlying algorithm
        column_transform = self.calc_fft(i_column)
        if type(self.image) is TiffImageFile:
            width = self.image.width
        else:
            width = self.image.shape[1]
        if not i_column + distance < width:
            return None
        distant_column_transform = self.calc_fft(i_column + distance)
        distant_column_transform = np.ma.conjugate(distant_column_transform)

        inverse = scipy.fftpack.ifft(distant_column_transform * column_transform)
        if type(self.image) is TiffImageFile:
            height = self.image.height
        else:
            height = self.image.shape[0]
        divider = (self.column_means[i_column] * self.column_means[i_column + distance] * height)
        crosscorr = np.real(inverse / divider)
        return crosscorr

    def calc_cross_corr_for_distance(self, distance: int) -> np.ndarray:
        """
        Returns all cross-correlations between two image columns with the given
        distance.

        :param distance: Distance between the columns to cross-correlate.
        :type distance: int
        :return: A stacked array of all column pairs cross-correlation results.
        :rtype: np.ndarray
        """
        if type(self.image) is TiffImageFile:
            n_column_pairs = self.image.width
        else:
            n_column_pairs = self.image.shape[1] - distance
        if n_column_pairs > 1:
            return np.stack([
                result for result in [
                    self.calc_cross_corr(x, distance)
                    for x in range(n_column_pairs)
                ] if result is not None
            ])

    def run(self) -> None:
        """
        Returns the analysis results for all distances.

        :return: A dictionary of mean column-pair cross-correlations by
        distance.
        :rtype: dict
        """
        self.results = {}
        for distance in self.distances:
            cross_correlations = self.calc_cross_corr_for_distance(distance)
            if isinstance(cross_correlations, np.ndarray):
                self.results[distance] = cross_correlations.mean(axis=0)
            else:
                self.results[distance] = None
        return self.results

    # def show_results(self, distances: list, compare_to_old: bool = True):
    #     n_rows = len(distances)
    #     x = range(self.image.width)
    #     if compare_to_old:
    #         old_results = parse_old_results()

    #     plt.subplots(n_rows, 1)

    #     for i_row, distance in enumerate(distances):
    #         ax = plt.subplot(n_rows, 1, i_row + 1)
    #         plt.title(f'Distance = {distance}')
    #         distance_results = self.results[distance]
    #         ax.plot(x, distance_results, label='Results')
    #         if compare_to_old:
    #             right_left = old_results[distance]['rightleft']
    #             ax.plot(
    #                 range(50),
    #                 right_left,
    #                 color='r',
    #             )
    #             left_right = old_results[distance]['leftright'][1:]
    #             ax.plot(
    #                 range(self.image.width - 1, self.image.width - 51, -1),
    #                 left_right,
    #                 color='r',
    #                 label='Old results',
    #             ),
    #             ax.legend()
    #     plt.show(block=False)


if __name__ == '__main__':
    analysis = Analysis(None, r'd:\git\flics\flics_data\Angoli_vena_conventional_Series012t5_6gradi.tif', None, 0, 320, 20, True)
    global_fit = GlobalFit(analysis.results)
    global_fit_res = global_fit.run()
    print(global_fit_res)
"""
#check_results:
    match_indxs = []
    no_match_indxs = []
    for index in analysis.results:
        filename = r'D:\git\flics\flics_data\correlation%sleftright.txt' %(index)
        f = open(filename, 'r')
        for i in range(51):
            val_file = np.array(float(f.readline().split(',')[1]))
            val_anals = np.array(analysis.results[index][i])
            if np.allclose(val_file, val_anals):
                if index not in match_indxs:
                    match_indxs.append(index)
                print('match, index = ', index, 'i=',i)
            else:
                for j in range(51,1025):
                    val_anals = np.array(analysis.results[index][i])
                    if np.allclose(val_file, val_anals):
                        print('originally match not found, match found between indexes:', i, j)
                if index not in no_match_indxs:
                    no_match_indxs.append(index)
                print('no match for index:', index, 'i =', i, 'val_file=', val_file,'val_anals=', val_anals, 'val_file/val_anals=', val_file/val_anals)
    print('matching indexes:', match_indxs, 'not matching:', no_match_indxs)
"""
