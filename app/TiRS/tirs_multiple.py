import multiprocessing as mp
import collections
from skimage.io import imread
from TiRS.radon import *


class tirs_multiple:

    """ Runs Threshold in radon space anslysis on multiple frames

        The object Roi must be initialized at main:
        if __name__ == '__main__':
        In order for run_tirs function to work """

    def __init__ (self,
                  roi : np.ndarray,     # Dims : (T,X,Y)
                  threshold_const : float = 0.35,
                  heaviside_const : float = 0.02,
                  pixel_to_micron: float = 1,
                  show: bool = True,
                  autorun: bool = True):

        self.roi = roi
        self.n_frames = self.roi.shape[0]
        self.threshold_const = threshold_const
        self.heaviside_const = heaviside_const
        self.pixel_to_micron = pixel_to_micron
        self.show = show
        self.diameters = mp.Manager().dict()

        if autorun:
            self.run_tirs()


    def run_tirs_on_frame(self, frame : np.ndarray, frame_index : int, show=True):

        frame = frame - np.mean(frame)
        sinogram = radon_transform(frame, show)
        normalized_sinogram = normalize_sinogram(sinogram, show)
        thresholded_sinogram = threshold_sinogram(normalized_sinogram, self.threshold_const, show)
        reconstrucred_img = iradon_transform(frame, thresholded_sinogram, show)
        frame = heaviside_thresholding(reconstrucred_img, self.heaviside_const, show)

        diameter_in_pixel, output_image = connectedness(frame, show)

        if show:
            output_image.show()

        diameter = diameter_in_pixel/0.4 * self.pixel_to_micron # 0.4 for rescaling the original image in radon.radon_transform
        self.diameters[frame_index] = diameter


    # ToDo: Disable the dependency of starmap to be called from main
    def run_tirs(self):
        frames = [(data, frame_index) for frame_index, data in enumerate(self.roi)]

        with mp.Pool() as p:
            p.starmap(self.run_tirs_on_frame, frames)


        try:
            self.diameters = collections.OrderedDict(sorted(self.diameters.items()))
            self.diameter = sum(self.diameters.values()) / len(self.diameters)
            # print(self.diameter)
            # print(self.diameters)
        except ZeroDivisionError:
            raise ZeroDivisionError(''
                                    'The analysis on this ROI did not managed to extract diameter ')