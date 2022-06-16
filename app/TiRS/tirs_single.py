from skimage.io import imread
from TiRS.radon import *


class tirs_single:

    """ Runs Threshold in radon space anslysis on a single frame """

    def __init__(self,
                 image : np.ndarray = None,
                 image_path : str = None,
                 threshold_const : float = 0.35,
                 heaviside_const : float = 0.02,
                 pixel_to_micron: float = 1,
                 autorun : bool = True,
                 show : bool = True,
                ):

        if not isinstance(image, np.ndarray) and not image_path:
            raise ValueError(
                'An instance of the TiRS analysis must be initialized with an image\'s data or path.'
            )
        elif not isinstance(image, np.ndarray) and image_path:
            self.image = self.load_image(image_path)
        else:
            self.image = image

        self.image_path = image_path
        self.threshold_const = threshold_const
        self.heaviside_const = heaviside_const
        self.pixel_to_micron = pixel_to_micron
        self.show = show

        if autorun:
            self.run(show)

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
            return imread(path, as_gray=True)
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(f'Failed to load image file from {path}')


    def run(self, show):
        self.image = self.image - np.mean(self.image)
        sinogram = radon_transform(self.image, show)
        normalized_sinogram = normalize_sinogram(sinogram, show)
        thresholded_sinogram = threshold_sinogram(normalized_sinogram, self.threshold_const, show)
        reconstrucred_img = iradon_transform(self.image, thresholded_sinogram, show)
        self.image = heaviside_thresholding(reconstrucred_img, self.heaviside_const, show)
        diameter_in_pixel, output_image = connectedness(self.image, show)

        print(diameter_in_pixel)

        self.diameter = diameter_in_pixel * self.pixel_to_micron

        if show:
            output_image.show()

