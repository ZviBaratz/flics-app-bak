import json
import numpy as np
import os
import PIL.Image
import tifffile


class ImageFile:
    _path = None

    def __init__(self, path: str):
        self.path = path
        self.name = self.get_name()
        # self.image = self.get_image()
        self.data = self.read_image()
        self.metadata = self.read_metadata()
        self.rois_file_name = f'{self.name}.json'
        self.rois_file_path = None

    def read_image(self):
        with tifffile.TiffFile(self.path, movie=True) as f:
            data = f.asarray(slice(1, None, 2))  # read the second channel only
            # metadata = f.scanimage_metadata
            return data

    def read_metadata(self):
        with tifffile.TiffFile(self.path, movie=True) as f:
            return f.scanimage_metadata

    # def get_image(self):
    #     """
    #     Returns the loaded image data.

    #     :param path: .tif image file path.
    #     :type path: str
    #     :raises FileNotFoundError: Failed to read image from file.
    #     :return: Loaded image data.
    #     :rtype: PIL.TiffImagePlugin.TiffImageFile
    #     """
    #     try:
    #         return PIL.Image.open(self.path)
    #     except (FileNotFoundError, OSError) as e:
    #         raise FileNotFoundError(
    #             f'Failed to load image file from {self.path}')

    # def get_data(self) -> np.ndarray:
    #     """
    #     Returns image data as numpy array.

    #     :return: Image data.
    #     :rtype: np.ndarray
    #     """
    #     return np.array(self.image, dtype=float)

    # def get_metadata(self) -> dict:
    #     with tifffile.TiffFile(self.path) as tif:
    #         metadata = tif.imagej_metadata.copy()
    #     return self.fix_raw_metadata(metadata)

    def get_name(self) -> str:
        return os.path.basename(self.path).split('.')[0]

    def get_roi_dict(self) -> dict:
        if self.rois_file_path and os.path.isfile(self.rois_file_path):
            with open(self.rois_file_path) as serialized_rois:
                rois = json.load(serialized_rois)
            return rois
        return dict(x=[], y=[], width=[], height=[])

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value: str):
        if not os.path.isfile(value) and value.endswith('.tif'):
            raise ValueError('Path must be set to a valid TIFF file path.')
        self._path = value
