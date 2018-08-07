import numpy as np
import os
import PIL.Image


class DataObject:
    _path = None

    def __init__(self, path: str):
        self.path = path
        self.name = self.get_name()
        self.image = self.get_image()
        self.data = self.get_data()

    def get_image(self):
        """
        Returns the loaded image data.

        :param path: .tif image file path.
        :type path: str
        :raises FileNotFoundError: Failed to read image from file.
        :return: Loaded image data.
        :rtype: PIL.TiffImagePlugin.TiffImageFile
        """
        try:
            return PIL.Image.open(self.path)
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(
                f'Failed to load image file from {self.path}')

    def get_data(self) -> np.ndarray:
        """
        Returns image data as numpy array.

        :return: Image data.
        :rtype: np.ndarray
        """
        return np.array(self.image, dtype=float)

    def get_name(self) -> str:
        return os.path.basename(self.path).split('.')[0]

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value: str):
        if not os.path.isfile(value) and value.endswith('.tif'):
            raise ValueError('Path must be set to a valid TIFF file path.')
        self._path = value
