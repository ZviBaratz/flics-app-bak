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
        self.image = self.get_image()
        self.data = self.get_data()
        self.rois_file_name = f'{self.name}.json'
        self.rois_file_path = None

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

    def fix_raw_metadata(self, raw_metadata: dict) -> dict:
        fixed_metadata = dict()
        for key in raw_metadata:
            if key == 'ImageJ':
                fixed_metadata['image_j'] = raw_metadata[key]
            elif key == 'Info':
                for att in raw_metadata[key].split('\n'):
                    try:
                        att_name, att_value = att.split('=')
                    except ValueError:
                        pass
                    try:
                        att_value = float(att_value)
                    except ValueError:
                        att_value = att_value.strip()
                    att_name = att_name.strip()
                    if '|' in att_name:
                        fields = att_name.split('|')
                        if fields[0] not in fixed_metadata:
                            fixed_metadata[fields[0]] = dict()
                        if len(fields) is 2:
                            fixed_metadata[fields[0]][fields[1]] = att_value
                        elif len(fields) is 3:
                            if fields[1] not in fixed_metadata[fields[0]]:
                                print(fields)
                                fixed_metadata[fields[0]][fields[1]] = dict()
                            fixed_metadata[fields[0]][fields[1]][fields[
                                2]] = att_value
                    else:
                        fixed_metadata[att_name] = att_value
            else:
                fixed_metadata[key] = raw_metadata[key]
        return fixed_metadata

    def get_metadata(self) -> dict:
        with tifffile.TiffFile(self.path) as tif:
            metadata = tif.imagej_metadata.copy()
        return self.fix_raw_metadata(metadata)

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
