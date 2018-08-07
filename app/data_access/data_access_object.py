import glob
import os
import pandas as pd

from .data_object import DataObject

DEFAULT_PATH = os.path.abspath('./analysis')


class DataAccessObject:
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = DEFAULT_PATH
        self.images_dir = os.path.join(self.path, 'data')
        self.results_dir = os.path.join(self.path, 'results')
        self.images = self.get_images()

    def get_image_paths(self) -> list:
        return glob.glob(os.path.join(self.images_dir, '*.tif'))

    def get_images(self) -> list:
        return [DataObject(path) for path in self.get_image_paths()]

    def to_df(self) -> list:
        return pd.DataFrame([{
            'Name':
            os.path.basename(image.path).split('.')[0],
            'Shape':
            str(image.data.shape),
        } for image in self.images])
