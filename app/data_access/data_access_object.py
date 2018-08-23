import glob
# import json
import os

from .image_file import ImageFile

DEFAULT_PATH = os.path.abspath('./analysis')

# EMPTY_ROI_SOURCE = dict(x=[], y=[], width=[], height=[])


class DataAccessObject:
    rois_file_name = '{name}.json'

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = DEFAULT_PATH
        self.images_dir = os.path.join(self.path, 'data')
        self.rois_dir = os.path.join(self.path, 'rois')
        self.results_dir = os.path.join(self.path, 'results')
        self.images = self.create_image_instances()
        self.update_image_instances()

    def get_image_paths(self) -> list:
        return glob.glob(os.path.join(self.images_dir, '*.tif'))

    def create_image_instances(self) -> list:
        return [ImageFile(path) for path in self.get_image_paths()]

    def get_roi_file_path(self, image: ImageFile) -> str:
        file_name = self.rois_file_name.format(name=image.name)
        return os.path.join(self.rois_dir, file_name)

    def update_image_instances(self) -> None:
        for image in self.images:
            image.rois_file_path = self.get_roi_file_path(image)

    def get_image(self, index: int):
        return self.images[index]

    def get_roi_source(self, image: ImageFile):
        file_path = image.rois_file_path
        if os.path.isfile(file_path):
            with open(file_path) as serialized_rois:
                rois = json.load(serialized_rois)
            return rois
        return EMPTY_ROI_SOURCE
