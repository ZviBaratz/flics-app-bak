import glob
import os

from .image_file import ImageFile

DEFAULT_PATH = os.path.abspath('./analysis')


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
