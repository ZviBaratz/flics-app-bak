import tifffile
import numpy as np
import os

IMAGE_EXT = '.tif'
BASE_DIR = r'd:\git\latest\flics\flics_data'

#frontend:

def get_file_paths(base_dir_input: str = BASE_DIR, ext: str = IMAGE_EXT) -> list:
    """
    Returns a list of files ending with 'ext' within a directory tree.

    :param ext: The target file extension, defaults to IMAGE_EXT.
    :type ext: str
    :return: A list of file paths.
    :rtype: list
    """
    image_paths = []
    for root, dirs, files in os.walk(base_dir_input):
        for f in files:
            if f.endswith(ext):
                image_paths.append(os.path.join(root, f))
    return image_paths


#backend:

def crop_img(roi_coordinates : np.array, frame : int, img_path: str) -> np.ndarray:
    roi_parsed = np.frombuffer(roi_coordinates, dtype=np.float)
    x_start, x_end, y_start, y_end = int(roi_parsed[0]), int(roi_parsed[1]), int(roi_parsed[2]), int(roi_parsed[3]) #todo: make pretty
    return get_current_image(img_path)[frame, y_start:y_end, x_start:x_end]


def get_current_image(img_path: str) -> np.ndarray:
    img = [read_image_file(img_path)][0]
    print('get_current_image(), img =  ',img)
    print('path = ', img_path, 'img.ndim=', img.ndim)
    if img.ndim == 2:
        return img[np.newaxis, :]
    elif img.ndim == 3:
        return img
    else:
        raise ValueError(f"Image dimension mismatch. Number of dimensions: {img.ndim}")


def read_image_file(img_path: str) -> np.ndarray:
    with tifffile.TiffFile(img_path, movie=True) as f:
        try:
            data = f.asarray(slice(1, None, 2))  # read the second channel only
        except ValueError:
            data = f.asarray()
        return data


def read_metadata(img_path: str) -> dict:
    with tifffile.TiffFile(img_path, movie=True) as f:
        return f.scanimage_metadata
