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
def get_roi_coordinates(roi_raw: np.array) -> tuple:
    """
    gets roi as array of: [x,y,width,height]
    returns roi as array of: [x_start, x_end, y_start, y_end]
    :param roi_raw: roi array
    :type roi_raw: str
    :return:
    """
    width = round(roi_raw[2])
    height = round(roi_raw[3])
    x_center = round(roi_raw[0])
    y_center = round(roi_raw[1])
    x_start = int(x_center - width // 2)
    x_end = int(x_center + width // 2)
    x_start, x_end = sorted([x_start, x_end])
    y_start = int(y_center - height // 2)
    y_end = int(y_center + height // 2)
    y_start, y_end = sorted([y_start, y_end])
    return x_start, x_end, y_start, y_end


def crop_img(roi_coordinates: np.array, frame: int, img_path: str) -> np.ndarray:
    if roi_coordinates is None:
        return get_current_image(img_path)[frame, :, :]
    else:
        roi_parsed = np.frombuffer(roi_coordinates, dtype=np.float)
        x_start, x_end, y_start, y_end = get_roi_coordinates(roi_parsed)
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
