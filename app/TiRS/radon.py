import numpy as np
import matplotlib.pyplot as plt
import math
import random
import PIL.Image

from itertools import product
from TiRS.ufarray import *
from skimage.transform import radon, rescale, iradon

"""
Set of functions that being used for TIRS
"""




def radon_transform (image : np.ndarray, show=True) -> np.ndarray:

    """
    Preforms Radon Transform for a given image as 2D numpy array

    :param image: Numpy array representing an image or ROI to analyse.
    :type image: np.ndarray
    :param show: Boolean flag which determines if the process will be printed as output.
    """

    # print(image.shape)

    image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
    theta = np.linspace(0., 180., num=180, endpoint=False)

    sinogram = radon(image, theta=theta, circle=True)
    # print(sinogram.shape)

    if show:

        fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))

        axes[0].set_title("Original")
        axes[0].imshow(image, cmap=plt.cm.Greys_r)
        axes[1].set_title("Radon transform\n(Sinogram)")
        axes[1].set_xlabel("Projection angle (deg)")
        axes[1].set_ylabel("Projection position (pixels)")
        axes[1].imshow(sinogram, cmap=plt.cm.Greys_r,
                          extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
        fig.tight_layout()
        plt.show()

    return sinogram


def calc_min_sinus(sinus : np.ndarray) -> float:

    """
    Calculates the minimum value over a given numpy array with one dimension.

    :param sinus: Numpy array representing an sinus in the transformed radon sinogram.
    :type sinus: np.ndarray

    """

    return np.amin(sinus)


def calc_max_diff(sinus : np.ndarray, min_sinus : float) -> float:

    """
    Calculates the maximum difference of a given sinus' values to the minimum value

    :param sinus: Numpy array representing an sinus in the transformed radon sinogram.
    :type sinus: np.ndarray
    :param min_sinus: The minimum value of hte given sinus
    :type min_sinus: float

    """

    max_diff = 0.
    for value in sinus:
        max_diff = max(max_diff, value-min_sinus)
    return max_diff


def normalize_sinus(sinus : np.ndarray):

    """
    Normalize a given sinus

    :param sinus: Numpy array representing an sinus in the transformed radon sinogram.
    :type sinus: np.ndarray

    """

    min_sinus = calc_min_sinus(sinus)
    max_diff = calc_max_diff(sinus, min_sinus)
    for i in range(len(sinus)):
        sinus[i] = (sinus[i] - min_sinus) / max_diff


def normalize_sinogram(sinogram : np.ndarray, show=True):

    """
    Normalize a given sinogram

    :param sinogram: Numpy array representing an radon transformed image.
    :type sinogram: np.ndarray
    :param show: Boolean flag which determines if the process will be printed as output.

    """

    for sinus in sinogram.transpose():
        normalize_sinus(sinus)

    if show:

        fig, axes = plt.subplots(1, 1, figsize=(8, 4.5))

        axes.set_title("Radon transform\n(Normalized)")
        axes.set_xlabel("Projection angle (deg)")
        axes.set_ylabel("Projection position (pixels)")
        axes.imshow(sinogram, cmap=plt.cm.Greys_r,
                          extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
        fig.tight_layout()
        plt.show()

    return sinogram


def calc_fwhm (sinus : np.ndarray, c_radon) -> (float, float):

    """
    Calculating the minimum and maximum ro, where the sinus, respectively
    exceeds and falls below the threshold Cradon

    :param sinus: Numpy array representing an sinus in the transformed radon sinogram.
    :type sinus: np.ndarray
    :param c_radon: Threshold value that determines the 'area to be manipulated' limits

    """

    ro_low = 0
    ro_high = 0
    higher = False
    for i in range(len(sinus)):
        if not higher and sinus[i] >= c_radon:
            ro_low = i if ro_low == 0 else ro_low
            higher = True
        elif higher and sinus[i] < c_radon:
            ro_high = i-1
            higher = False

    return ro_low, ro_high


def threshold_sinus(sinus : np.ndarray, c_radon :float):

    """
    Thresholding a given sinus inbetween the given limits

    :param sinus: Numpy array representing an sinus in the transformed radon sinogram.
    :type sinus: np.ndarray
    :param c_radon: Threshold value

    """

    ro_low, ro_high = calc_fwhm(sinus, c_radon)
    for i in range(len(sinus)):
        sinus[i] = 1 if ro_low <= i <= ro_high else 0


def threshold_sinogram (sinogram : np.ndarray, c_radon, show=True):

    """
    Thresholds a sinogram by thrsholding every sinus in it

    :param sinogram: Numpy array representing an radon transformed image.
    :type sinogram: np.ndarray
    :param c_radon: Threshold value.
                    Set to 0.35 according to Gao-Drew article "Thresholding in Radon Space"
    :param show: Boolean flag which determines if the process will be printed as output.

    """

    for sinus in sinogram.transpose():
        threshold_sinus(sinus, c_radon)

    if show:

        fig, axes = plt.subplots(1, 1, figsize=(8, 4.5))

        axes.set_title("Radon transform\n(Threholded)")
        axes.set_xlabel("Projection angle (deg)")
        axes.set_ylabel("Projection position (pixels)")
        axes.imshow(sinogram, cmap=plt.cm.Greys_r,
                          extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
        fig.tight_layout()
        plt.show()

    return sinogram


def iradon_transform (image : np.ndarray, sinogram : np.ndarray, show=True) -> np.ndarray:

    """
    Preforms an Inverse Radon Transform on a given transformed image

    :param image: Numpy array representing an image or ROI to analyse.
    :type image: np.ndarray
    :param sinogram: Numpy array representing a radon transformed image or ROI.
    :param show: Boolean flag which determines if the process will be printed as output.

    """
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
    theta = np.linspace(0., 180., num=180, endpoint=False)
    reconstructed_img = iradon(sinogram, theta=theta, circle=True)
    # error = reconstructed_img - image
    # print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))


    if show:
        imkwargs = dict(vmin=-0.2, vmax=0.2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                       sharex=True, sharey=True)
        ax1.set_title("Reconstruction\nFiltered back projection")
        ax1.imshow(reconstructed_img, cmap=plt.cm.Greys_r)
        # ax2.set_title("Reconstruction error\nFiltered back projection")
        # ax2.imshow(error, cmap=plt.cm.Greys_r, **imkwargs)
        plt.show()

    return reconstructed_img

# ToDo: understand the usage of the constant in heaviside function
def heaviside_thresholding (image : np.ndarray, const : float, show=True) -> np.ndarray:

    """
    Preforms a Heaviside thresholding on a given reconstructed image

    :param image: Numpy array representing an image or ROI to analyse.
    :type image: np.ndarray
    :param const: Heaviside constant
    :param show: Boolean flag which determines if the process will be printed as output.

    """

    heaviside_image = np.heaviside(image - const, 0)

    if show:
        imkwargs = dict(vmin=-0.2, vmax=0.2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                                       sharex=True, sharey=True)
        ax1.set_title("Reconstrucred Image")
        ax1.imshow(image, cmap=plt.cm.Greys_r)
        ax2.set_title("Full Radon Transform\n(After Heaviside)")
        ax2.imshow(heaviside_image, cmap=plt.cm.Greys_r, **imkwargs)
        plt.show()

    return heaviside_image


def find_max_distance (component : set):

    """
    Finds the maximum euclidean distance between points in the given component
    of contiguous pixels of a labeled image

    :param component: Dictionary represents connected points (by 8 connectedness
     algorithm) in an image as (x,y)
    :type component: set

    """

    max_distance = 0

    for point1 in component:
        x1 = point1[0]
        y1 = point1[1]
        for point2 in component:
            x2 = point2[0]
            y2 = point2[1]
            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            max_distance = max(max_distance, distance)

    return max_distance


def find_largest_comp (labels : dict):

    """
    Finds the largest component of contiguous pixels in a labeled image according
    to euclidean distance

    :param labels: Dictionary represents points (x,y) of an image and their label z : ((x,y),z)
    :type labels: dict

    """

    largest_component = -1
    max_distance = 0

    for label in set(labels.values()):
        if label == 0:  # label represents pixels that are contiguous with image's borders
            continue
        component = set()
        for (x, y) in labels:
            if labels[(x, y)] == label:
                component.add((x, y))
        max_distance_in_comp = find_max_distance(component)
        if max_distance_in_comp > max_distance:
            max_distance = max_distance_in_comp
            largest_component = label

    return largest_component, max_distance


def labeling (data : np.ndarray):

    """
    Labeling pixels of a given image.
    For details see documentation of connectedness function

    :param image: Numpy array representing an image or ROI to analyse.
    :type image: np.ndarray
    """

    uf = UFarray()
    uf.makeLabel()  # label = 0 for backround
    labels = {}

    width, height = data.shape
    for y, x in product(range(height), range(width)):

        if data[x, y] == 1:  # white
            pass

        elif x == 0 or y == 0 or x == width - 1 or y == height - 1:  # Pixel osculates the border
            labels[x, y] = 0

        elif y > 0 and data[x, y - 1] == 0:
            labels[x, y] = labels[(x, y - 1)]

        elif x + 1 < width and y > 0 and data[x + 1, y - 1] == 0:

            c = labels[(x + 1, y - 1)]
            labels[x, y] = c

            if x > 0 and data[x - 1, y - 1] == 0:
                a = labels[(x - 1, y - 1)]
                uf.union(c, a)

            elif x > 0 and data[x - 1, y] == 0:
                d = labels[(x - 1, y)]
                uf.union(c, d)

        elif x > 0 and y > 0 and data[x - 1, y - 1] == 0:
            labels[x, y] = labels[(x - 1, y - 1)]

        elif x > 0 and data[x - 1, y] == 0:
            labels[x, y] = labels[(x - 1, y)]

        else:  # All the neighboring pixels are white, therefore the current pixel is a new component
            labels[x, y] = uf.makeLabel()

    return data, labels, uf

# ToDo: Decrease running time from O(n**2)
def connectedness(image : np.ndarray, show=True):

    """
    Applies 8 connectedness algorithm on a given binary image.
    Finds the largest contiguous component of pixels,
    and returns the maximum euclidean distance between pixels in that component.
        *** Largest component can not osculate the borders of the given image

    Uses ufarray as Union-Find implementation for the algorithm.
    For further information about the algorithm of about ufarray, see:
        https://github.com/spwhitt/cclabel

    :param show: Boolean flag which determines if the process will be printed as output,
    and an image will be generated
    """

    data = image
    data, labels, uf = labeling(data)
    uf.flatten()

    for (x,y) in labels:
        data[x][y] = labels[(x,y)]

    for (x, y) in labels:     # Uniting labels to components
        if labels[(x,y)] == 0:    # (x,y) osculates the image's border
            continue
        component = uf.find(labels[(x, y)])
        labels[(x, y)] = component

    largest_comp, diameter_in_pixel = find_largest_comp(labels)

    output_img = None

    if show:

        colors = {}
        width, height = data.shape
        output_img = PIL.Image.new("RGB", (width, height))
        outdata = output_img.load()

        for (x, y) in labels:
            component = labels[(x, y)]
            if component == 0:  # component osculates the image's border
                continue
            elif component not in colors:
                colors[component] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            outdata[x, y] = colors[component]

    return diameter_in_pixel, output_img
