#!/usr/bin/python

#
# Implements 8-connectivity connected component labeling
#
# Algorithm obtained from "Optimizing Two-Pass Connected-Component Labeling
# by Kesheng Wu, Ekow Otoo, and Kenji Suzuki
#

import PIL.Image, PIL.ImageDraw

import sys, os
import math, random
import numpy as np
from skimage.io import imread
from itertools import product
from TiRS.ufarray import *
# from TiRS.tirs import Tirs


def find_max_distance (component : dict):

    max_distance = -1

    for x1 in component.keys():
        y1 = component[x1]
        for x2 in component.keys():
            y2 = component[x2]
            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            max_distance = max(max_distance, distance)

    return max_distance


def find_largest_comp (labels : dict):

    largest_component = -1
    max_distance = -1
    calculated_labels = set()

    for label in labels.values():
        if label in calculated_labels or label == -1:
            continue
        calculated_labels.add(label)
        component = {}
        for (x,y) in labels:
            if labels[(x,y)] == label:
                component[x] = y
        max_distance_in_comp = find_max_distance(component)
        if max_distance_in_comp > max_distance:
            max_distance = max_distance_in_comp
            largest_component = label

    return largest_component, max_distance


def run(img):
    data = img # img is 0-1 img
    width, height = img.shape

    # Union find data structure
    uf = UFarray()

    #
    # First pass
    #

    # Dictionary of point:label pairs
    labels = {}

    for y, x in product(range(height), range(width)):

        #
        # Pixel names were chosen as shown:
        #
        #   -------------
        #   | a | b | c |
        #   -------------
        #   | d | e |   |
        #   -------------
        #   |   |   |   |
        #   -------------
        #
        # The current pixel is e
        # a, b, c, and d are its neighbors of interest
        #
        # 255 is white, 0 is black
        # White pixels part of the background, so they are ignored
        # If a pixel lies outside the bounds of the image, it default to white
        #

        # If the current pixel is white, it's obviously not a component...
        if data[x, y] == 1:
            pass

        # This is on border pixel
        elif (x ==0 or y == 0 or x == width-1 or y == height-1):
            labels[x,y] = -1

        # If pixel b is in the image and black:
        #    a, d, and c are its neighbors, so they are all part of the same component
        #    Therefore, there is no reason to check their labels
        #    so simply assign b's label to e
        elif y > 0 and data[x, y - 1] == 0:
            labels[x, y] = labels[(x, y - 1)]

        # If pixel c is in the image and black:
        #    b is its neighbor, but a and d are not
        #    Therefore, we must check a and d's labels
        elif x + 1 < width and y > 0 and data[x + 1, y - 1] == 0:

            c = labels[(x + 1, y - 1)]
            labels[x, y] = c

            # If pixel a is in the image and black:
            #    Then a and c are connected through e
            #    Therefore, we must union their sets
            if x > 0 and data[x - 1, y - 1] == 0:
                a = labels[(x - 1, y - 1)]
                uf.union(c, a)

            # If pixel d is in the image and black:
            #    Then d and c are connected through e
            #    Therefore we must union their sets
            elif x > 0 and data[x - 1, y] == 0:
                d = labels[(x - 1, y)]
                uf.union(c, d)

        # If pixel a is in the image and black:
        #    We already know b and c are white
        #    d is a's neighbor, so they already have the same label
        #    So simply assign a's label to e
        elif x > 0 and y > 0 and data[x - 1, y - 1] == 0:
            labels[x, y] = labels[(x - 1, y - 1)]

        # If pixel d is in the image and black
        #    We already know a, b, and c are white
        #    so simpy assign d's label to e
        elif x > 0 and data[x - 1, y] == 0:
            labels[x, y] = labels[(x - 1, y)]

        # All the neighboring pixels are white,
        # Therefore the current pixel is a new component
        else:
            labels[x, y] = uf.makeLabel()

    #
    # Second pass
    #

    uf.flatten()

    colors = {}

    # Image to display the components in a nice, colorful way
    output_img = PIL.Image.new("RGB", (width, height))
    outdata = output_img.load()

    # Uniting components
    for (x, y) in labels:

        # Name of the component which the current point belongs to
        component = uf.find(labels[(x, y)])

        # Update the labels with correct information
        labels[(x, y)] = component


    largest_comp, diameter_in_pixel = find_largest_comp (labels, width, height)
    print(diameter_in_pixel)

    for (x, y) in labels:

        # # Name of the component the current point belongs to
        # component = uf.find(labels[(x, y)])
        # # if x == 0 or y == 0 or x == width-1 or y == height-1:
        # #     component = -1
        #
        # # Update the labels with correct information
        # labels[(x, y)] = component

        component = labels[(x,y)]
        if component == -1:
            continue

        # Associate a random color with this component
        elif component not in colors:
            colors[component] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


        # Colorize the image
        outdata[x, y] = colors[component]

    return (labels, output_img)


def main():
    # Open the image
    data_dir = r"C:\Users\user\Desktop\Uni\Lab\Data Set - Examples"

    files = os.listdir(data_dir)
    path = os.path.join(data_dir, files[2])

    # Threshold the image, this implementation is designed to process b+w
    # images only
    img = imread(path, as_gray=True)

    # labels is a dictionary of the connected component data in the form:
    #     (x_coordinate, y_coordinate) : component_id
    #
    # if you plan on processing the component data, this is probably what you
    # will want to use
    #
    # output_image is just a frivolous way to visualize the components.
    (labels, output_img) = run(img)

    output_img.show()


if __name__ == "__main__": main()