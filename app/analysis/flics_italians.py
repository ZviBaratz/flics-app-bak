import PIL.Image

import scipy

import scipy.fftpack

from numpy import *

image = PIL.Image.open(r"d:\git\flics\flics_data\imges\Angoli_vena_conventional_Series012t5_6gradi.tif"
                       )  # here you have to write the name of the file

corrlimit = 50  # half of the y dimension (in pixels) since we compute the correlation function with FFTs

columnlimit = 320  # maximum distance (in pixels) between the columns to cross-correlate;

columnstep = 20  # possible column distances go from 0 to columnlimit, with step equal to columnstep (in pixels): see line 49. (usually we choose columnstep equal to 5 or 10, and columnlimit equal to 30-50, but it depends on the data)

threshold = 0  # here you can select a threshold to highlight the fluorescent diagonal lines with respect to black stripes (the intensity value of a pixel is put to zero every time it is below the chosen threshold)

# definiton of the cross-correlation function, that will be used at line 48


def correlation(t1, m1, t2, m2):

    fft1 = scipy.fftpack.fft([(t1[i]) - m1 for i in range(y)])

    fft2 = scipy.fftpack.fft([(t2[i]) - m2 for i in range(y)])

    fft1 = ma.conjugate(fft1)

    crosscorr = real(scipy.fftpack.ifft(fft1 * fft2) / (m1 * m2 * y))

    return crosscorr


# storage of the .tif file as an image matrix + data thresholding (if needed)

x = image.size[0]

y = image.size[1]

data = [[0 for i in range(y)] for j in range(x)]

a = list(image.getdata())

for i in range(y):

    for j in range(x):

        data[j][i] = float(a[i * x + j])

for i in range(y):

    for j in range(x):

        if data[j][i] < threshold:

            data[j][i] = 0.0

# computation of the average value for each image column + computation of the cross-correlation function

mean = [0.0 for i in range(x)]

for i in range(x):

    for j in range(y):

        mean[i] = mean[i] + data[i][j] / y

for i in range(
        0, columnlimit, columnstep
):  # the distance between the columns to cross-correlate varies from 0 to columnlimit, with step equal to columnstep.

    print(i)

    output = [0 for k in range(y)]

    for j in range(
            x - (i)
    ):  #For each distance, ALL the pairs of columns at that distance are exploited. The average cross-correlation function is provided (convenient for statistical reasons)

        corr = correlation(data[j + i], mean[j + i], data[j], mean[j])

        output = [output[l] + corr[l] / (x - i) for l in range(y)]
        #print('index is:', i, 'output is:', output)


    """
    res_right_to_left = []
    for k in range(corrlimit):
        res_right_to_left.append(output[k])
    print('index is:', i, 'res_right_to_left is:', res_right_to_left)

    res_left_to_right = []
    for k in range(corrlimit):
        res_left_to_right.append(output[y - k - 1])
    print('index is:', i, 'res_left_to_right is:', res_left_to_right)
    """

    """
    f = open(
        "correlation" + str(i) + "rightleft.txt", "r"
    )  # correlation functions computed from right to left columns (you obtain columlimit/columstep functions)

    for k in range(corrlimit):

        f.write(str(k) + "," + str(output[k]) + "\n")

    f.close()

    f = open(
        "correlation" + str(i) + "leftright.txt", "w"
    )  # correlation functions computed from left to right columns (you obtain columlimit/columstep functions)

    f.write(str(0) + "," + str(output[0]) + "\n")

    for k in range(corrlimit):

        f.write(str(k + 1) + "," + str(output[y - k - 1]) + "\n")
    f.close() 
    """
