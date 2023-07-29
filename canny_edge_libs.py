import numpy as np


def gaussian_filter(size, sigma):
    """    
    Returns given gaussian filter with given size and sigma values.

    Parameters:

       size (int): Gaussian filter's matrix size (sizeXsize)
       sigma (double): sigma value of filter

    Returns:

    gaussian filter    

    """

    # for first step i wanna check if the size is odd number. If not i ll throw an exception.
    if (size % 2 == 0):
        raise Exception("Filter size must be odd number")

    # now i ll determine x and y matrix partials, so i can apply those values on formula
    x = np.zeros((size, size))
    y = np.zeros((size, size))

    # next i ll find the center of matrix and other pixels would be distance from the center respect to x and y
    size = (size - 1) / 2

    for i in range(x[0].size):
        for j in range(x[0].size):
            x[i][j] = i - size

    y = np.transpose(x)

    # for example for 3X3 matrix
    #     -1 -1 -1           -1  0  1
    # x =  0  0  0       y = -1  0  1
    #      1  1  1           -1  0  1

    # now i can apply the formula
    filter = (1 / (2 * np.pi * (sigma ** 2))) * \
        np.exp(-(x ** 2 + y ** 2)/(2*(sigma**2)))

    # normalizing filter
    filter /= np.sum(filter)

    return filter


def convolution(image, kernel, padding):
    """
    Convolution operation.

    Parameters:

       image (np.array):    image file as numpy array
       kernel (np.array):   filter 
       padding (string):    defines that convolution operation gonna make padding or not.
                            none: no padding
                            symmetrical: fill padding with symmetrical of values

    Returns:

    operation result  

    """

    # getting symmetrical of kernel for convolution first, (so i can do dot product)
    conv_filter = np.flip(kernel)

    # getting shapes of image data
    img_x, img_y = image.shape

    # getting shapes of filter
    ker_x, ker_y = kernel.shape

    # finding differences
    diff = (ker_x - 1) / 2
    diff = int(diff)

    # copying image as a result so we can keep both original image data and result
    result = image.copy()

    # defining none padding
    if (padding == "none"):
        for i in range(diff, img_x-diff):
            for j in range(diff, img_y-diff):
                conv = 0
                for x in range(ker_x):
                    for y in range(ker_y):
                        # dot product
                        conv += image[i+x-diff][j+y-diff] * conv_filter[x][y]
                # filling result data
                result[i][j] = conv

    # defining symmetrical padding
    elif (padding == "symmetrical"):
        for i in range(0, img_x):
            for j in range(0, img_y):
                conv = 0
                for x in range(ker_x):
                    for y in range(ker_y):
                        # checking if we hit negative
                        if (((i+x-diff) < 0) or ((j+y-diff) < 0)):
                            # determining which axis hit the negative
                            X = i+x-diff
                            Y = j+y-diff
                            if (X < 0):
                                X = -X
                            if (Y < 0):
                                Y = -Y

                            # checking if we hit beyond the edges of data
                            if (X >= img_x):
                                X = X - 2 * diff

                            if (Y >= img_y):
                                Y = Y - 2 * diff

                            # dot product
                            conv += image[X][Y] * conv_filter[x][y]

                        else:
                            # checking if we hit beyond the edges of data
                            X = i+x-diff
                            Y = j+y-diff

                            if (X >= img_x):
                                X = X - 2 * diff
                            if (Y >= img_y):
                                Y = Y - 2 * diff

                            # dot product
                            conv += image[X][Y] * \
                                conv_filter[x][y]
                # filling result data
                result[i][j] = conv

    else:
        raise Exception("Padding is not defined")

    return result


def magnitude_slope(imageX, imageY):
    """    
    Calculates magnitude and slope of gradients

    Parameters:

       imageX (np.array): X partial derivation result
       imageY (np.array): Y partial derivation result

    Returns:

    gradient and slope(theta)

    """

    # getting shapes of inputs, both inputs have same shapes, so i ll use one
    x, y = imageX.shape

    # making result matrix as an array of zeros
    result = np.zeros((x, y))

    # filling result matrix
    for i in range(x):
        for j in range(y):
            result[i][j] = (imageX[i][j]**2 + imageY[i][j]**2)**(1/2)

    # finding thetas
    theta = np.arctan2(imageY, imageX)

    # normalization of result
    result = result / result.max() * 255

    return result, theta


def non_max_suppression(gradient, theta):
    """    
    Determines direction and direction of neighbours. Then find local maximums

    Parameters:

       gradient (np.array): Gradient image as numpy array
       theta (np.array):    Slope array

    Returns:

    Local maxima

    """

    # getting shapes
    x, y = gradient.shape

    # making result matrix as an array of zeros
    result = np.zeros((x, y))

    # important part is theta has the values from -pi to pi, for purpose of easier operations, i ll add value pi to negative values
    theta[theta < 0] += np.pi

    # next, determine the neighbours

    for i in range(1, x-1):
        for j in range(1, y-1):

            # now i ll determine 2 neighbours on the direction of theta. Lets initialize them first
            # (picking initial value of 255 so any other pixel value cannot be greater than this)
            n1 = 255
            n2 = 255

            # first part is determine the direction. Notice that we using angle from 0 to pi. (because we add pi to negative values)
            # i have total of 4 directions, and each neighboorhood has total pi / 4 angle value

            # (0 to pi / 8) and (7 * pi / 8 to pi) are the same directions
            if ((0 <= theta[i][j] < np.pi / 8) or (7 * np.pi / 8 < theta[i][j] <= np.pi)):
                n1 = gradient[i][j+1]
                n2 = gradient[i][j-1]

            # pi / 8 to 3 * pi / 8
            elif (np.pi / 8 <= theta[i][j] < 3 * np.pi / 8):
                n1 = gradient[i+1][j-1]
                n2 = gradient[i-1][j+1]

            # 3 * pi / 8 to 5 * pi / 88
            elif (3 * np.pi / 8 <= theta[i][j] < 5 * np.pi / 8):
                n1 = gradient[i+1][j]
                n2 = gradient[i-1][j]

            # 5 * pi / 8 to 7 * pi / 8
            elif (5 * np.pi / 8 <= theta[i][j] < 7 * np.pi / 8):
                n1 = gradient[i-1][j-1]
                n2 = gradient[i+1][j+1]

            # non-max supression
            if ((gradient[i][j] >= n1) and (gradient[i][j] >= n2)):
                result[i][j] = gradient[i][j]
            else:
                result[i][j] = 0

    return result


def double_threshold(image, low_threshold, high_threshold):
    """    
    Split image data into 3 values: 0, 100 and 255 (black, thin, thick)

    Parameters:

       image (np.array):            image data as numpy array
       low_threshold (double):      Low threshold value
       high_threshold (double):     High threshold value


    Returns:

    Image data after double thresholding

    """

    # getting shapes
    x, y = image.shape

    # making result matrix as an array of zeros
    result = np.zeros((x, y))

    # i define edges as thin or thick. thick edges are greater than high_threshold (final edges),
    # thin edges are greater low_threshold but lower than high threshold (potentially edges)

    # defining the edge values where thin edges value is 100 which is close to gray and thick edges as white (255)
    thin = 100
    thick = 255

    # checking image matrix
    for i in range(x):
        for j in range(y):
            # if value of pixel greater than high threshold, make it 255.
            if (image[i][j] >= high_threshold):
                result[i][j] = thick
            # if value of pixel greater than lower threshold, and lower than high threshold, make it 100.
            elif (low_threshold <= image[i][j] < high_threshold):
                result[i][j] = thin
            # i use other pass for other cases because i defined result as zero matrix. so leave their values as it is.
            else:
                pass

    return result


def hysteresis(image):
    """    
    Checks if the thin line connected on thick line or not


    Parameters:

       image (np.array):    double thresholded image data as numpy array


    Returns:

    Image data with strong lines = Canny Edge Detection result

    """
    # getting shapes
    x, y = image.shape

    # making result of image copy of image data, so i can keep the original values and compare with it.
    result = image.copy()

    # same values on double_threshold function.
    thin = 100
    thick = 255

    # defining final edges, thick edges are already in it. Lets check for thin ones:
    for i in range(x):
        for j in range(y):
            if (image[i][j] == thin):
                try:
                    # checking 8 neighbours of thin layer, if any of them is thick, make it thick.
                    if ((image[i+1][j+1] == thick) or (image[i+1][j] == thick) or
                        (image[i+1][j-1] == thick) or (image[i][j+1] == thick) or
                        (image[i][j-1] == thick) or (image[i-1][j+1] == thick) or
                            (image[i-1][j] == thick) or (image[i-1][j-1] == thick)):
                        result[i][j] = thick
                    else:
                        result[i][j] = 0
                # using try except block for if we hitting out of boundaries.
                except:
                    pass
    return result
