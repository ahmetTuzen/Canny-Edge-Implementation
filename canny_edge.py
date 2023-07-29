import canny_edge_libs as lib  # functions/libraries i wrote
from PIL import Image  # using pillow for reading images
import numpy as np  # using numpy for easier matrix/filter operations


def canny_edge(imgName, gaussian_size, sigma, padding, low_threshold, high_threshold, saveAfterEveryStep):
    """
        Canny edge detection algorithm written by Ahmet Tüzen
        16.11.2020

        This function calls multiple functions on canny_edge_libs.py

        Parameters:

            imgName: image full name
            gaussian_size: size of gaussian filter, MxM
            sigma: standart deviation of gaussian filter
            padding: how to do convolution operation. 2 options: none or symmetrical
            low_threshold: low threshold value
            high_threshold: high threshold value
            saveAfterEveryStep: if true, save image file after every step.

    """
    # 1. Converting to Grayscale

    # reading image then converting to grayscale
    img = Image.open(imgName).convert("L")

    # adding parameters name to save file

    imgName = "g" + str(gaussian_size) + "_s" + str(sigma) + "_p" + padding + \
        "_lt" + str(low_threshold) + "_ht" + \
        str(high_threshold) + "_" + imgName

    # converting image data to numpy array so we can see the image as a matrix
    img = np.asarray(img, dtype="int32")

    # print(img)
    # print(img.shape)

    # saving after the first step
    if (saveAfterEveryStep):
        step1save = Image.fromarray(np.asarray(
            np.clip(img, 0, 255), dtype="uint8"), "L")
        step1save.save("step1_grayscale_" + imgName)

    # 2. Smoothing

    # defining gaussian filter with size(sizeXsize matrix) and sigma parameters
    kernel = lib.gaussian_filter(size=gaussian_size, sigma=sigma)

    # applying filter with convolution operation, using image data and gaussian filter, default padding is none, there is also symmetrical padding is defined.
    img = lib.convolution(image=img, kernel=kernel, padding=padding)

    # saving after the second step
    if (saveAfterEveryStep):
        step2save = Image.fromarray(np.asarray(
            np.clip(img, 0, 255), dtype="uint8"), "L")
        step2save.save("step2_blur_" + imgName)

    # 3. Finding Gradients

    # defining sobel filters

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # applying sobel filters
    img_sobel_x = lib.convolution(img, sobel_x, padding)

    img_sobel_y = lib.convolution(img, sobel_y, padding)

    # saving sobel filter results
    if (saveAfterEveryStep):
        step3save_1 = Image.fromarray(np.asarray(
            np.clip(img_sobel_x, 0, 255), dtype="uint8"), "L")
        step3save_1.save("step3_sobel_x_" + imgName)

        step3save_2 = Image.fromarray(np.asarray(
            np.clip(img_sobel_y, 0, 255), dtype="uint8"), "L")
        step3save_2.save("step3_sobel_y_" + imgName)

    # now, finding magnitude and theta values

    img, theta = lib.magnitude_slope(img_sobel_x, img_sobel_y)

    # print(theta)

    # saving the magnitude image
    if (saveAfterEveryStep):
        step3save_3 = Image.fromarray(np.asarray(
            np.clip(img, 0, 255), dtype="uint8"), "L")
        step3save_3.save("step3_gradient_" + imgName)

    # 4. Non-maximum suppression

    # applying Non-maximum supression
    img = lib.non_max_suppression(img, theta)

    # saving the supression result
    if (saveAfterEveryStep):
        step4save = Image.fromarray(np.asarray(
            np.clip(img, 0, 255), dtype="uint8"), "L")
        step4save.save("step4_non-maximum_" + imgName)

    # 5. Double thresholding

    # double thresholding
    img = lib.double_threshold(
        img, low_threshold=low_threshold, high_threshold=high_threshold)

    # saving after the double thresholding
    if (saveAfterEveryStep):
        step5save = Image.fromarray(np.asarray(
            np.clip(img, 0, 255), dtype="uint8"), "L")
        step5save.save("step5_doubleThresholding_" + imgName)

    # 6. Edge tracking by hysteresis:

    # applying hysteresis
    img = lib.hysteresis(img)

    # saving the final result
    if (saveAfterEveryStep):
        step6save = Image.fromarray(np.asarray(
            np.clip(img, 0, 255), dtype="uint8"), "L")
        step6save.save("step6_hysteresis_" + imgName)

    result = Image.fromarray(np.asarray(
        np.clip(img, 0, 255), dtype="uint8"), "L")
    result.save("canny_edge_" + imgName)


# canny_edge(imgName="Lenna.png", gaussian_size=3, sigma=1.4, padding="symmetrical",
#            low_threshold=3, high_threshold=40, saveAfterEveryStep=False)

# canny_edge(imgName="fruit-bowl.jpg", gaussian_size=5, sigma=5, padding="symmetrical",
#            low_threshold=3, high_threshold=40, saveAfterEveryStep=False)

# canny_edge(imgName="cameraman.jpg", gaussian_size=5, sigma=5, padding="symmetrical",
#            low_threshold=3, high_threshold=25, saveAfterEveryStep=False)

# canny_edge(imgName="house.jpg", gaussian_size=3, sigma=1.4, padding="symmetrical",
#            low_threshold=3, high_threshold=25, saveAfterEveryStep=False)

# canny_edge(imgName="woman.JPG", gaussian_size=7, sigma=3, padding="symmetrical",
#            low_threshold=5, high_threshold=25, saveAfterEveryStep=False)

