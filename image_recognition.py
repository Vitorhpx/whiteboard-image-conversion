import cv2
import numpy as np
import json
from operator import itemgetter
from statistics import mean 
from image_methods import *

# First version developed, it looked good on shapes but as we dilate too mutch before thinning the image all 
# handwriting became unreadable, and complex images became a mess.
def convert_image(image_array):
    img_decoded = cv2.imdecode(image_array, cv2.COLOR_BGR2GRAY)
    colored_image = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB)
    edges = apply_canny(img_decoded)
    dilatedImg = apply_dilate(edges)
    thinnedImg = apply_thinning(dilatedImg)
    contours, hierarchy = get_contours(thinnedImg)

    return json.dumps(list(convert_contours_to_objects(contours, colored_image)), cls=NumpyEncoder)

# Second version developed, instead of using canny we opted for using threshold, this gave a better result
# recognizing handwriting and complex images as we did not simplify with canny, but on simple images we got
# a lot of strokes. We also removed shadows, this was important for the threshold algorithm to work on poor lighting conditions
def convert_image_v2(image_array):
    colored_image = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB)
    img_without_shadows = apply_remove_shadows(colored_image)
    gray_img = cv2.cvtColor(img_without_shadows, cv2.COLOR_RGB2GRAY)
    edges = apply_treshold(gray_img)
    thinnedImg = apply_thinning(edges)
    contours, hierarchy = get_contours(thinnedImg)

    return json.dumps(list(convert_contours_to_objects(contours, colored_image)), cls=NumpyEncoder)

# Another interation of the first version, this time we removed the duplicated contours by using the hierarchy given
# by the get_contour algorithm, this reduced the number of duplication in the same shape.
def convert_image_v12(image_array):
    img_decoded = cv2.imdecode(image_array, cv2.COLOR_BGR2GRAY)
    colored_image = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB)
    edges = apply_canny(img_decoded)
    dilatedImg = apply_dilate(edges)
    thinnedImg = apply_thinning(dilatedImg)
    
    contours, hierarchy = get_contours(thinnedImg)
    less_contours = get_outside_contours(contours, hierarchy)

    return json.dumps(list(convert_contours_to_objects(less_contours, colored_image)), cls=NumpyEncoder)

# Third interation of the algorith, this time we absorved the shadow removal and applied to the first algorithm.
# Removing shadows gave a better result in the canny without the need of dilation, and because of this gave
# a better result in handwriting and basic shapes, a good midpoint between the first and second algorithm.
def convert_image_v3(image_array):
    colored_image = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB)
    img_without_shadows = apply_remove_shadows(colored_image)
    gray_img = cv2.cvtColor(img_without_shadows, cv2.COLOR_RGB2GRAY)
    edges = apply_canny(gray_img)
    thinnedImg = apply_thinning(edges)
    contours, hierarchy = get_contours(thinnedImg)

    return json.dumps(list(convert_contours_to_objects(contours, colored_image)), cls=NumpyEncoder)

# A second version of the third interation, this time we applied the duplicated contour removal.
# This gave the best results with less duplication and a good enough result in both shapes and handwriting.
def convert_image_v32(image_array):
    colored_image = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB)
    img_without_shadows = apply_remove_shadows(colored_image)
    gray_img = cv2.cvtColor(img_without_shadows, cv2.COLOR_RGB2GRAY)
    edges = apply_canny(gray_img)
    thinnedImg = apply_thinning(edges)
    contours, hierarchy = get_contours(thinnedImg)
    less_contours = get_outside_contours(contours, hierarchy)

    return json.dumps(list(convert_contours_to_objects(less_contours, colored_image)), cls=NumpyEncoder)