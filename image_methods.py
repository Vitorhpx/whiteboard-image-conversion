import cv2
import numpy as np
import json
from statistics import mean
from operator import itemgetter

def read_image_with_color(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB)

def get_color_for_contour(contour, image):
    colors = [image[point[0][1], point[0][0]] for point in contour]
    red_channel = mean([color[0] for color in colors])
    green_channel = mean([color[1] for color in colors])
    blue_channel = mean([color[2] for color in colors])
    return [red_channel, green_channel, blue_channel]

def get_color_for_contours(contours, colored_image):
    return [get_color_for_contour(contour, colored_image) for contour in contours]

def read_image(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_canny(image):
    return cv2.Canny(image, 25, 150, L2gradient=True)

def apply_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 15)

def apply_dilate(image):
    kernel = np.ones((13, 13), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def apply_thinning(image):
    return cv2.ximgproc.thinning(image, cv2.ximgproc.THINNING_ZHANGSUEN)

def apply_erosion(image):
    return cv2.erode(image, np.ones((7, 7), np.uint8), iterations=1)

def apply_remove_shadows(image):
    rgb_planes = cv2.split(image)
    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    result = cv2.merge(result_planes)
    return result

def get_contours(image):
    return cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

def get_outside_contours(contours, hierarchy):
    real_contours = []
    hier = list(enumerate(hierarchy[0]))
    curr = hier[0]
    next_contour = curr[1][0]
    while next_contour != -1:
        real_contours.append(contours[curr[0]])
        curr = hier[next_contour]
        next_contour = curr[1][0]
    real_contours.append(contours[curr[0]])
    return real_contours

def map_to_object(contour, colored_image):
    colors = get_color_for_contour(contour, colored_image)
    return {
        "color": {
            "r": int(colors[2]),
            "g": int(colors[1]),
            "b": int(colors[0])
        },
        "points": [point[0] for point in contour],
        "thickness": 7
    }

def convert_contours_to_objects(contours, colored_image):
    return [map_to_object(contour, colored_image) for contour in contours]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
