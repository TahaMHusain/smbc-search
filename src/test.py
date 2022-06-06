import os

import numpy as np
import cv2

from basic_utils import image_binary, show_images


def process_panel(panel, kernel_size=(2, 1), iterations=2, t=130):
    display = [panel]

    # kernel = np.ones(kernel_size, np.uint8)
    # img_dilated = cv2.dilate(panel, kernel, iterations=iterations)
    # display.append(img_dilated)

    img_binary = image_binary(panel, t)
    display.append(img_binary)

    # img_dilated_binary = image_binary(cv2.dilate(panel, kernel, iterations=iterations), t)
    # display.append(img_dilated_binary)

    panel = cv2.normalize(panel, None, alpha=0, beta=244, norm_type=cv2.NORM_MINMAX)
    res, panel = cv2.threshold(panel, 64, 255, cv2.THRESH_BINARY)

    cv2.floodFill(panel, None, (0, 0), 255)
    cv2.floodFill(panel, None, (0, 0), 0)
    display.append(panel)

    return display


def display(path, single_panel=False, params=None):
    params = {} if not params else params
    if single_panel:
        panel = cv2.imread(path)
        show_images(process_panel(panel, **params))
    else:
        panels = [cv2.imread(f.path) for f in os.scandir(path)]
        display_panels = [process_panel(panel, **params) for panel in panels]
        show_images(display_panels)


def main():
    img_name = 'bigger-half'
    single_panel = True
    params = {
        'kernel_size': (1, 1),
        'iterations': 0,
        't': 100,
    }

    if single_panel:
        path = f'../data/images/raw/{img_name}.png'
    else:
        path = f'../data/images/panel-boxed/{img_name}'

    display(path, single_panel=single_panel, params=params)


if __name__ == '__main__':
    main()

