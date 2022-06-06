import os
from pathlib import Path

import pytesseract

import numpy as np
import cv2

from src.basic_utils import show_images, image_binary, image_canny
from src.panel_detection import panel_boxes


def ocr(path, single_panel=False):
    if single_panel:
        panel = cv2.imread(path)
        panel = process_panel(panel)
        return pytesseract.image_to_string(panel)

    panels = panel_boxes(path)
    text = []

    for panel in panels:
        panel = process_panel(panel)
        text.append(pytesseract.image_to_string(panel))

    return text


def process_panel(panel):
    panel = image_binary(panel, t=130)

    panel = cv2.normalize(panel, None, alpha=0, beta=244, norm_type=cv2.NORM_MINMAX)
    res, panel = cv2.threshold(panel, 64, 255, cv2.THRESH_BINARY)

    cv2.floodFill(panel, None, (0, 0), 255)
    cv2.floodFill(panel, None, (0, 0), 0)

    kernel = np.ones((2, 1), np.uint8)
    panel = cv2.dilate(panel, kernel, iterations=2)

    return panel


def main():
    img_name = 'real-job'
    path = f'../data/images/raw/{img_name}.png'
    print(pytesseract.image_to_string(path))
    print(f'{"="*20}\n{"="*20}\n{"="*20}')
    ocr(path, single_panel=True)


if __name__ == '__main__':
    main()
