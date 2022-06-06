import os

import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

from src.basic_utils import show_images, image_binary, image_canny
from src.panel_detection import panel_boxes

# Useful shorthands
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def letter_boxes(path):
    img = cv2.imread(path)
    panels = panel_boxes(path)
    lboxed = []

    print(f'num panels: {len(panels)}')
    for i, panel in enumerate(panels):
        pnl_h = panel.shape[0]

        # print(pytesseract.image_to_string(panel))

        # Tesseract's image_to_boxes for boxes around letters (terrible performance)
        col_boxes = pytesseract.image_to_boxes(panel)
        col_boxed = panel.copy()
        for box in col_boxes.splitlines():
            box = box.split(' ')
            rect_start = (int(box[1]), pnl_h - int(box[2]))
            rect_end = (int(box[3]), pnl_h - int(box[4]))
            col_boxed = cv2.rectangle(col_boxed, rect_start, rect_end, (0, 0, 255), 2)

        binary = image_binary(panel, 200)
        boxes = pytesseract.image_to_boxes(binary)
        bin_boxed = binary.copy()
        for box in boxes.splitlines():
            box = box.split(' ')
            rect_start = (int(box[1]), pnl_h - int(box[2]))
            rect_end = (int(box[3]), pnl_h - int(box[4]))
            bin_boxed = cv2.rectangle(bin_boxed, rect_start, rect_end, (0, 0, 0), 2)

        lboxed.append(col_boxed)
        lboxed.append(bin_boxed)
    print(f'num boxes: {len(lboxed)}')
    return lboxed


def word_boxes(path):
    img = cv2.imread(path)
    panels = panel_boxes(path)
    wboxed = []
    for panel in panels:
        # Tesseract's image_to_data for boxes around words (ish)
        d = pytesseract.image_to_data(panel, output_type=pytesseract.Output.DICT)
        n_boxes = len(d['level'])

        boxed_panel = panel.copy()
        for i in range(n_boxes):
            x, y, w, h = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            cv2.rectangle(boxed_panel, (x, y), (x + w, y + h), (0, 0, 255), 2)
        wboxed.append(boxed_panel)

    return wboxed


def main():
    img_name = 'insincere-apology'
    path = f'../data/images/raw/{img_name}.png'

    show_images(letter_boxes(path))


if __name__ == '__main__':
    main()

