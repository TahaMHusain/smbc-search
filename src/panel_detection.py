import cv2
import scipy.ndimage
import numpy as np

from src.basic_utils import show_images, image_binary, image_canny

# Max panels per horizontal line, for better panel detection; 4 for SMBC
MAX_HORIZONTAL_PANELS = 4
# Useful shorthands
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_boxes(img, slices, color=(0, 0, 255)):
    ret = img.copy()
    for y, x in slices:
        ret[(y.start, y.stop - 1), x] = color
        ret[y, (x.start, x.stop - 1)] = color
    return ret


def clean_box_slices(img, boxes):
    w = img.shape[1]
    h = img.shape[0]

    # Don't include boxes that are too small to be panels, or are just the whole image
    panel_width = (w // MAX_HORIZONTAL_PANELS) - int(w * 0.01)
    boxes2 = [box for box in boxes if box[1].stop - box[1].start > panel_width and
              (box[0].start > 10 or box[1].start > 10 or box[0].stop < (h - 40) or box[1].stop < (w - 10)) and
              box[0].stop - box[0].start > 10]
    # Try to take out boxes that are within panels (other boxes)
    box_slices2 = []
    for box in boxes2:
        is_inside_box = False
        for box2 in boxes2:
            if box[0].start > box2[0].start and box[1].start > box2[1].start \
                    and box[0].stop < box2[0].stop and box[1].stop < box2[1].stop:
                is_inside_box = True
        if not is_inside_box:
            box_slices2.append(box)

    return box_slices2


# Creates boxes around panels
def panel_boxes(path):
    img = cv2.imread(path)

    # Create mask for borders (border if pixels above or to the left are different)
    up = np.any((img[1:, 1:] != img[:-1, 1:]), axis=2)
    left = np.any((img[1:, 1:] != img[1:, :-1]), axis=2)
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[1:, 1:] = up | left
    # Draw mask boundaries for debugging
    mask_image = np.zeros(img.shape)
    mask_image[~mask] = (255, 255, 255)

    # Mask panel boxes
    mask_labels, n = scipy.ndimage.label(mask)
    mask_boxes_raw = scipy.ndimage.find_objects(mask_labels)
    mask_boxes = clean_box_slices(img, mask_boxes_raw)
    mask_panels = [img[y, x] for y, x in mask_boxes]

    # # Binary panel box detection - doesn't work as well as masks?
    # binary = image_binary(img, 200)
    # binary_labels, n = scipy.ndimage.label(binary)
    # binary_boxes_raw = scipy.ndimage.find_objects(binary_labels)
    # binary_boxes = clean_box_slices(img, binary_boxes_raw)
    # binary_panels = [img[y, x] for y, x in binary_boxes]

    return mask_panels


def panel_boxes_hough(path):
    img = cv2.imread(path)
    binary = image_binary(img)
    canny = image_canny(img)

    min_length = 1000
    max_gap = 0
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 1, min_length, max_gap)
    line_dst = np.zeros(img.shape)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) > 0 else 1001
        length = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))
        if (slope < 0.1 or slope > 1000) and length > 5:
            cv2.line(line_dst, (x1, y1), (x2, y2), (0, 0, 255))

    return line_dst


def main():
    img_name = 'three-year-gym-membership'
    path = f'../data/images/raw/{img_name}.png'
    display = panel_boxes(path)
    for i in range(len(display)):
        opath = f'../data/images/panel-boxed/{img_name}/{i}.png'
        cv2.imwrite(opath, display[i])
    for d in display:
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(d, kernel, iterations=1)
    show_images(display)


if __name__ == '__main__':
    main()
