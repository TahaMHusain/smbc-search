import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1000 is height of my monitor, so ensure images are shorter than that
MAX_HEIGHT = 1000
# Useful shorthands
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# for tall images so they fit on a monitor without squishing
def vert_split(image):
    num_windows = (image.shape[0] // MAX_HEIGHT) + 1
    return [image[i * MAX_HEIGHT: (i + 1) * MAX_HEIGHT] for i in range(num_windows) if
            image[i * MAX_HEIGHT: (i + 1) * MAX_HEIGHT].shape[0] > 0]


def show_images(images, limit=6):
    images = [images] if isinstance(images, np.ndarray) else images
    for i in range(len(images)):
        split_imgs = vert_split(images[i]) if images[i].shape[0] > 1000 else [images[i]]
        for j in range(min(len(split_imgs), limit)):
            title = f'{i}'
            if j > 0:
                title = ''.join([title, f' part {j}'])
            cv2.imshow(title, split_imgs[j])
            # plt.imshow(split_imgs[j])
            # plt.show()
    cv2.waitKey(0)


def image_binary(image, t=150):
    # make grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # make binary
    return cv2.threshold(image_gray, t, 255, cv2.THRESH_BINARY)[1]


def image_canny(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(image_gray, 50, 200)
