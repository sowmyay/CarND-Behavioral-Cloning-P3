import cv2, os
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def preprocess(img):
    """
    Combine all preprocess functions into one
    """
    img = img[60:-25, :, :]
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img


def choose_image(center, left, right, steering_angle, choice= np.random.choice(3), flip=False):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """

    img, angle = cv2.imread(center), steering_angle
    if choice == 0:
        img, angle = cv2.imread(left), steering_angle + 0.2
    elif choice == 1:
        img, angle = cv2.imread(right), steering_angle - 0.2

    if flip:
        img = cv2.flip(img,1)
        angle = -1*angle

    return img, angle


def generator(paths, angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(paths.shape[0]):
            center, left, right = paths[index]
            steering_angle = angles[index]
            # argumentation
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = choose_image(center, left, right, steering_angle, flip=(np.random.rand()<0.5))
            else:
                image = cv2.imread(center)
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
