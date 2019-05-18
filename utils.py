import cv2, os
import numpy as np
from sklearn.utils import shuffle


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def augment_brightness_camera_images(image):
    '''
    :param image: Input image
    :return: output image with reduced brightness
    '''

    # convert to HSV so that its easy to adjust brightness
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # randomly generate the brightness reduction factor
    # Add a constant so that it prevents the image from being completely dark
    random_bright = .25+np.random.uniform()

    # Apply the brightness reduction to the V channel
    image1[:,:,2] = image1[:,:,2]*random_bright

    # convert to RBG again
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def preprocess(img):
    """
    Combine all preprocess functions into one
    """
    img = img[60:-25, :, :]
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def choose_image(center, left, right, steering_angle, choice= np.random.choice(3), flip=np.random.choice([True, False])):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """

    img, angle = cv2.imread(center), steering_angle
    if choice == 1:
        img, angle = cv2.imread(left), steering_angle + 0.25
    elif choice == 2:
        img, angle = cv2.imread(right), steering_angle - 0.25

    img = preprocess(img)

    if flip:
        img = cv2.flip(img,1)
        angle = -1*angle

    return img, angle


def generator(paths, angles, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    
    num_samples = len(paths)
   
    while 1: 
        shuffle(paths, angles) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_paths = paths[offset:offset+batch_size]
            batch_angles = angles[offset:offset + batch_size]

            images = []
            angles = []
            for index, batch_sample in enumerate(batch_paths):
                    if is_training:
                        for i in range(0,3): #we are taking 3 images, first one is center, second is left and third is right
                            image, angle = choose_image(*batch_sample, batch_angles[index], choice=i, flip=False)
                            images.append(image)
                            angles.append(angle)

                            image, angle = choose_image(*batch_sample, batch_angles[index], choice=i, flip=True)
                            images.append(image)
                            angles.append(angle)
                    else:
                        image, angle = choose_image(*batch_sample, batch_angles[index], choice=0, flip=False)
                        images.append(image)
                        angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)
            
            
            
            
            
    # images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    # steers = np.empty(batch_size)
    # while True:
    #     i = 0
    #     for index in np.random.permutation(paths.shape[0]):
    #         center, left, right = paths[index]
    #         steering_angle = angles[index]
    #         # argumentation
    #         if is_training and np.random.rand() < 0.6:
    #             image, steering_angle = choose_image(center, left, right, steering_angle)
    #         else:
    #             image = cv2.imread(center)
    #         # add the image and steering angle to the batch
    #         images[i] = preprocess(image)
    #         steers[i] = steering_angle
    #         i += 1
    #         if i == batch_size:
    #             break
    #     yield images, steers
