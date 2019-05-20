import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self, model=None, base_path='data', epochs=2):
        self.data = []
        self.model = model
        self.epochs = epochs
        self.training_samples = []
        self.validation_samples = []
        self.correction_factor = 0.2
        self.base_path = base_path
        self.image_path = self.base_path + '/IMG/'
        self.driving_log_path = self.base_path + '/driving_log.csv'
        self.driving_log_udacity_path = self.base_path + '/driving_log_udacity.csv'
        self.image_size = (66, 200)

    def import_data(self):
        print(self.driving_log_path)
        with open(self.driving_log_path) as csvfile:
            reader = csv.reader(csvfile)
            # Skip the column names row
            next(reader)

            for line in reader:
                self.data.append(line)

        return None

    def process_batch(self, batch_sample):
        steering_angle = np.float32(batch_sample[3])
        images, steering_angles = [], []

        for image_path_index in range(3):
            image_name = batch_sample[image_path_index].split('/')[-1]

            image = cv2.imread(self.image_path + image_name)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cropped = rgb_image[60:-25, :, :]
            resized = cv2.resize(cropped, self.image_size)

            images.append(resized)

            if image_path_index == 1:
                steering_angles.append(steering_angle + self.correction_factor)
            elif image_path_index == 2:
                steering_angles.append(steering_angle - self.correction_factor)
            else:
                steering_angles.append(steering_angle)

            if image_path_index == 0:
                flipped_center_image = cv2.flip(resized, 1)
                images.append(flipped_center_image)
                steering_angles.append(-steering_angle)

        return images, steering_angles

    def data_generator(self, samples, batch_size=128):
        num_samples = len(samples)

        while True:
            shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images, steering_angles = [], []

                for batch_sample in batch_samples:
                    augmented_images, augmented_angles = self.process_batch(batch_sample)
                    images.extend(augmented_images)
                    steering_angles.extend(augmented_angles)

                X_train, y_train = np.array(images), np.array(steering_angles)
                yield shuffle(X_train, y_train)

    def split_data(self):
        train, validation = train_test_split(self.data, test_size=0.2)
        self.training_samples, self.validation_samples = train, validation

        return None
    
    def run(self):
        self.split_data()
        train_generator = self.data_generator(samples=self.training_samples, batch_size=128)
        validation_generator = self.data_generator(samples=self.validation_samples, batch_size=128)
        history_object = self.model.fit_generator(generator=train_generator,
                                 validation_data=validation_generator,
                                 epochs=self.epochs,
                                 steps_per_epoch=len(self.training_samples) * 2,
                                 validation_steps=len(self.validation_samples))
        print(history_object.history.keys())
        ### plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig('loss_plot.jpg')
        self.model.save('model.h5')

def crop_and_resize(rgb_image, image_size=(66, 200)):
    cropped = rgb_image[60:-25, :, :]
    resized = cv2.resize(cropped, image_size)
    return resized
