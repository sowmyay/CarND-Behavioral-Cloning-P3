
from utils import Pipeline
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
import os


def model(loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(Lambda(lambda x:  (x / 255.0) - 0.5, input_shape=(200,66,3)))
    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    return model

def main():

    # Instantiate the pipeline
    pipeline = Pipeline(model=model(), base_path="data", epochs=2)

    # Feed driving log data into the pipeline
    pipeline.import_data()
    # Start training
    pipeline.run()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    main()