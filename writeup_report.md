[//]: # (Image References)
[image1]: ./examples/nvidia.png "Model Visualization"
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create the model
* utils.py containing the script for pipeline to create the training data, validation data using generators (including preprocessing steps on input images)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run2.mp4 or video generate by using the trained model for driving autonomously

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I chose the NVIDIA model architecture for my implementation. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 16). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (utils.py line 91-97). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 23).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also flipped the data to augment the training batch samples 

For details about how I created the training data, see the data_generator() and process_batch() function in utils.py in the Pipeline class 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first crop and scale the images so that only essential information is retained in the images. This meant cropping out the sky and car's hood from the images. I then scaled the images to fit the input size of the NVIDIA model.
In addition to the NVIDIA architecture, which includes a normalization layer and 5 convolutional layers, followed by 3 fully connected layers, I also added a dropout layer, to avoid overfitting

At first, I had given the input images to the network in the sizes of 16x320x3, which is the size of the images saved by simulator. However, when I tested this model on the simulator, it failed midway in the lap. I then chose to scaled the images to a size that is expected in the NVIDIA network model

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 9-25) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and the other that involved multiple recoveries from lane boundaries 

To augment the data sat, I also flipped images and angles thinking that this would be able to provide more information to model especially while recovering in turns
After the collection process, I had 2880 number of image data points. I then preprocessed this data by cropping the images to remove unwanted information in the image and then scaled it to the size expected by the nvidia network

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I was able to get a good result within two epochs of training itself
I used an adam optimizer so that manually training the learning rate wasn't necessary.
I also included the video file (run1.mp4) for the autonomous mode achieved by this model.

Although the model is able to autonomously drive and finish the lap, we notice that some of the turns are jerky and don't resemble human behavior. This could be improved upon by providing more data for the model to train on and for more epochs.
 