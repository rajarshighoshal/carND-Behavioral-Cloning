# **Behavioral Cloning** 

## Udacity Self-Driving Car Engineer Nanodegree - Behavioral Cloning Project

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/NVIDIA.jpeg "Model Visualization"
[image2]: ./writeup_imgs/center_crop_sample.jpg "Sample Data Image"
[image3]: ./writeup_imgs/model_architecture.jpeg "Final Model Archgitecture"
[image4]: ./writeup_imgs/sample_img.jpg "Sample Image"
[image5]: ./writeup_imgs/cropped_sample_img.jpg "Cropped Sample Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### Model Overview
I decided to test the model provided by NVIDIA as suggested by Udacity. The model architecture is described by NVIDIA [here](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). As an input this model takes in image of the shape (60,266,3) but our dashboard images/training images are of size (160,320,3). I decided to keep the architecture of the remaining model same but instead feed an image of different input shape which I will discuss later.<br/>Here is the model architecture:
![alt text][image1]

#### Loading Data
I used the the dataset provided by Udacity
I am using OpenCV to load the images, by default the images are read by OpenCV in BGR format but we need to convert to RGB as in drive.py it is processed in RGB format.
Since we have a steering angle associated with three images we introduce a correction factor for left and right images since the steering angle is captured by the center angle.
I decided to introduce a correction factor of 0.2
For the left images I increase the steering angle by 0.2 and for the right images I decrease the steering angle by 0.2.<br/>Sample Image:<br/>
![alt text][image2]

#### Preprocessing
* I decided to shuffle the images so that the order in which images comes doesn't matters to the CNN
* Augmenting the data- i decided to flip the image horizontally and adjust steering angle accordingly, I used cv2 to flip the images.
* In augmenting after flipping multiply the steering angle by a factor of -1 to get the steering angle for the flipped image.
* So according to this approach we were able to generate 6 images corresponding to one entry in .csv file

#### Creation of the Training Set & Validation Set
* I analyzed the udacity dataset and found out that it contains 9 laps of track 1 with recovery data. I was satisfied with the data and decided to move on.
* I decided to split the dataset into training and validation set using sklearn preprocessing library.
* I decided to keep 15% of the data in Validation Set and remaining in Training Set
* I am using generator to generate the data so as to avoid loading all the images in the memory and instead generate it at the run time in batches of 32. Even Augmented images are generated inside the generators.

#### Final Model Architecture
* I made a little changes to the original NVIDIA architecture. My final architecture:
![alt_text][image3]
* As it is clear from the model summary my first step is to apply normalization to the all the images.
* Second step is to crop the image 70 pixels from top and 25 pixels from bottom. The image was cropped from top because I did not wanted to distract the model with trees and sky and 25 pixels from the bottom so as to remove the dashboard that is coming in the images.<br/>Sample Image:<br/>
![alt_text][image4]
<br/>Cropped Image:<br/>
![alt_text][image5]
* Next Step is to define the first convolutional layer with filter depth as 24 and filter size as (5,5) with (2,2) stride followed by ELU activation function
* Moving on to the second convolutional layer with filter depth as 36 and filter size as (5,5) with (2,2) stride followed by ELU activation function
* The third convolutional layer with filter depth as 48 and filter size as (5,5) with (2,2) stride followed by ELU activation function
* Next we define two convolutional layer with filter depth as 64 and filter size as (3,3) and (1,1) stride followed by ELU activation funciton
* Next step is to flatten the output from 2D to side by side
* Here we apply first fully connected layer with 100 outputs
* Here is the first time when we introduce Dropout with Dropout rate as 0.25 to combact overfitting
* Next we introduce second fully connected layer with 50 outputs
* Then comes a third connected layer with 10 outputs
* And finally the layer with one output.
Here we require one output just because this is a regression problem and we need to predict the steering angle.

#### Attempts to reduce overfitting in the model

The model contains dropout layers with dropoout rate as 0.25 in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
I have decide to use 5 number of eopchs and a batch size of 32 after some testing.
