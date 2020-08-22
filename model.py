from urllib.request import urlretrieve
import os
import csv
from zipfile import ZipFile
import numpy as np
import cv2

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D


def download(url, file):
    if not os.path.isfile(file):
        urlretrieve(url, file)


download('https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip',
         'data.zip')  

def uncompress_features_labels(directory, name):
    if os.path.isdir(name):
        print('Data extracted')
    else:
        with ZipFile(directory) as zipfile:
            zipfile.extractall('data')

uncompress_features_labels('data.zip', 'data')

samples = []

with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples,
                                                     test_size=0.15)

def generator(sample_list, batch_size=32):
    num_samples = len(sample_list)

    while 1:
        shuffle(sample_list)  # shuffling the total images
        for offset in range(0, num_samples, batch_size):

            batch_samples = sample_list[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(0, 3):  # we are taking 3 images, first one is center, second is left and third is right

                    name = './data/data/IMG/' + batch_sample[i].split('/')[-1]
                    center_image = cv2.cvtColor(cv2.imread(name),
                                                cv2.COLOR_BGR2RGB)  
                    center_angle = float(batch_sample[3])  # getting the steering angle measurement
                    images.append(center_image)

                    # correction for left and right images
                    # if image is in left we increase the steering angle by 0.2
                    # if image is in right we decrease the steering angle by 0.2
                    if i == 0:
                        angles.append(center_angle)
                    elif i == 1:
                        angles.append(center_angle + 0.2)
                    elif i == 2:
                        angles.append(center_angle - 0.2)

                    images.append(cv2.flip(center_image, 1))
                    if i == 0:
                        angles.append(center_angle * -1)
                    elif i == 1:
                        angles.append((center_angle + 0.2) * -1)
                    elif i == 2:
                        angles.append((center_angle - 0.2) * -1)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train,
                                        y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Create model
model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

# trim image to only see section with road
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
model.add(Activation('elu'))

# layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
model.add(Activation('elu'))

# layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
model.add(Activation('elu'))

# layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))

# layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64, 3, 3))
model.add(Activation('elu'))

# flatten image from 2D to side by side
model.add(Flatten())

# layer 6- fully connected layer 1
model.add(Dense(100))
model.add(Activation('elu'))

# dropout layer
model.add(Dropout(0.25))

# layer 7- fully connected layer 1
model.add(Dense(50))
model.add(Activation('elu'))

# layer 8- fully connected layer 1
model.add(Dense(10))
model.add(Activation('elu'))

# layer 9- fully connected layer 1
model.add(
    Dense(1))  

# the output is the steering angle
# using mean squared error loss function is the right choice for this regression problem
# adam optimizer is used here
model.compile(loss='mse', optimizer='adam')


model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)


model.save('model.h5')

model.summary()
