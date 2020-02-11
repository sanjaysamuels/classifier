# -*- coding: utf-8 -*-
"""
Spyder Editor

Code by: Sanjay Sekar Samuel
"""


'''
Below are the modeules needed to be exported to build the CNN
Followed by the modlules needed to preprocess the image
'''
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D


from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.models import model_from_json
import numpy
import os


Classifier_DN = Sequential()  # This will import the neural network as sequential

'''
STEP 1: CONVOLUTION LAYER

This is the convolutional layer, and this 2D deals with the 2D array of images
Here the first parameter: filter
Second parameter: shape of filter
Third paramter: shape of image
Fourth parameter: activation function relu
'''
Classifier_DN.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

'''
STEP 2: POOLING LATER

After the convolutional layer, we perform max pooling to the image to extract the maximum values from the pixels
This takes a 2x2 matrix to reduce the number of nodes to the next layer
'''
Classifier_DN.add(MaxPooling2D(pool_size = (2, 2)))


'''
STEP 3: FLATTENING LAYER

Flattening is then done to convert the 2D array to a single linear vector
'''
Classifier_DN.add(Flatten())


'''
STEP 4: FULLY CONNECTED LAYER

Here the input from flatterning goes to the dese function which is the fully connected layer with the activation function of relu and sigmoid
they are also refered to as hidden layers
Output node can only have 1 node because this is a binary Classifier_DN
'''
Classifier_DN.add(Dense(units = 64, activation = 'relu'))
Classifier_DN.add(Dense(units = 128, activation = 'relu'))
Classifier_DN.add(Dense(units = 1, activation = 'sigmoid'))
Classifier_DN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Preprocessing the data to get the right input image for the network
'''
Preprocessing code is taken from keras website
https://keras.io/preprocessing/image/
'''
train_data_process = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data_process = ImageDataGenerator(rescale = 1./255)    # This is to make the picture to have a value between 0 and 1
training_dataset = train_data_process.flow_from_directory('car_plane_dataset/training_dataset', target_size = (64, 64), batch_size = 16, class_mode = 'binary')
testing_dataset = test_data_process.flow_from_directory('car_plane_dataset/testing_dataset', target_size = (64, 64), batch_size = 16, class_mode = 'binary')


Classifier_DN.fit_generator(training_dataset, steps_per_epoch = 20, epochs = 10, validation_data = testing_dataset, validation_steps = 2000)


'''
The code below saves the weightage in JSON h5py format to be used for future prediction
'''
model_json = Classifier_DN.to_json()
with open("Classifier_DN.json", "w") as json_file:
    json_file.write(model_json)
Classifier_DN.save_weights("Classifier_DN.h5")
print("Saved model to disk")

# The code below will load the image that needed to be predicted and make the prediction
prediction_image = image.load_img('car_plane_dataset/validation/1.jpg', target_size = (64, 64))
prediction_image = image.img_to_array(prediction_image)
prediction_image = numpy.expand_dims(prediction_image, axis = 0)
result = Classifier_DN.predict(prediction_image)
training_dataset.class_indices
if result[0][0] == 1:
    print('This image is a plane')
else:
    print('This image is a car')
