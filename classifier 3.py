# -*- coding: utf-8 -*-
"""
Spyder Editor

Code by: Sanjay Sekar Samuel
"""

'''
Below are the modeules needed to be exported to build the CNN
Followed by the modlules needed to preprocess the image
'''
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy
import os



'''
The code below loads the stored weightage from the previous code and uses it to predict the new image
'''
json_file = open('Classifier_DN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("Classifier_DN.h5")
print("Loaded the h5py file from the disk, ready for prediction!")



'''
Preprocessing code is taken from keras website
https://keras.io/preprocessing/image/
'''
train_data_process = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_data_process = ImageDataGenerator(rescale = 1./255)    # This is to make the picture to have a value between 0 and 1
training_dataset = train_data_process.flow_from_directory('car_plane_dataset/training_dataset', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
testing_dataset = test_data_process.flow_from_directory('car_plane_dataset/testing_dataset', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

'''
The code below will load the image that needed to be predicted and make the prediction
For car choose the '1.jpg' and for plain '2.jpg'
'''
prediction_image = image.load_img('car_plane_dataset/validation/1.jpg', target_size = (64, 64))     # To get the image in the 64x64
prediction_image = image.img_to_array(prediction_image)
prediction_image = numpy.expand_dims(prediction_image, axis = 0)
result = loaded_model.predict(prediction_image)
training_dataset.class_indices
if result[0][0] == 1:
    print("The predicted image is a plane")
else:
    print('The predicted image is a car')
