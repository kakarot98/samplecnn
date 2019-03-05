# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 20:00:26 2019

@author: Ashish
"""

#download cifar10 dataset - https://www.cs.toronto.edu/~kriz/cifar.html

import pickle
import numpy as np
from keras.utils import np_utils

def load_cifar10_data(batch_number):
    
    with open('C:/Users/Ashish/Desktop/ML tuts/Practice CNN/cifar-10-batches-py/data_batch_' + str(batch_number), 'rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data']
    labels = batch['labels']
    return features, labels

batch_1, labels_1 = load_cifar10_data(1)
batch_2, labels_2 = load_cifar10_data(2)
batch_3, labels_3 = load_cifar10_data(3)
batch_4, labels_4 = load_cifar10_data(4)
batch_5, labels_5 = load_cifar10_data(5)


X_train = np.append(batch_1[0:8000], batch_2[0:8000], axis=0)
X_train = np.append(X_train,batch_3[0:8000],axis=0)
X_train = np.append(X_train,batch_4[0:8000],axis=0)
X_train = np.append(X_train,batch_5[0:8000],axis=0)
print("Length of X_train:", len(X_train))

Y_train = np.append(labels_1[0:8000], labels_2[0:8000], axis=0)
Y_train = np.append(Y_train, labels_3[0:8000], axis=0)
Y_train = np.append(Y_train, labels_4[0:8000], axis=0)
Y_train = np.append(Y_train, labels_5[0:8000], axis=0)
print("Length of y_train:", len(Y_train))

'''can use test set as validation set'''

X_validation = np.append(batch_1[8001:9000], batch_2[8001:9000], axis=0)
X_validation = np.append(X_validation, batch_3[8001:9000], axis=0)
X_validation = np.append(X_validation, batch_4[8001:9000], axis=0)
X_validation = np.append(X_validation, batch_5[8001:9000], axis=0)

Y_validation = np.append(labels_1[8001:9000], labels_2[8001:9000], axis=0)
Y_validation = np.append(Y_validation, labels_3[8001:9000], axis=0)
Y_validation = np.append(Y_validation, labels_4[8001:9000], axis=0)
Y_validation = np.append(Y_validation, labels_5[8001:9000], axis=0)

X_test = np.append(batch_1[9001:10000], batch_2[9001:10000], axis=0)
X_test = np.append(X_test, batch_3[9001:10000], axis=0)
X_test = np.append(X_test, batch_4[9001:10000], axis=0)
X_test = np.append(X_test, batch_5[9001:10000], axis=0)

Y_test = np.append(labels_1[9001:10000], labels_2[9001:10000], axis=0)
Y_test = np.append(Y_test, labels_3[9001:10000], axis=0)
Y_test = np.append(Y_test, labels_4[9001:10000], axis=0)
Y_test = np.append(Y_test, labels_5[9001:10000], axis=0)

Y_train_one_hot = np_utils.to_categorical(Y_train, 10)
Y_validation_one_hot = np_utils.to_categorical(Y_validation, 10)
Y_test_one_hot = np_utils.to_categorical(Y_test, 10)

X_train = X_train.reshape((len(X_train), 3, 32, 32)).transpose(0,2,3,1)
X_validation = X_validation.reshape((len(X_validation), 3, 32, 32)).transpose(0,2,3,1)
X_test = X_test.reshape((len(X_test), 3, 32, 32)).transpose(0,2,3,1)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json


classifier = Sequential()

classifier.add(Convolution2D(32,(3,3),padding='same', input_shape=(32,32,3), activation='relu'))
classifier.add(Convolution2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(32,(3,3),padding='same',activation='relu'))
classifier.add(Convolution2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25)) 

classifier.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
 
classifier.add(Flatten())
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(10,activation='softmax'))

classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)
train_set = train_datagen.flow(X_train, Y_train_one_hot, batch_size=32)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_set = validation_datagen.flow(X_validation, 
                                         Y_validation_one_hot, 
                                         batch_size=32)


classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit_generator(train_set,
                    steps_per_epoch=40000,epochs=10,
                    validation_data=(validation_set), 
                    validation_steps=4995, shuffle=True)


#The following code is to save your model's weights and then load it back again---------------------------

# serialize model to JSON
classifier_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

