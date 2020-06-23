import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
import PIL

train_path = '/home/reshma/model/Data/Train'
valid_path = '/home/reshma/model/Data/Test'

model = Sequential()
model.add(Convolution2D(16, 5, 5, activation='relu', input_shape=(28,28, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Convolution2D(32, 5, 5, activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))

model.add(Dense(5, activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(28,28),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_path,
        target_size=(28,28),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=78,
        nb_epoch=3,
        validation_data=validation_generator,
        nb_val_samples=49)

model.save_weights('/home/reshma/model/Data/modelminuscomma.h5')