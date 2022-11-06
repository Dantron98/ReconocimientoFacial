import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Activation, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop, SGD
import os
import pathlib
np.set_printoptions(precision=4)
#with open('data/list_attr_celeba.csv', 'r') as f:
#    print('skipping : ' + f.readline())
#    print('skipping headers : ' + f.readline())
#    with open('data/list_attr_celeba.csv', 'w') as newf:
#        for line in f:
#            new_line = ' '.join(line.split())
#            newf.write(new_line)
#            newf.write('\n')

df = pd.read_csv('data/list_attr_celeba.csv', sep=',', header=None)
df = df[1:]

#df.iloc[:, 1:].replace(to_replace=-1, value=0)
#print(df.head())
#exit()

#print(df.head())
#print('-----------')
#print(df.iloc[:, 1:])
#exit()


x = np.asarray(df.iloc[:, 1:]).astype('int64')
df.replace(to_replace=-1, value=0, inplace=True)
df.replace(to_replace='-1', value=0, inplace=True)
df.replace(to_replace='1', value=1, inplace=True)

files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(x)
data = tf.data.Dataset.zip((files, attributes))

path_to_images = 'data/img_align_celeba/'

def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)

num_class = 39
epochs = 30
batch_size = 50
num_train = 150000
num_test = len(df) - num_train
epochs_step = num_train // batch_size
test_step = num_test // batch_size
data_train = labeled_images.take(num_train)
data_test = labeled_images.skip(num_train)


print(data_train)
print(data_test)
#a = np.array([1]).astype('int64')
#b = tf.constant(a)
a = tf.data.Dataset.range(1)
#print(b)
data_train = a.concatenate(data_train)
#data_train = tf.concat([b, data_train], 0)
print(data_train)


#data_train = tf.expand_dims(data_train, 0)
#data_test = data_test[None, :, :, :]
#data_train = tf.reshape(data_train, [None, 192, 192, 3])
#tf.data.AUTOTUNE()

#model = Sequential()
inputs = keras.Input(shape=(192, 192, 3), name='input')
#model.add(Conv2D(10, (3, 3), input_shape=(192, 192, 3)))
x = tf.keras.layers.Conv2D(10, (3, 3))(inputs)
#model.add(Activation('relu'))
x = tf.keras.layers.Activation('relu')(x)
#model.add(MaxPooling2D(pool_size=(2, 2)))
x = tf.keras.layers.MaxPooling2D(2)(x)
#model.add(Conv2D(10, (3, 3)))
x = tf.keras.layers.Conv2D(10, (3, 3))(x)
#model.add(Activation('relu'))
x = tf.keras.layers.Activation('relu')(x)
#model.add(MaxPooling2D(pool_size=(2, 2)))
x = tf.keras.layers.MaxPooling2D(2)(x)
#model.add(Conv2D(20, (3, 3)))
x = tf.keras.layers.Conv2D(10, (3, 3))(x)
#model.add(Activation('relu'))
x = tf.keras.layers.Activation('relu')(x)
#model.add(MaxPooling2D(pool_size=(2, 2)))
x = tf.keras.layers.MaxPooling2D(2)(x)
#model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
x = tf.keras.layers.Dropout(0.2)(x)
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(40, activation='sigmoid')(x)
model = tf.keras.Model(inputs= inputs, outputs=output)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(
    data_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=data_test
)


