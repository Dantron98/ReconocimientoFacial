import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime

np.set_printoptions(precision=4)
df = pd.read_csv('data/list_attr_celeba.csv', sep=',', header=None)
df = df[1:]
df.replace(to_replace=-1, value=0, inplace=True)
df.replace(to_replace='0', value=0, inplace=True)
df.replace(to_replace='-1', value=0, inplace=True)
df.replace(to_replace='1', value=1, inplace=True)
x = np.asarray(df.iloc[:, 1:]).astype('int64')

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
batch_size = 50
labeled_images = data.map(process_file).batch(batch_size)

num_class = 39
epochs = 10

num_train = 150000
num_test = len(df) - num_train
epochs_step = num_train // batch_size
test_step = num_test // batch_size
data_train = labeled_images.take(num_train)
data_test = labeled_images.skip(num_train)

inputs = keras.Input(shape=(192, 192, 3), name='input')
x = tf.keras.layers.Conv2D(10, (3, 3))(inputs)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(10, (3, 3))(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(10, (3, 3))(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(40, activation='sigmoid')(x)
model = tf.keras.Model(inputs=inputs, outputs=output)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

model.summary()
model.compile(optimizer=tf.optimizers.RMSprop(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

model.fit(data_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=data_test,
    callbacks=[tbCallBack])

model.save('test1.h5')