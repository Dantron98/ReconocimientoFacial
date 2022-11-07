import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime


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
print(df.head())
#exit()
#df.iloc[:, 1:].replace(to_replace=-1, value=0)
#print(df.head())
#exit()

#print(df.head())
#print('-----------')
#print(df.iloc[:, 1:])
#exit()



df.replace(to_replace=-1, value=0, inplace=True)
df.replace(to_replace='0', value=0, inplace=True)
df.replace(to_replace='-1', value=0, inplace=True)
df.replace(to_replace='1', value=1, inplace=True)
x = np.asarray(df.iloc[:, 1:]).astype('int64')
print(df.head())

files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(x)
data = tf.data.Dataset.zip((files, attributes))
print(attributes)
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
#AUTOTUNE = tf.data.AUTOTUNE

#def configure_for_performance(ds):
#    ds = ds.shuffle(buffer_size=1000)
#    ds = ds.batch(batch_size)
#    ds = ds.prefetch(buffer_size=AUTOTUNE)
#    return ds

#print(data_train)
#print(data_test)
#a = np.array([1]).astype('int64')
#b = tf.constant(a, dtype=tf.float32)
#c = tf.data.Dataset.range(1)

#print('este es c:')
#print(c)
#tf.cast(a, tf.float32)
#print(a)
#print(b)
#exit()
#data_train = b.concatenate(data_train)
#data_train = tf.concat([b, data_train], 0)
#print(data_train)
#for images, labels in data_train.take(1):
#    X_train = images.numpy()
#    y_train = labels.numpy()
#    print('antes del performance')
#    print(y_train.shape)
#    print(len(y_train))
#    print(y_train)

#data_train = configure_for_performance(data_train)
#data_test = configure_for_performance(data_test)
#for images, labels in data_train.take(1):
#    X_train = images.numpy()
#    y_train = labels.numpy()
#    print('-------------')
#    print(y_train.shape)
#    print(len(y_train))
#    print(y_train)

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


