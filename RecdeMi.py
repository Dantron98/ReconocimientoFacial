import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import datetime
ih, iw = 192, 192
input_shape = (ih, iw, 3)

train_dir = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/train/'
test_dir = '/home/wexon/pycharm/PycharmProjects/ReconocimientoFacial/data/test/'
num_class = 2
bath_size = 50
epochs = 30
num_train = 128 * 2
num_test = 160 * 2 - num_train
epoch_step = num_train // bath_size
test_step = num_test // bath_size

gentrain = ImageDataGenerator(rescale=1./255.)
train = gentrain.flow_from_directory(train_dir,
                                     batch_size=bath_size,
                                     target_size=(iw, ih),
                                     class_mode='binary')
gentest = ImageDataGenerator(rescale=1./255.)
test = gentrain.flow_from_directory(test_dir,
                                     batch_size=bath_size,
                                     target_size=(iw, ih),
                                     class_mode='binary')

base_model = tf.keras.models.load_model('Model_for_test.h5')
count = 0
for layer in base_model.layers:
    layer.trainable = False
    count += 1
print(count)
x = tf.keras.layers.Dropout(0.2)(base_model.output)
outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='serenade_for_strings')(base_model.output)
model = tf.keras.models.Model(base_model.input, outputs)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
model.summary()
model.compile(optimizer=tf.optimizers.RMSprop(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
model.fit(train,
          epochs=epochs,
          steps_per_epoch=epoch_step,
          validation_data=test,
          validation_steps=test_step,
          callbacks=[tbCallBack])
model.save('NN_for_my_face.h5')