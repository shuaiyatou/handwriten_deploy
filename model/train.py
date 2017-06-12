from __future__ import print_function
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical, plot_model
import keras.backend as K

# 导入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_row, img_col = 28, 28
batch_size = 128
num_classes = 10
nb_epochs = 12

# 调整数据的shape
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_row, img_col)
    x_test = x_test.reshape(x_test.shape[0], 1, img_row, img_col)
    input_shape = (1, img_row, img_col)
else:
    x_train = x_train.reshape(x_train.shape[0], img_row, img_col, 1)
    x_test = x_test.reshape(x_test.shape[0], img_row, img_col, 1)
    input_shape = (img_row, img_col, 1)

# 对input做一些调整，有助于训练
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# one hot标签
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 搭建模型
model = Sequential()
# (-1, 28, 28, 1) -> (-1, 26, 26, 32)
model.add(Conv2D(filters=32, kernel_size=(3, 3),
                 strides=(1, 1),
                 activation='elu',
                 input_shape=input_shape))
# (-1, 26, 26, 32) -> (-1, 24, 24, 64)
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 strides=(1, 1),
                 activation='elu', ))
# (-1, 24, 24, 64) -> (-1, 12, 12, 64)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# (-1, 21, 21, 64) -> (-1, 9216)
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 将模型结构保存图片
plot_model(model, to_file='model.png', show_shapes=True)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=nb_epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 保存模型
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')
print('Saved mdoel to disk!')
