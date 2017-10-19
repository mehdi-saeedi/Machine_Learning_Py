# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers
from sklearn import preprocessing
from keras.optimizers import Adam

import Augmentor

# fix random seed for reproducibility
np.random.seed(7)

directory = '../../Datasets/Digital_Recognizer/'
train_input = pd.read_csv(directory + 'train.csv')

train_target = train_input['label']
train_input.drop(['label'], axis=1, inplace=True)

train_input = train_input.astype('float32')

train_input = train_input / 255.
train_target = to_categorical(train_target, 10)

print(train_target.shape)

train_input = train_input.values

batch_size = 256

epochs = 100
dropout = 0.05
num_classes = 10

X_train, X_cv, y_train, y_cv = train_test_split(train_input, train_target, test_size=0.20, random_state=0)

X_train = X_train.reshape(-1, 28, 28, 1)
X_cv = X_cv.reshape(-1, 28, 28, 1)
input_shape = (28, 28, 1)

conv_model = Sequential()

conv_model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))
conv_model.add(BatchNormalization())
conv_model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
conv_model.add(BatchNormalization())
conv_model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
conv_model.add(BatchNormalization())
conv_model.add(MaxPooling2D(strides=(2, 2)))
conv_model.add(Dropout(dropout))

conv_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
conv_model.add(BatchNormalization())
conv_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
conv_model.add(BatchNormalization())
conv_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
conv_model.add(BatchNormalization())
conv_model.add(MaxPooling2D(strides=(2, 2)))
conv_model.add(Dropout(dropout))

conv_model.add(Flatten())
conv_model.add(Dense(512, activation='relu'))
conv_model.add(Dropout(dropout))
conv_model.add(Dense(1024, activation='relu'))
conv_model.add(Dropout(dropout))
conv_model.add(Dense(10, activation='softmax'))

# print summary of the conv_model
conv_model.summary()

datagen = ImageDataGenerator(zoom_range = 0.1,
                            height_shift_range = 0.1,
                            width_shift_range = 0.1,
                            rotation_range = 10)

# Compile conv_model
conv_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy']) #RMSprop()


# hist = conv_model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
#                            steps_per_epoch=len(X_train)//batch_size,
#                            epochs=epochs,
#                            verbose=2,  #1 for ETA, 0 for silent
#                            validation_data=(X_cv,y_cv))
#
# # hist = conv_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=False,
# #           validation_data=(X_cv, y_cv))
#
# # evaluate the conv_model
# scores = conv_model.evaluate(X_train, y_train)
# print("\n%s: %.2f%%" % (conv_model.metrics_names[1], scores[1] * 100))
#
# # evaluate the conv_model
# scores = conv_model.evaluate(X_cv, y_cv)
# print("\n%s: %.2f%%" % (conv_model.metrics_names[1], scores[1] * 100))
#
# # check the wrong images
# p_cv = np.round(conv_model.predict(X_cv)).argmax(axis=1)
# wrong_pixels = X_cv[p_cv != y_cv.argmax(axis=1)]
# wrong_y = conv_model.predict(wrong_pixels)
# print('[CV]: number of wrong items is:', len(wrong_pixels), 'out of', len(X_cv))

# evaluate test data
test_input = pd.read_csv(directory + 'test.csv')
test_input = test_input.astype('float32')
test_input = test_input.values
test_input = test_input / 255.

test_input = test_input.reshape(-1, 28, 28, 1)

# p_test = np.round(conv_model.predict(test_input)).argmax(axis=1)
# # write to a file
# out_df = pd.DataFrame(
#     {'ImageId': np.arange(1, test_input.shape[0] + 1), 'Label': p_test}).to_csv(
#     'out.csv', header=True, index=False)
#
# #visually check 100 wrong cases
# f, axarr = plt.subplots(10, 20)
# for i in range(0, 10):
#     for j in range(0, 10):
#         idx = np.random.randint(0, wrong_pixels.shape[0])
#         axarr[i][j].imshow(wrong_pixels[idx, :].reshape(28, 28), cmap=cm.Greys_r)
#         tit = str(wrong_y[idx, :].argmax())
#         axarr[i][j + 10].text(0.5, 0.5, tit)
#         axarr[i][j].axis('off')
#         axarr[i][j + 10].axis('off')

f, axarr = plt.subplots(10, 10)
for i in range(0, 10):
    for j in range(0, 10):
        axarr[i][j].imshow(test_input[i*10+j, :].reshape(28, 28), cmap=cm.Greys_r)
        axarr[i][j].axis('off')

plt.show()
