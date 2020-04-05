import numpy as np

np.random.seed(123)  # pasetint kad atsikartotu
import os; os.environ['KERAS_BACKEND'] = 'theano' # neveike norejo visada tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) #pakeisti duomenu forma, buvo 1,28,28 neveike
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# 6. Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print Y_train.shape
# 7. Define model architecture
model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1))) #update from model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
print model.input_shape
print model.output_shape
model.add(Conv2D(8, (5, 5), activation='relu'))
print model.output_shape
model.add(MaxPooling2D(pool_size=(2, 2)))
print model.output_shape
model.add(Dropout(0.4))
print model.output_shape
model.add(Flatten())
print model.output_shape
model.add(Dense(128, activation='relu'))
print model.output_shape
model.add(Dropout(0.75))
print model.output_shape
model.add(Dense(10, activation='softmax'))
print model.output_shape
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# 9. Fit model on training data
model.fit(X_train, Y_train, batch_size=32, epochs=1, verbose=1)
# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
print model.output_shape


model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

import imageinput
while 1:
    x = np.asarray(
        imageinput.imageprepare("C:\\source\\ocr" + '\\' + raw_input("name of image") + ".png")).reshape((-1,28,28,1))
    a = model.predict_classes(x)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    print(a)
    print(model.predict(x))


