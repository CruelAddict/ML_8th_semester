from mnist import MNIST
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mndata = MNIST(f'samples')
categories_amount = 10
input_shape = (28, 28, 1)
batch_size = 50
num_epoch = 12

x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()
x_test_orig = x_test.copy()

x_train = np.reshape(x_train, (len(x_train),) + input_shape) / 255
x_test = np.reshape(x_test, (len(x_test),) + input_shape) / 255

y_train = tf.keras.utils.to_categorical(y_train, categories_amount)
y_test = tf.keras.utils.to_categorical(y_test, categories_amount)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(
    filters=150,
    kernel_size=3,
    strides=1,
    padding='valid',
    activation='relu',
    input_shape=input_shape
))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(
    filters=100,
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    input_shape=(13, 13, 50)
))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv2D(
    filters=50,
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu',
    input_shape=(6, 6, 50)
))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(categories_amount, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

model_log = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=num_epoch,
                      verbose=1,
                      validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print(score[0])
print(score[1])

predictions = model.predict_proba(x_test)

print(tf.math.confusion_matrix(
    np.argmax(y_test, axis=1), np.argmax(predictions, axis=1), num_classes=10
))

pictures_matrix = []
for i_class in range(categories_amount):
    pictures_matrix.append([])

    # Saving the original index in the data set to restore the image
    items_of_class_i = [(index, obj_predictions) for index, obj_predictions in enumerate(predictions) if
                        np.argmax(obj_predictions) == i_class]
    for j_class in range(categories_amount):

        #  List of model predictions for objects of class i that they are of class j
        i_is_j_predictions = [obj_predictions[1][j_class] for obj_predictions in items_of_class_i]

        most_confusing_prediction = max(
            [prediction for idx, prediction in enumerate(i_is_j_predictions)])

        pictures_matrix[i_class].append(
            items_of_class_i[  # Array that stores the original indexes
                i_is_j_predictions.index(most_confusing_prediction)  # Element index in i_is_j_pr == index in i_of_c_i
            ][0])
print(pictures_matrix)

fig, plts = plt.subplots(categories_amount, categories_amount, num='Confusing images')
for i in range(categories_amount):
    for j in range(categories_amount):
        image = np.array(x_test_orig[pictures_matrix[i][j]], dtype='float')
        pixels = image.reshape((28, 28))
        plts[i][j].imshow(pixels, cmap='gray')
        plts[i][j].axis('off')
plt.show()

# model.summary()
