#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras as K
import matplotlib.pyplot as plt

def preprocess_data(X, Y):
    resized_images = []
    for image in X:
        resized_image = tf.image.resize(image, [224, 224])
        resized_images.append(resized_image.numpy())
    return np.array(resized_images), Y

(X_train, y_train), (X_test, y_test) = K.datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train = K.utils.to_categorical(y_train, 10)
y_test = K.utils.to_categorical(y_test, 10)
X_train_resized, y_train_resized = preprocess_data(X_train, y_train)
X_test_resized, y_test_resized = preprocess_data(X_test, y_test)

base_model = K.applications.resnet50.ResNet50(weights='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = K.layers.GlobalAveragePooling2D()(x)
x = K.layers.Dense(1024, activation='relu')(x)
predictions = K.layers.Dense(10, activation='softmax')(x)

model = K.models.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=K.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_resized, y_train, batch_size=32, epochs=10, validation_data=(X_test_resized, y_test))

for layer in base_model.layers[-10:]:
    layer.trainable = True

model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(X_train_resized, y_train, batch_size=32, epochs=10, validation_data=(X_test_resized, y_test))

history.history['loss'].extend(history_fine.history['loss'])
history.history['val_loss'].extend(history_fine.history['val_loss'])
history.history['accuracy'].extend(history_fine.history['accuracy'])
history.history['val_accuracy'].extend(history_fine.history['val_accuracy'])

# Plot the training and validation loss
plt.figure(figsize=(12, 8))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.figure(figsize=(12, 8))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

