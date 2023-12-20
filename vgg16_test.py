# 1. I want like to train a custom DLNN based on the VGG16 network architecture.
# 2. The custom DLNN needs to classify my dataset into three categories according on the names of the subfolders in the main dataset folder.
# 3. The dataset consists of images. Use a temporary placeholder for the folder path. I will modify it later.
# 4. I want to divide the dataset to be randomized and divided according to 70:30 ratio (training:validation). Do some common preprocessing suitable for image data.
# 5. I want to use the ADAM algorithm to train the custom DLNN. I also want to perform transfer learning, please download and update the weights based on a completed model.
# 6. After training, I want the custom DLNN to generate 2 confusion matrices for each of the training, and validation dataset. These plots should be displayed in 2 different figures.
# 7. I also want to view the training plot of the custom DLNN. This plot should be displayed in a different figure from step 7.
# Please write the complete code using Python. You may ask questions to clarify your task.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools

# Placeholder for the dataset path
dataset_path = 'C:\\Users\\ihsan\\Downloads\\batik_dataset'

# Load and preprocess data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3  # for validation split
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # set as training data
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # set as validation data
)

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(3, activation='softmax')(x)  # Assuming 3 categories

# Define the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Generate confusion matrices
# Predictions for training set
train_predictions = model.predict(train_generator)
train_cm = confusion_matrix(train_generator.classes, np.argmax(train_predictions, axis=1))
plt.figure(figsize=(10, 10))
plot_confusion_matrix(train_cm, train_generator.class_indices.keys(), title='Confusion Matrix - Training Set')

# Predictions for validation set
validation_predictions = model.predict(validation_generator)
validation_cm = confusion_matrix(validation_generator.classes, np.argmax(validation_predictions, axis=1))
plt.figure(figsize=(10, 10))
plot_confusion_matrix(validation_cm, validation_generator.class_indices.keys(), title='Confusion Matrix - Validation Set')

# Plot training history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()


