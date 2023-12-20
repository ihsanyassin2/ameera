import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Placeholder for dataset path
dataset_path = 'E:\\Ameera\\Spectrogram\\CNN 600 data\\AlexNet\\NonFiltered\\Dot\\Fz\\128'

# Preprocess data
def preprocess_data(dataset):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    return dataset.map(lambda x, y: (normalization_layer(x), y))

# Modify the load_dataset function to return both the dataset and its size
def load_dataset(path):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        shuffle=True,
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=32
    )
    size = dataset.cardinality().numpy()
    return dataset, size

data, dataset_size = load_dataset(dataset_path)
data = preprocess_data(data)

# Calculate the number of batches for training and validation
train_size = int(0.7 * dataset_size)
val_size = dataset_size - train_size

# Split dataset
train_data = data.take(train_size)
val_data = data.skip(train_size)

# Define AlexNet-based model
def alexnet_model(num_classes=3):
    model = tf.keras.Sequential([
        # First Convolutional Layer
        tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Second Convolutional Layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Remaining Convolutional Layers
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        # Flatten and Fully Connected Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Output Layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


model = alexnet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=val_data, epochs=10)

def generate_confusion_matrix(data, model):
    true_labels = []
    predictions = []
    for x, y in data:
        true_labels.extend(y.numpy())
        preds = model.predict(x)
        predictions.extend(np.argmax(preds, axis=1))
    conf_matrix = confusion_matrix(true_labels, predictions)
    return conf_matrix

train_conf_matrix = generate_confusion_matrix(train_data, model)
val_conf_matrix = generate_confusion_matrix(val_data, model)
plot_training_history(history)


