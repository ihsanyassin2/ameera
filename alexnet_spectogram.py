import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
history = model.fit(train_data, validation_data=val_data, epochs=2)

def generate_and_plot_confusion_matrix(data, model, class_names):
    true_labels = []
    predictions = []
    
    # Iterate over the dataset
    for imgs, labels in data:
        # Get the true labels (argmax of one-hot encoded vectors)
        true_labels_batch = np.argmax(labels, axis=1)
        true_labels.extend(true_labels_batch)
        
        # Predict the batch and get the argmax (class with highest probability)
        preds = model.predict(imgs)
        preds = np.argmax(preds, axis=1)
        predictions.extend(preds)
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Plot the confusion matrix
    plt.figure()  # Create a new figure
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return conf_matrix

# You would call this function like this:
class_names = ['High WM', 'Low WM', 'Med WM']  # replace with your actual class names
train_conf_matrix = generate_and_plot_confusion_matrix(train_data, model, class_names)
val_conf_matrix = generate_and_plot_confusion_matrix(val_data, model, class_names)

def plot_training_history(history):
    # Create a new figure for the accuracy plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Create another new figure for the loss plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Show plots
    plt.show()

plot_training_history(history)


