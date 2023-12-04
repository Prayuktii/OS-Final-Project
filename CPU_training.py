import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import time
import matplotlib.pyplot as plt

# Printing out all the compute devices visible to TensorFlow.

print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

# Import the MNIST dataset of 60,000 28x28 pixel handwritten digit images along with corresponding digit label (0-9) for each image.
# This dataset is divided into a training set of 50,000 images and a test set of 10,000 images.

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# Displaying the actual sample images from the MNIST dataset.

num_samples_to_display = 5
plt.figure(figsize=(10, 4))
for i in range(num_samples_to_display):
    plt.subplot(1, num_samples_to_display, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"Label:{train_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Defining a complex model with deeper layers and dropout.

def create_complex_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Defining and training models on CPU.

def train_model_on_device(model, train_images, train_labels, test_images, test_labels, device):
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print(f"Using {device} for computations.")
    start_time = time.time()
    with tf.device(device):
        history = model.fit(train_images, train_labels,
                            epochs=10, batch_size=128,
                            validation_data=(test_images, test_labels))
    end_time = time.time()

    return history, end_time - start_time


# CPU Training

with tf.device('/CPU:0'):
    cpu_model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    cpu_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    start_time_cpu = time.time()
    cpu_history = cpu_model.fit(train_images, train_labels,
                                epochs=10, batch_size=128,
                                validation_data=(test_images, test_labels))
    end_time_cpu = time.time()

    cpu_time = end_time_cpu - start_time_cpu

# Plotting the CPU training time.

x = ['CPU Time']
y = [cpu_time]

plt.figure(figsize=(6, 4))
plt.bar(x, y, color=['orange', 'blue'])
plt.ylabel("Time (sec)")
plt.title("CPU Training Time")
plt.show()

# Visualizing the training history trend for accuracy and loss in case CPU Model.

def plot_training_history(history, title):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(title + ' - Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title + ' - Loss')

    plt.tight_layout()
    plt.show()


plot_training_history(cpu_history, 'CPU Model')