import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# Pull down our raw data
data = tf.keras.datasets.mnist

# Test train split (6:1 ratio)
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Scale the greyscale to [0,1]
train_images = train_images / 255.0
test_images  = test_images  / 255.0

# Build the model
model = tf.keras.Sequential([
    # Input layer must be 28*28. Since we are passing in 
    # a matrix essentially, the flatten function will 
    # smooth it out to a single layer. Note, input layer is 784 nodes
    tf.keras.layers.Flatten(input_shape=(28,28)),

    # One hidden layer with 128 nodes. We will experiment with more less nodes
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),

    # One output layer (there can only be 10 options (0-9))
    tf.keras.layers.Dense(10)
])

# Compile the model with an optimizer (some form of gradient decent) (we may want to fiddle around with this in the future)
# a loss function
# and a metric we want to measure, in this case, accuracy
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Feed the model the data.
# Epochs is the amount of times the model goes over all the data
model.fit(train_images, train_labels, epochs=10)

# Evaluate the models preformance on our test data
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('\nTest accuracy:', test_acc)