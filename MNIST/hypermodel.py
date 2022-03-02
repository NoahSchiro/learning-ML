import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

# Pull down our raw data
data = tf.keras.datasets.mnist

# Test train split (6:1 ratio)
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Function to construct a hypermodel
def build_model(hp):

    # Model to be returned
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Flatten(input_shape=(28, 28)))

    # Add a variable amount of densly connected layers for the hidden layers
    hp_units  = hp.Int('units', min_value=400, max_value=600, step = 4)
    model.add(keras.layers.Dense(units=hp_units, activation = 'relu'))

    hp_units_second  = hp.Int('units', min_value=400, max_value=600, step = 4)
    model.add(keras.layers.Dense(units=hp_units_second, activation = 'relu'))

    # Output layer
    model.add(keras.layers.Dense(10))

    # Build the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3)

# Stop training when a monitored metric has stopped improving.
# Patience equals the number of epochs where we see no improvement
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Search for the best parameters
tuner.search(train_images, train_labels, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters (as determined by objective)
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first 
densly-connected layer is {best_hp.get('units')}.
""")


# Build a fresh model with our parameters
model = tuner.hypermodel.build(best_hp)
# Fit the model to the data
history = model.fit(train_images, train_labels, epochs=50, validation_split=0.2)

# Record the best epoch
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# Create a new model with the same hyper parameters
hypermodel = tuner.hypermodel.build(best_hp)

# Retrain the model with the best number of epochs
hypermodel.fit(train_images, train_labels, epochs=best_epoch, validation_split=0.2)

# Evaulate the preformance of the hypermodel
eval_result = hypermodel.evaluate(test_images, test_labels)
print("[test loss, test accuracy]:", eval_result)