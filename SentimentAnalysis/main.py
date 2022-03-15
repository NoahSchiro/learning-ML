import pandas as pd
import numpy as np
import tensorflow as tf
import re
import string

# Grab the data
data = pd.read_csv("data.csv")

# Clean the data
def clean_data(input):

	# Lowercase only
    lowered = input.lower()

    # Replace non-alphabetical characters with space
    removed = re.sub(r'[^a-zA-Z ]+','',lowered)  # replacing the non alphabets with space

    return(removed) # returning the cleaned words

# Apply the data cleaning to our inputs
data['Sentence'] = data['Sentence'].apply(clean_data)

# Save all the possible vocabulary
corpus = ' '.join(data['Sentence']) 			# As a string
corpus_list = corpus.split()					# As a list

# Measure the number of unique words
vocab_size = len(set(corpus_list))

# END OF PREPROCESSING
# STARTING THE MACHINE LEARNING

# This will convert strings to an integer representation
vector_layer = tf.keras.layers.TextVectorization(
			    max_tokens   = vocab_size, 
			    split        = 'whitespace',
			    output_mode  = 'tf_idf',
)

# This essentially does the mapping from string -> int
vector_layer.adapt(data['Sentence'])

# Create a simple model
inputs = tf.keras.Input(shape=(vector_layer.vocabulary_size(),))
outputs = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.Model(inputs, outputs)


model = tf.keras.Model([

	# Input is the size of the vocab
	tf.keras.Input(shape=(vector_layer.vocabulary_size(),)),

	# There are three possible outputs
	tf.keras.layers.Dense(3)
])

# Give it a basic optimizer, loss function, and measure accuracy
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Feed the model the data.
# Epochs is the amount of times the model goes over all the data
model.fit(
	data["Sentence"],
	data["Sentiment"],
	epochs = 6
)