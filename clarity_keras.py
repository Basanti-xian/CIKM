'''This example demonstrates the use of Convolution1D for text classification.
Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.layers import Embedding
from keras.layers import Conv2D, GlobalMaxPooling2D
from keras.layers.pooling import MaxPooling2D
from keras.datasets import imdb
import numpy as np
from Lazada import *
from sklearn.feature_extraction.text import CountVectorizer
import re

# set parameters:
max_features = 100
maxlen = 50
batch_size = 32
embedding_dims = 256
filters = 25
kernel_size = 5
hidden_dims = 256
epochs = 2


# Load data
print("Loading data...")
#x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text = read_data('../data/training/data_train.csv')
y_labels = read_data('../data/training/clarity_train.labels')
x_validation = read_data('../data/validation/data_valid.csv')
print ('loaded training data and labels')
x_text=get_titles(x_text)
x_validation = get_titles(x_validation)

vectorizer = CountVectorizer(min_df=0, lowercase=True)
vectorizer.fit(x_text)
vocab = vectorizer.vocabulary_
vocab_len= len(vocab)


x = []
for sen in x_text:
    vect = []
    len = []
    for str in sen.split(" "):
        str = re.sub('[^A-Za-z0-9]+', '', str).lower()
        vect.append(vocab.get(str, vocab_len+1))
    x.append(vect)


x_train, x_test, y_train, y_test = train_test_split(x, y_labels, test_size = 0.10,
                                                            random_state = random.randint(0,100))

validation = []
for sen in x_validation:
    vect = []
    for str in sen.split(" "):
        str = re.sub('[^A-Za-z0-9]+', '', str).lower()
        vect.append(vocab.get(str, vocab_len+1))
    validation.append(vect)


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
validation = sequence.pad_sequences(validation, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('validation shape:', validation.shape)


print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(vocab_len+1,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.1))

model.add(Reshape((50,16,16)))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv2D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=(1,1)))
# we use max pooling:
model.add(MaxPooling2D())

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv2D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=(1,1)))
# we use max pooling:
model.add(GlobalMaxPooling2D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
validation_data=(x_test, y_test))

vpreds = model.predict(validation)
pred = model.predict(x_test)

for p in pred:
    print (p)

pred_fn = 'clarity_valid.predict'
write_predictions(pred_fn, vpreds)