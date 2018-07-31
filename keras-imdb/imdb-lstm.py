from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten
from keras.datasets import imdb
import wandb
from wandb.keras import WandbCallback
import imdb
import numpy as np
from keras.preprocessing import text

wandb.init()
config = wandb.config

# set parameters:
config.vocab_size = 1000
config.maxlen = 300
config.batch_size = 32
config.embedding_dims = 16
config.filters = 16
config.kernel_size = 3
config.hidden_dims = 100
config.epochs = 10

(X_train, y_train), (X_test, y_test) = imdb.load_imdb()

tokenizer = text.Tokenizer(num_words=config.vocab_size)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)

#state of the art text processing arch
model = Sequential()
model.add(Embedding(config.vocab_size,
                    config.embedding_dims,
                    input_length=config.maxlen))
#This converts the lstm into lstm-cnn hybrid.
#Still under research. 
#Intuition: cnn can take care of the local patterns while
#lstm can analyze the the long term patterns
model.add(Conv1D(config.filters,
                 config.kernel_size,
                 padding='valid',
                 activation='relu'))
model.add(LSTM(config.hidden_dims, activation="sigmoid"))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=config.batch_size,
          epochs=config.epochs,
          validation_data=(X_test, y_test), callbacks=[WandbCallback()])
