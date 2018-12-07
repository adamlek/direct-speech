from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, TimeDistributed, Activation, Dense

def create_model(vocabulary):
    vocabulary = len(vocabulary)
    hidden_size = 100
    model = Sequential()
    # create word embeddings for each input word, size of each embedding vector = hidden_size
    model.add(Embedding(vocabulary, hidden_size))
    # LSTM: size = hidden_size
    model.add(LSTM(100, return_sequences=True, stateful=False))
    model.add(LSTM(100, return_sequences=True, stateful=False))
    #model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(0.3))
    # dense adds Dense=x output value(s) to each input word
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    #, activation='sigmoid'model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    print(model.summary())
    
    return model