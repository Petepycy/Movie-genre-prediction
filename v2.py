import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import RMSprop

def lr_scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch > 0:
        return lr * 0.9  # Reduce learning rate every 5 epochs
    return lr

# Load the dataset
data = pd.read_csv('movies_metadata.csv')

# Preprocess the text
data['overview'] = data['overview'].astype(str)
data['overview'] = data['overview'].fillna('')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['overview'])
sequences = tokenizer.texts_to_sequences(data['overview'])
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences, maxlen=400)

# Convert 'genres' column to list of dictionaries
data['genres'] = data['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Extract unique genres
all_genres = [genre['name'] for genres_list in data['genres'] for genre in genres_list]
unique_genres = list(set(all_genres))

# Encode genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=unique_genres)
genres_encoded = mlb.fit_transform(data['genres'])

# Define the LSTM model
# model = Sequential()
# model.add(Embedding(len(word_index) + 1, 128, input_length=400))
# model.add(SpatialDropout1D(0.3))
# model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
# model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))
# model.add(Dense(len(mlb.classes_), activation='sigmoid'))
# model.add(BatchNormalization())  # Add Batch Normalization for stability
model = Sequential()
model.add(Embedding(len(word_index) + 1, 128, input_length=400))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(len(mlb.classes_), activation='sigmoid'))
model.add(BatchNormalization())
# Compile the model
optimizer = RMSprop(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_schedule = LearningRateScheduler(lr_scheduler)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, genres_encoded, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_schedule])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest Accuracy: {test_accuracy}')

# Save the model and preprocessing information
model.save('overview_genre_model_updated_v2.keras')
np.save('word_index_updated_v2.npy', word_index)
np.save('mlb.classes_updated_v2.npy', mlb.classes_)
