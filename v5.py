import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import classification_report, roc_auc_score

def load_glove_embeddings(glove_path, word_index, embedding_dim=200):  # Increase embedding_dim for more complex relationships
    embedding_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def lr_scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch > 0:
        return lr * 0.9  # Reduce learning rate every 5 epochs
    return lr

def preprocess_data(train_data, test_data):
    tokenizer = Tokenizer()

    # Convert non-string values to empty strings
    train_data = train_data.apply(lambda x: str(x) if isinstance(x, (str, float)) else '')
    test_data = test_data.apply(lambda x: str(x) if isinstance(x, (str, float)) else '')

    tokenizer.fit_on_texts(train_data)
    sequences_train = tokenizer.texts_to_sequences(train_data)
    sequences_test = tokenizer.texts_to_sequences(test_data)
    word_index = tokenizer.word_index
    padded_sequences_train = pad_sequences(sequences_train, maxlen=400)
    padded_sequences_test = pad_sequences(sequences_test, maxlen=400)
    return word_index, padded_sequences_train, padded_sequences_test

# Load the dataset
data = pd.read_csv('movies_metadata.csv', low_memory=False)

# Extract unique genres
all_genres = data['genres'].apply(lambda x: eval(x) if isinstance(x, str) else [])
all_genres = all_genres.apply(lambda x: [genre['name'] for genre in x] if isinstance(x, list) else [])

# Encode genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(all_genres)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['overview'], genres_encoded, test_size=0.2, random_state=42)

# Preprocess the text
word_index, padded_sequences_train, padded_sequences_test = preprocess_data(X_train, X_test)

# Load GloVe embeddings
glove_path = 'glove.6B.200d.txt'  # Use a higher-dimensional GloVe embedding
embedding_dim = 200
embedding_matrix = load_glove_embeddings(glove_path, word_index, embedding_dim)

# Define the LSTM model
model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=400, trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)))
model.add(Dense(len(mlb.classes_), activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Compile the model
optimizer = RMSprop(learning_rate=0.0001)  # Fine-tune learning rate
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)  # Patience increased for more training
lr_schedule = LearningRateScheduler(lr_scheduler)

# Train the model
model.fit(padded_sequences_train, y_train, batch_size=128, epochs=50, validation_data=(padded_sequences_test, y_test), callbacks=[early_stopping, lr_schedule])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(padded_sequences_test, y_test, verbose=2)
print(f'\nTest Accuracy: {test_accuracy}')

# Additional metrics
predictions = model.predict(padded_sequences_test)
roc_auc = roc_auc_score(y_test, predictions)
print(f'ROC AUC: {roc_auc}')

classification_metrics = classification_report(y_test, predictions > 0.5, target_names=mlb.classes_)
print(classification_metrics)

# Save the model and preprocessing information
model.save('overview_model_updated_v5.keras')
np.save('word_index_overview.npy', word_index)
np.save('mlb.classes_overview.npy', mlb.classes_)
