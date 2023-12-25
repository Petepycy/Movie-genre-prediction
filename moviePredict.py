import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def create_embedding_matrix(embedding_file_path, word_index, embedding_dim=100):
    embeddings_index = {}
    with open(embedding_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


# Load the dataset
data = pd.read_csv('IMDB-Movie-Data.csv')  # Replace 'your_dataset.csv' with the actual filename

# Preprocess the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Description'])
sequences = tokenizer.texts_to_sequences(data['Description'])
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences, maxlen=300)

# Encode genres
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(data['Genre'].apply(lambda x: x.split(',')))

# Load pre-trained word embeddings (e.g., GloVe)
# Create embedding matrix
embedding_matrix = create_embedding_matrix('glove.6B.100d.txt', word_index)

# Define the LSTM model
model = Sequential()
model.add(Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=300, trainable=False))
model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))  # Increase units and adjust dropout
model.add(Dense(len(mlb.classes_), activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, genres_encoded, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest Accuracy: {test_accuracy}')

# Make predictions on a few examples from the test set
num_examples = 5
sample_indices = np.random.choice(len(X_test), num_examples, replace=False)
sample_X = X_test[sample_indices]
sample_y_true = y_test[sample_indices]

# Predict genres for the sample examples
sample_y_pred = model.predict(sample_X)
threshold = 0.3  # Experiment with different values

# Convert predictions to binary classes based on a threshold (e.g., 0.5)
sample_y_pred_classes = (sample_y_pred > threshold).astype(int)

# Print the results for each example
for i in range(num_examples):
    print(f'\nExample {i + 1}:')
    print(f'Input Text: {tokenizer.sequences_to_texts([sample_X[i]])[0]}')

    # Convert the true and predicted values to NumPy arrays
    true_genres_np = np.array([sample_y_true[i]])
    pred_genres_np = np.array([sample_y_pred_classes[i]])

    # Use inverse_transform with NumPy arrays
    print(f'True Genres: {mlb.inverse_transform(true_genres_np)[0]}')
    print(f'Predicted Genres: {mlb.inverse_transform(pred_genres_np)[0]}')

# Save the model for future use
model.save('my_model.keras')