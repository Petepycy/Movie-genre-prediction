import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

# Load the saved model
model = load_model('overview_genre_model.keras')

# Load the preprocessing information
word_index = np.load('word_index.npy', allow_pickle='TRUE').item()
mlb_classes = np.load('mlb.classes_.npy')

# Your test data (assume 'test_data' is your input data for testing)
test_data = ['as the war of panem escalates to the destruction of other districts katniss everdeen the reluctant leader of the rebellion must bring together an army against president snow while all she holds dear hangs in the balance',
             'the story of chesley sullenberger an american pilot who became a hero after landing his damaged plane on the hudson river in order to save the flights passengers and crew',
             'when the newly crowned queen elsa accidentally uses her power to turn things into ice to curse her home in infinite winter her sister anna teams up with a mountain man his playful reindeer and a snowman to change the weather condition'
             ]
# True Genres: ('Action', 'Adventure', 'Crime')
# True Genres: ('Biography', 'Drama')
# True Genres: ('Adventure', 'Animation', 'Comedy')
# Preprocess the test text
tokenizer = Tokenizer()
tokenizer.word_index = word_index
sequences = tokenizer.texts_to_sequences(test_data)
padded_sequences = pad_sequences(sequences, maxlen=300)

# Make predictions
predictions = model.predict(padded_sequences)

# Convert predictions to binary classes based on a threshold (e.g., 0.5)
threshold = 0.5
predictions_classes = (predictions > threshold).astype(int)

# Inverse transform the predicted classes to genre labels
predicted_genres = [mlb_classes[idxs] for idxs in predictions_classes]

# Print or use the predicted_genres as needed
print(predicted_genres)
