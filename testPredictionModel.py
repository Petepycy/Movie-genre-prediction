# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from predictor import preprocess_text, tokenizer, mlb
#
# # Load the model
# model = load_model('movie_genre_prediction_model.h5')
#
# # Example: preprocess new description
# new_description = preprocess_text("A group of friends embark on an epic journey.")
# new_sequence = tokenizer.texts_to_sequences([new_description])
# new_padded_sequence = pad_sequences(new_sequence, maxlen=300)
#
# # Make predictions
# new_pred = model.predict(new_padded_sequence)
# new_pred_classes = (new_pred > 0.5).astype(int)
#
# # Decode predictions
# predicted_genres = mlb.classes_[new_pred_classes.flatten() == 1]
#
# print("Predicted Genres:", predicted_genres)
