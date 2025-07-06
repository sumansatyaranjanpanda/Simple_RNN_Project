# 1. Import Libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# 2. Load Word Index and Reverse Map
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

# 3. Load Pretrained Model
model = load_model('simple_rnn_imdb.h5')

# 4. Decode Review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# 5. Preprocess Text
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# 6. Predict Sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# 7. Streamlit App
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Type a movie review below and check whether it is Positive or Negative.')

user_input = st.text_area('✍️ Movie Review')

if st.button('🔍 Classify'):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        st.success(f'**Sentiment:** {sentiment}')
        st.info(f'**Prediction Score:** {score:.4f}')
    else:
        st.warning('Please enter a valid movie review!')
