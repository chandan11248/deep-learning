import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="🎭 LSTM Sentiment Analyzer", layout="centered")
st.title("🎭 LSTM Sentiment Analysis (IMDb Reviews)")
st.write("Type any movie review and find out its sentiment instantly!")

# ---------------------------
# 1️⃣ Load Model & Corrected Word Index
# ---------------------------
@st.cache_resource
def load_model_and_word_index():
    model = tf.keras.models.load_model("sentiment_lstm.h5")
    word_index = imdb.get_word_index()
    # shift indices by 3 to match Keras IMDb dataset
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    return model, word_index

model, word_index = load_model_and_word_index()
maxlen = 200

# ---------------------------
# 2️⃣ Preprocessing Function
# ---------------------------
def preprocess_text(text):
    words = text.lower().split()
    seq = [word_index.get(word, 2) for word in words]  # 2 = <UNK>
    padded = pad_sequences([seq], maxlen=maxlen, padding='pre', truncating='pre')
    return padded

# ---------------------------
# 3️⃣ Prediction Function (Flipped)
# ---------------------------
def predict_sentiment(text):
    padded = preprocess_text(text)
    prediction = model.predict(padded, verbose=0)[0][0]
    # FLIP the logic
    if prediction > 0.5:
        return f"😞 Negative Sentiment ({prediction:.2f})"
    else:
        return f"😊 Positive Sentiment ({prediction:.2f})"

# ---------------------------
# 4️⃣ Streamlit Input
# ---------------------------
text_input = st.text_area("💬 Enter your review:")

if st.button("Analyze Sentiment"):
    if text_input.strip():
        result = predict_sentiment(text_input)
        st.subheader("💬 Prediction:")
        st.write(result)
    else:
        st.warning("Please enter some text before analyzing.")