import streamlit as st
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification
import tensorflow as tf

# Load model and tokenizer
model = TFAlbertForSequenceClassification.from_pretrained("./")
tokenizer = AlbertTokenizer.from_pretrained("./")

# Streamlit app
st.title("Emotion Detection with ALBERT")

text = st.text_area("Enter a sentence to detect emotion:")

if st.button("Detect Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
        logits = model(inputs).logits
        probs = tf.sigmoid(logits).numpy()[0]

        # Update this list if you have specific emotions in your dataset
        labels = [ "example_very_unclear", "admiration", "amusement", "anger", "annoyance", 
"approval", "caring", "confusion", "curiosity", "desire", "disappointment", 
"disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", 
"grief", "joy", "love", "nervousness", "optimism", "pride", "realization", 
"relief", "remorse", "sadness", "surprise", "neutral"
]

        st.subheader("Emotion Probabilities:")
        for label, prob in zip(labels, probs):
            st.write(f"{label}: {prob:.2f}")
