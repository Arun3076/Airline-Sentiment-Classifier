import streamlit as st
from transformers import pipeline
import os
import pandas as pd

# Set page config FIRST
st.set_page_config(
    page_title="Airline Tweet Sentiment Classifier",
    page_icon="✈️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
    .stButton>button:hover {
        background-color: #1C86EE;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #d3d3d3;
    }
    .sentiment-positive { color: #28A745; font-weight: bold; }
    .sentiment-negative { color: #DC3545; font-weight: bold; }
    .sentiment-neutral { color: #FFC107; font-weight: bold; }
    .example-tweet { 
        cursor: pointer; 
        padding: 0.5rem; 
        margin: 0.2rem; 
        border-radius: 5px; 
        background-color: #f8f9fa; 
    }
    .example-tweet:hover { background-color: #e9ecef; }
    </style>
""", unsafe_allow_html=True)

# Load model from local fine-tuned folder
@st.cache_resource
def load_pipeline():
    model_dir = "fine-tuned-airline-model"
    try:
        return pipeline("text-classification", model=model_dir, tokenizer=model_dir)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

clf = load_pipeline()

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar with examples and info
with st.sidebar:
    st.header("About")
    st.markdown(
        "This app classifies the sentiment of airline-related tweets using a fine-tuned transformer model. "
        "Enter a tweet or select an example below to analyze its sentiment."
    )
    
    st.subheader("Example Tweets")
    example_tweets = [
        "Loved the service on my recent Airline flight! Crew was amazing!",
        "Worst experience ever with Airline. Delayed flight and rude staff.",
        "Currently waiting at the airport lounge."
    ]
    for ex_tweet in example_tweets:
        if st.button(ex_tweet, key=ex_tweet):
            st.session_state.tweet_input = ex_tweet

# Main App UI
st.title("✈️ Airline Tweet Sentiment Classifier")
st.markdown("Analyze the sentiment of airline-related tweets. Enter your own tweet or try an example from the sidebar.")

# Input area
tweet = st.text_area(
    "✍️ Your Tweet:",
    value=st.session_state.get('tweet_input', ''),
    height=100,
    placeholder="Type your airline-related tweet here..."
)

# Predict button and results
# Predict button and results
if st.button("🧠 Predict Sentiment"):
    if not clf:
        st.error("Model not loaded. Please check the model directory.")
    elif tweet.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                result = clf(tweet)[0]
                # Map numeric labels to sentiment names
                label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
                sentiment = label_map.get(result['label'], result['label']).capitalize()
                confidence = result['score']
                
                # Style sentiment output
                sentiment_class = (
                    "sentiment-positive" if sentiment.lower() == "positive" else
                    "sentiment-negative" if sentiment.lower() == "negative" else
                    "sentiment-neutral"
                )
                
                st.markdown(
                    f"**Sentiment:** <span class='{sentiment_class}'>{sentiment}</span>  \n"
                    f"**Confidence:** {confidence:.2%}",
                    unsafe_allow_html=True
                )

                # Add to history
                st.session_state.history.append({
                    'tweet': tweet,
                    'sentiment': sentiment,
                    'confidence': confidence
                })

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# Display history
if st.session_state.history:
    st.markdown("### Recent Analyses")
    history_df = pd.DataFrame(st.session_state.history)
    history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.2%}")
    st.dataframe(
        history_df[['tweet', 'sentiment', 'confidence']],
        use_container_width=True,
        column_config={
            'tweet': 'Tweet',
            'sentiment': 'Sentiment',
            'confidence': 'Confidence'
        }
    )

    # Option to clear history
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()



#!/usr/bin/env python
# coding: utf-8

# import streamlit as st
# from transformers import pipeline
# import os

# # Set page config FIRST
# st.set_page_config(page_title="Airline Tweet Sentiment Classifier", layout="centered")

# # Load model from local fine-tuned folder
# @st.cache_resource
# def load_pipeline():
#     model_dir = "fine-tuned-airline-model"
#     return pipeline("text-classification", model=model_dir, tokenizer=model_dir)

# clf = load_pipeline()

# # App UI
# st.title("✈️ Airline Tweet Sentiment Classifier")
# st.markdown("Enter an airline-related tweet below and classify its sentiment.")

# tweet = st.text_area("✍️ Your Tweet:", height=100)

# if st.button("🧠 Predict Sentiment"):
#     if tweet.strip() == "":
#         st.warning("Please enter a tweet to analyze.")
#     else:
#         with st.spinner("Analyzing..."):
#             result = clf(tweet)[0]
#             sentiment = result['label']
#             confidence = result['score']
#             st.success(f"**Sentiment:** `{sentiment}` \n\n **Confidence:** `{confidence:.2f}`")
