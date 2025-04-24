import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
from tempfile import NamedTemporaryFile

# Set page configuration
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# Title and description
st.title("Speech Emotion Recognition (SER) Application")
st.markdown("""
This application analyzes speech audio to identify and present the emotion expressed within it.
The model is an 87.23% accuracy LSTM-based deep learning network that classifies 
emotions in speech from audio features.
""")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info("""
    Developed by Meidan Greenberg & Linoy Hadad.
    Original project supervised by Dr. Dima Alberg.
    Industrial Engineering and Management dept.
    SCE College, Israel.
    
    The model classifies speech into 8 emotions:
    - Neutral
    - Calm
    - Happy
    - Sad
    - Angry
    - Fearful
    - Disgust
    - Surprised
    """)

# Load the model
@st.cache_resource
def load_emotion_model():
    try:
        # You'll need to provide the path to your model files
        # For demo purposes, we'll assume they're in the current directory
        saved_model_path = 'model8723.json'
        saved_weights_path = 'model8723.weights.h5'
        
        # Reading the model from JSON file
        with open(saved_model_path, 'r') as json_file:
            json_savedModel = json_file.read()
        
        # Loading the model architecture, weights
        model = tf.keras.models.model_from_json(json_savedModel)
        model.load_weights(saved_weights_path)
        
        # Compiling the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='RMSProp',
            metrics=['categorical_accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing function
def preprocess(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=24414)
        
        # Extract MFCC features: 15 features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=15)
        
        # Transpose to (timesteps, features)
        mfcc = mfcc.T  # shape: (T, 15)
        
        # Pad or truncate to 339 timesteps
        if mfcc.shape[0] < 339:
            pad_width = 339 - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:339, :]
        
        # Final shape: (1, 339, 15)
        return np.expand_dims(mfcc, axis=0)
    
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

# Emotion labels
emo_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
emotions = {i: emo for i, emo in enumerate(emo_list)}

# Main app functionality
st.header("Speech Emotion Analysis")

# File upload option
uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

# Process the audio file if uploaded
if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Save uploaded file to a temporary file
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(uploaded_file.getvalue())
    
    # Process the audio file
    with st.spinner("Processing audio..."):
        start_time = time.perf_counter()
        
        # Load model (should be cached)
        model = load_emotion_model()
        
        if model is None:
            st.error("Failed to load the model. Please check model files.")
        else:
            # Preprocess the audio
            X = preprocess(temp_path)
            
            if X is not None:
                # Make predictions
                predictions = model.predict(X)
                pred_np = np.squeeze(predictions, axis=0)
                
                # Get the predicted emotion
                max_emo_idx = np.argmax(pred_np)
                predicted_emotion = emotions[max_emo_idx]
                
                # Calculate processing time
                processing_time = time.perf_counter() - start_time
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"Predicted Emotion: **{predicted_emotion.upper()}**")
                    st.info(f"Processed in {processing_time:.2f} seconds")
                    
                    # Display confidence
                    confidence = pred_np[max_emo_idx] * 100
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                with col2:
                    # Create a bar chart of emotion probabilities
                    fig, ax = plt.subplots(figsize=(10, 5))
                    bars = ax.bar(emo_list, pred_np, color='darkturquoise')
                    
                    # Highlight the predicted emotion
                    bars[max_emo_idx].set_color('red')
                    
                    ax.set_ylabel("Probability")
                    ax.set_title("Emotion Prediction")
                    ax.set_ylim(0, 1)
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45)
                    
                    # Use Streamlit's pyplot function
                    st.pyplot(fig)
                
                # Display a table with probabilities
                st.subheader("Emotion Probabilities")a
                prob_df = pd.DataFrame({
                    'Emotion': emo_list,
                    'Probability': pred_np,
                    'Percentage': [f"{p*100:.2f}%" for p in pred_np]
                })
                st.table(prob_df)
            else:
                st.error("Failed to process the audio file.")
    
    # Clean up the temporary file
    os.unlink(temp_path)

# Add missing import
if 'pd' not in locals():
    import pandas as pd

# Add instructions for running the app
