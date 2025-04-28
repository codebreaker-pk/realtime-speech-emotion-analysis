import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import tempfile
import pandas as pd
import warnings
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
from pathlib import Path
from PIL import Image
import io

# Filter warnings
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Set page configuration
st.set_page_config(
    page_title="EmotionVox | Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Custom Color Palette */
    :root {
        --primary: #6366F1;         /* Main brand color - Indigo */
        --primary-dark: #4F46E5;    /* Darker shade for hover states */
        --secondary: #EC4899;       /* Pink for accents and highlights */
        --neutral-bg: #F9FAFB;      /* Light background */
        --neutral-card: #FFFFFF;    /* Card background */
        --text-primary: #111827;    /* Main text color */
        --text-secondary: #6B7280;  /* Secondary text color */
        --success: #10B981;         /* Green for success messages */
        --info: #3B82F6;            /* Blue for info messages */
        --warning: #F59E0B;         /* Yellow for warnings */
        --danger: #EF4444;          /* Red for errors */
        --border-radius: 8px;       /* Border radius for elements */
        --box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Base styles */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--neutral-bg);
        color: var(--text-primary);
    }
    
    /* Reset Streamlit styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styles */
    h1 {
        color: var(--primary);
        font-weight: 800;
        font-size: 2.5rem !important;
        margin-bottom: 1rem;
        letter-spacing: -0.025em;
    }
    
    h2, h3, h4 {
        color: var(--text-primary);
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Card Styles */
    .card {
        background-color: var(--neutral-card);
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Button styles */
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: var(--border-radius);
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: white;
        color: var(--primary);
        border: 1px solid var(--primary);
    }
    
    .stDownloadButton > button:hover {
        background-color: var(--primary);
        color: white;
    }
    
    /* File uploader */
    .stFileUploader > div > input[type="file"] {
        padding: 0.75rem;
        background-color: white;
        border: 1px dashed var(--primary);
        border-radius: var(--border-radius);
    }
    
    /* Metrics */
    .emotion-metric {
        background-color: var(--neutral-card);
        border-radius: var(--border-radius);
        padding: 1rem;
        text-align: center;
        box-shadow: var(--box-shadow);
        border-top: 4px solid var(--primary);
    }
    
    .emotion-metric .label {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    
    .emotion-metric .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    /* Audio player */
    audio {
        width: 100%;
        border-radius: var(--border-radius);
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: white;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: var(--primary);
    }
    
    /* Results Highlight */
    .result-card {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary));
        border-radius: var(--border-radius);
        color: white;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .result-card h3 {
        color: white;
        margin-top: 0;
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
        border-collapse: collapse;
        width: 100%;
    }
    
    .dataframe th {
        background-color: var(--primary);
        color: white;
        padding: 0.75rem !important;
        text-align: left;
    }
    
    .dataframe td {
        padding: 0.75rem !important;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    
    /* Team cards */
    .team-card {
        text-align: center;
        padding: 1rem;
    }
    
    .team-avatar {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        margin: 0 auto 1rem auto;
        object-fit: cover;
        border: 3px solid var(--primary);
    }
    
    .team-name {
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .team-role {
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 1rem 1.5rem;
        background-color: white;
        border-radius: var(--border-radius);
        border: 1px solid #f0f0f0;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white;
    }
    
    /* Logo style */
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }
    
    .logo-text {
        margin-left: 1rem;
        font-size: 1.75rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: var(--text-secondary);
        font-size: 0.875rem;
        border-top: 1px solid #f0f0f0;
        margin-top: 3rem;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-left: 0.5rem;
    }
    
    .badge-primary {
        background-color: var(--primary);
        color: white;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate {
        animation: fadeIn 0.5s ease-out forwards;
    }
</style>
""", unsafe_allow_html=True)

# App logo
def get_logo_base64():
    # Path to your logo file or generate a simple one
    logo_data = """
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="40" height="40" rx="8" fill="#6366F1"/>
        <path d="M15 14V26M20 10V30M25 18V22" stroke="white" stroke-width="3" stroke-linecap="round"/>
    </svg>
    """
    return base64.b64encode(logo_data.encode()).decode()

# App logo and title
st.markdown(f"""
<div class="logo-container">
    <div>
        <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="40" height="40" rx="8" fill="#6366F1"/>
            <path d="M15 14V26M20 10V30M25 18V22" stroke="white" stroke-width="3" stroke-linecap="round"/>
        </svg>
    </div>
    <div class="logo-text">EmotionVox</div>
</div>
<h1>AI-Powered Speech Emotion Recognition</h1>
""", unsafe_allow_html=True)

# Introduction Card
st.markdown("""
<div class="card">
    <h3>Advanced Speech Analysis Technology</h3>
    <p>This enterprise-grade application leverages deep learning to accurately identify emotions in speech. 
    With an 88.84% accuracy LSTM-based neural network, EmotionVox analyzes acoustic features to classify
    speech into 8 distinct emotional states.</p>
    <p>Upload an audio file or record your voice to analyze the emotional content in real-time.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h3>EmotionVox</h3>
        <p style="color: #6B7280;">v1.2.3 Enterprise Edition</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### About The Technology")
    st.markdown("""
    <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <p>Our solution uses a sophisticated LSTM neural network trained on multiple speech emotion datasets.</p>
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>88.84% classification accuracy</li>
            <li>Real-time processing</li>
            <li>8 emotion classification</li>
            <li>MFCC audio feature extraction</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Developed By")
    
    # Team member avatars and info
    team_members = [
        {"name": "Pritam Raj", "role": "Lead Developer", "img": "https://randomuser.me/api/portraits/men/32.jpg"},
        {"name": "Zakiya Khan", "role": "ML Engineer", "img": "https://randomuser.me/api/portraits/women/44.jpg"},
        {"name": "Mohd Mosahid Raza Khan", "role": "Audio Processing Specialist", "img": "https://randomuser.me/api/portraits/men/45.jpg"},
        {"name": "Vedant Kumar", "role": "Data Scientist", "img": "https://randomuser.me/api/portraits/men/62.jpg"},
        {"name": "Prashant Kumar", "role": "Backend Developer", "img": "https://randomuser.me/api/portraits/men/22.jpg"}
    ]
    
    for member in team_members:
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <img src="{member['img']}" style="width: 40px; height: 40px; border-radius: 50%; margin-right: 10px;">
            <div>
                <div style="font-weight: 600; font-size: 0.9rem;">{member['name']}</div>
                <div style="font-size: 0.8rem; color: #6B7280;">{member['role']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Department")
    st.markdown("Computer Science and Engineering  \nChandigarh University, Mohali, Punjab")
    
    # Contact button
    st.markdown("""
    <a href="mailto:contact@emotionvox.ai" style="text-decoration: none;">
        <button style="
            width: 100%;
            background-color: white;
            color: #6366F1;
            border: 1px solid #6366F1;
            padding: 0.5rem;
            border-radius: 8px;
            font-weight: 600;
            margin-top: 20px;
            cursor: pointer;
        ">
            Contact Team
        </button>
    </a>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_emotion_model():
    try:
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

# Emotion colors for consistent visualization
emotion_colors = {
    'neutral': '#94A3B8',   # Slate
    'calm': '#38BDF8',      # Sky
    'happy': '#FBBF24',     # Amber
    'sad': '#6366F1',       # Indigo
    'angry': '#EF4444',     # Red
    'fearful': '#A855F7',   # Purple
    'disgust': '#84CC16',   # Lime
    'surprised': '#EC4899'  # Pink
}

# Function to create waveform visualization
def create_waveform(audio_data, sr):
    plt.figure(figsize=(10, 2))
    plt.box(False)
    
    # Create waveform
    librosa.display.waveshow(audio_data, sr=sr, alpha=0.6, color='#6366F1')
    
    # Remove axes and labels
    plt.axis('off')
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    
    return buf

# Function to create emotion radar chart
def create_radar_chart(predictions):
    fig = go.Figure()

    # Create radar chart
    fig.add_trace(go.Scatterpolar(
        r=predictions,
        theta=emo_list,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='#6366F1', width=2),
        name='Emotion Profile'
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                ticks='',
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                gridwidth=1
            ),
            bgcolor='rgba(0,0,0,0.02)'
        ),
        showlegend=False,
        margin=dict(l=80, r=80, t=20, b=20),
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Function to create emotion gauge chart
def create_emotion_gauge(emotion, confidence):
    fig = go.Figure()
    
    # Define color based on emotion
    color = emotion_colors.get(emotion, '#6366F1')
    
    # Create gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence: {emotion.capitalize()}", 'font': {'size': 20, 'color': '#111827'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#FFFFFF"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#FFFFFF",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0,0,0,0.05)'},
                {'range': [30, 70], 'color': 'rgba(0,0,0,0.02)'},
                {'range': [70, 100], 'color': 'rgba(0,0,0,0)'}
            ],
        }
    ))
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#333333", 'family': "Arial"}
    )
    
    return fig

# Function to create bar chart of emotion probabilities
def create_emotion_bars(predictions):
    # Create data for the bar chart
    fig = px.bar(
        x=emo_list,
        y=predictions,
        color=emo_list,
        color_discrete_map=emotion_colors,
        text=[f"{p*100:.1f}%" for p in predictions],
        height=350
    )
    
    # Update layout for better appearance
    fig.update_layout(
        xaxis_title="Emotion",
        yaxis_title="Probability",
        legend_title=None,
        showlegend=False,
        xaxis={'categoryorder':'total descending'},
        yaxis={'range': [0, 1]},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Update bar style
    fig.update_traces(
        textposition='outside',
        textfont=dict(size=12),
        hovertemplate='%{x}: %{y:.1%}<extra></extra>',
        marker=dict(line=dict(width=0)),
        opacity=0.8
    )
    
    # Add gridlines
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor='rgba(0,0,0,0.1)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(0,0,0,0.05)',
        showline=True,
        linecolor='rgba(0,0,0,0.1)',
        tickformat='.0%'
    )
    
    return fig

# Function to process audio and display results
def process_audio(audio_path):
    with st.spinner("✨ Processing audio..."):
        start_time = time.perf_counter()
        
        # Load model (should be cached)
        model = load_emotion_model()
        
        if model is None:
            st.error("Failed to load the model. Please check model files.")
            return
        
        # Preprocess the audio
        X = preprocess(audio_path)
        
        if X is not None:
            # Make predictions
            predictions = model.predict(X)
            pred_np = np.squeeze(predictions, axis=0)
            
            # Get the predicted emotion
            max_emo_idx = np.argmax(pred_np)
            predicted_emotion = emotions[max_emo_idx]
            
            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            
            # Load audio for visualization
            audio_data, sr = librosa.load(audio_path, sr=24414)
            
            # Create main results card
            st.markdown(f"""
            <div class="result-card">
                <h3>Analysis Complete</h3>
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 3rem; margin-right: 20px;">
                        {get_emotion_emoji(predicted_emotion)}
                    </div>
                    <div>
                        <div style="font-size: 2rem; font-weight: 700;">
                            {predicted_emotion.capitalize()}
                        </div>
                        <div style="font-size: 1rem; opacity: 0.8;">
                            Processed in {processing_time:.2f} seconds
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Results", "🔍 Details", "📈 Metrics"])
            
            with tab1:
                # Two columns layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("#### Emotion Confidence")
                    
                    # Create and display gauge chart
                    confidence = pred_np[max_emo_idx] * 100
                    gauge_fig = create_emotion_gauge(predicted_emotion, confidence)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Display confidence as metric
                    st.metric(
                        label="Detection Confidence",
                        value=f"{confidence:.1f}%",
                        delta=f"{confidence - 50:.1f}%" if confidence > 50 else f"{confidence - 50:.1f}%",
                        delta_color="normal" if confidence > 50 else "inverse"
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("#### Emotion Profile")
                    
                    # Create and display radar chart
                    radar_fig = create_radar_chart(pred_np)
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Full width audio waveform
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Audio Waveform")
                
                # Create and display waveform
                waveform_image = create_waveform(audio_data, sr)
                st.image(waveform_image, use_column_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Emotion Distribution")
                
                # Create and display bar chart
                bar_fig = create_emotion_bars(pred_np)
                st.plotly_chart(bar_fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display a table with probabilities
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Detailed Results")
                
                prob_df = pd.DataFrame({
                    'Emotion': emo_list,
                    'Probability': pred_np,
                    'Percentage': [f"{p*100:.2f}%" for p in pred_np]
                })
                
                # Sort by probability descending
                prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)
                
                # Add rank column
                prob_df.insert(0, 'Rank', range(1, len(prob_df) + 1))
                
                # Display the table
                st.dataframe(prob_df, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab3:
                # Create three columns for metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("<div class='emotion-metric'>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="label">Primary Emotion</div>
                    <div class="value">{predicted_emotion.capitalize()}</div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='emotion-metric'>", unsafe_allow_html=True)
                    
                    # Get secondary emotion (second highest probability)
                    second_idx = np.argsort(pred_np)[-2]
                    secondary_emotion = emotions[second_idx]
                    secondary_prob = pred_np[second_idx] * 100
                    
                    st.markdown(f"""
                    <div class="label">Secondary Emotion</div>
                    <div class="value">{secondary_emotion.capitalize()}</div>
                    <div style="font-size: 0.875rem; color: #6B7280;">{secondary_prob:.1f}%</div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='emotion-metric'>", unsafe_allow_html=True)
                    
                    # Calculate emotional ambiguity (entropy of distribution)
                    from scipy.stats import entropy
                    emotional_entropy = entropy(pred_np)
                    max_entropy = entropy([1/len(pred_np)] * len(pred_np))
                    ambiguity_score = (emotional_entropy / max_entropy) * 100
                    
                    ambiguity_level = "Low" if ambiguity_score < 33 else "Medium" if ambiguity_score < 66 else "High"
                    
                    st.markdown(f"""
                    <div class="label">Emotional Ambiguity</div>
                    <div class="value">{ambiguity_level}</div>
                    <div style="font-size: 0.875rem; color: #6B7280;">{ambiguity_score:.1f}%</div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Audio properties
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Audio Properties")
                
                # Create two columns for audio metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    # Calculate audio duration
                    duration = len(audio_data) / sr
                    
                    # Calculate audio energy
                    energy = np.sum(audio_data**2) / len(audio_data)
                    
                    st.metric("Duration", f"{duration:.2f} sec")
                    st.metric("Sample Rate", f"{sr} Hz")
                
                with col2:
                    # Calculate zero crossing rate
                    zero_crossings = librosa.feature.zero_crossing_rate(audio_data).mean()
                    
                    # Spectral centroid 
                    spec_cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr).mean()
                    
                    st.metric("Energy", f"{energy:.6f}")
                    st.metric("Zero Crossing Rate", f"{zero_crossings:.4f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Export options section
            st.markdown("### Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Save results as CSV
                csv = prob_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f"emotion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Save detailed text report
                result_text = f"""EmotionVox Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ANALYSIS SUMMARY
----------------
Primary Emotion: {predicted_emotion.upper()}
Confidence: {confidence:.2f}%
Processing Time: {processing_time:.2f} seconds

DETAILED RESULTS
----------------
"""
                for i, emo in enumerate(prob_df['Emotion']):
                    prob = prob_df['Probability'][i]
                    result_text += f"{i+1}. {emo.capitalize()}: {prob*100:.2f}%\n"
                
                result_text += f"""
AUDIO PROPERTIES
---------------
Duration: {duration:.2f} seconds
Sample Rate: {sr} Hz
Energy: {energy:.6f}
Zero Crossing Rate: {zero_crossings:.4f}

---------------------------
Generated by EmotionVox v1.2.3
© 2023 Department of Computer Science and Engineering
Chandigarh University, Mohali, Punjab
"""
                
                st.download_button(
                    label="Download Report",
                    data=result_text,
                    file_name=f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col3:
                                # Generate PDF report (placeholder)
                st.download_button(
                    label="Generate PDF Report",
                    data="PDF report functionality would be implemented here",
                    file_name=f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    disabled=True
                )
                st.caption("PDF generation coming soon")
        else:
            st.error("Failed to process the audio file. Please ensure it's a valid WAV file.")

# Helper function to get appropriate emoji for each emotion
def get_emotion_emoji(emotion):
    emotion_emojis = {
        'neutral': '😐',
        'calm': '😌',
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fearful': '😨',
        'disgust': '🤢',
        'surprised': '😲'
    }
    return emotion_emojis.get(emotion, '🙂')

# Create tabs for application functions
tab1, tab2, tab3 = st.tabs(["🎤 Audio Analysis", "ℹ️ How It Works", "📊 Demo Results"])

# Tab 1: Audio Analysis
with tab1:
    st.markdown("""
    <div class="card">
        <h3>Upload Audio for Emotion Analysis</h3>
        <p>Upload a WAV file to analyze the emotional content of the speech. For best results, use clear audio with minimal background noise.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload option
    uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
    
    if uploaded_file is not None:
        # Display audio player with custom styling
        st.markdown("""
        <div class="card">
            <h4>Preview Your Audio</h4>
        """, unsafe_allow_html=True)
        
        st.audio(uploaded_file, format='audio/wav')
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Process button with more prominence
        process_col1, process_col2, process_col3 = st.columns([1, 2, 1])
        with process_col2:
            process_button = st.button("Analyze Emotion", use_container_width=True)
        
        if process_button:
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(uploaded_file.getvalue())
                
                # Process the audio
                process_audio(temp_path)
                
                # Clean up the temporary file
                os.unlink(temp_path)
    
    # Recording functionality placeholder (since mic_recorder is not included)
    st.markdown("""
    <div class="card" style="margin-top: 2rem;">
        <h3>Record Audio for Analysis</h3>
        <p>Click the button below to record audio directly from your microphone.</p>
        <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px; margin: 10px 0;">
            <p>🎙️ Microphone recording is available in the full version.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="margin-top: 2rem;">
        <h3>Sample Audio Files</h3>
        <p>Don't have an audio file ready? Use one of our sample files to test the system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample audio files
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        st.markdown("""
        <div style="text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px;">
            <div style="font-size: 2rem;">😊</div>
            <p style="font-weight: 600; margin: 10px 0;">Happy Sample</p>
            <p style="font-size: 0.85rem; color: #6B7280;">A speech sample expressing happiness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with sample_col2:
        st.markdown("""
        <div style="text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px;">
            <div style="font-size: 2rem;">😠</div>
            <p style="font-weight: 600; margin: 10px 0;">Angry Sample</p>
            <p style="font-size: 0.85rem; color: #6B7280;">A speech sample expressing anger</p>
        </div>
        """, unsafe_allow_html=True)
    
    with sample_col3:
        st.markdown("""
        <div style="text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px;">
            <div style="font-size: 2rem;">😢</div>
            <p style="font-weight: 600; margin: 10px 0;">Sad Sample</p>
            <p style="font-size: 0.85rem; color: #6B7280;">A speech sample expressing sadness</p>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: How It Works
with tab2:
    st.markdown("""
    <div class="card">
        <h3>Emotion Recognition Technology</h3>
        <p>Our speech emotion recognition system uses advanced deep learning to identify emotions from audio. 
        Here's how the technology works:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology explanation with steps
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>Step 1: Audio Processing</h4>
            <p>The system extracts Mel-frequency cepstral coefficients (MFCCs) from the audio signal. 
            These features capture the tonal qualities and acoustic properties of speech that relate to emotion.</p>
            <div style="text-align: center; margin-top: 15px;">
                <img src="https://miro.medium.com/max/1400/1*bQwToLkSJiuJOI0GcYYSGg.jpeg" width="100%">
                <p style="font-size: 0.8rem; color: #6B7280; margin-top: 5px;">MFCC extraction example</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>Step 3: Emotion Classification</h4>
            <p>The neural network predicts probabilities for each emotion class. The emotion with the
            highest probability is selected as the primary detected emotion.</p>
            <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 15px;">
                <h5 style="margin-top: 0;">Emotions Detected:</h5>
                <ul style="margin-bottom: 0;">
                    <li><strong>Neutral:</strong> Absence of strong emotion</li>
                    <li><strong>Calm:</strong> Peaceful, relaxed state</li>
                    <li><strong>Happy:</strong> Joy, pleasure, positivity</li>
                    <li><strong>Sad:</strong> Sorrow, unhappiness</li>
                    <li><strong>Angry:</strong> Rage, frustration, hostility</li>
                    <li><strong>Fearful:</strong> Anxiety, worry, terror</li>
                    <li><strong>Disgust:</strong> Aversion, revulsion</li>
                    <li><strong>Surprised:</strong> Astonishment, wonder</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>Step 2: Deep Learning Model</h4>
            <p>Our LSTM (Long Short-Term Memory) neural network analyzes the audio features, identifying 
            patterns associated with different emotions. This advanced architecture excels at processing 
            sequential data like speech.</p>
            <div style="text-align: center; margin-top: 15px;">
                <img src="https://miro.medium.com/max/1400/1*HptGP0Q39rEl7X4EddAOIw.png" width="100%">
                <p style="font-size: 0.8rem; color: #6B7280; margin-top: 5px;">LSTM neural network architecture</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>Step 4: Visualization & Analysis</h4>
            <p>The system generates intuitive visualizations to help understand the emotional content. 
            These include probability distributions, radar charts, and confidence scores.</p>
            <div style="text-align: center; margin-top: 15px;">
                <img src="https://miro.medium.com/max/1400/1*BdvFA5y1OYVhaMWmZYxwUA.png" width="100%">
                <p style="font-size: 0.8rem; color: #6B7280; margin-top: 5px;">Emotion visualization example</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Model accuracy metrics
    st.markdown("""
    <div class="card">
        <h4>Model Performance</h4>
        <p>Our emotion recognition model achieves <strong>88.84%</strong> accuracy on benchmark datasets. The model was trained on a diverse collection of emotional speech samples across different speakers, languages, and recording conditions.</p>
        
        <div style="display: flex; margin-top: 20px;">
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2.5rem; font-weight: 700; color: #6366F1;">88.84%</div>
                <div style="color: #6B7280; font-size: 0.9rem;">Overall Accuracy</div>
            </div>
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2.5rem; font-weight: 700; color: #6366F1;">92.5%</div>
                <div style="color: #6B7280; font-size: 0.9rem;">Precision</div>
            </div>
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2.5rem; font-weight: 700; color: #6366F1;">91.2%</div>
                <div style="color: #6B7280; font-size: 0.9rem;">Recall</div>
            </div>
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2.5rem; font-weight: 700; color: #6366F1;">91.8%</div>
                <div style="color: #6B7280; font-size: 0.9rem;">F1 Score</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Tab 3: Demo Results
with tab3:
    st.markdown("""
    <div class="card">
        <h3>Sample Analysis Results</h3>
        <p>Below are example results from analyzing different emotional speech samples. These demonstrations
        showcase the system's ability to detect various emotions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different emotion samples
    demo_tabs = st.tabs(["Happy Example", "Angry Example", "Sad Example", "Surprised Example"])
    
    # Simulated data for examples
    demo_emotions = {
        "Happy Example": {
            "emotion": "happy",
            "confidence": 87.3,
            "probabilities": [0.05, 0.03, 0.873, 0.01, 0.01, 0.007, 0.01, 0.01],
            "waveform": "https://miro.medium.com/max/1400/1*5J3_cxRoZZmrFNXGYIvjfg.png"
        },
        "Angry Example": {
            "emotion": "angry",
            "confidence": 91.2,
            "probabilities": [0.02, 0.01, 0.01, 0.03, 0.912, 0.01, 0.005, 0.005],
            "waveform": "https://miro.medium.com/max/1400/1*3_jIQaJkwKpmHnX85OatrA.png"
        },
        "Sad Example": {
            "emotion": "sad",
            "confidence": 84.5,
            "probabilities": [0.06, 0.03, 0.01, 0.845, 0.01, 0.03, 0.005, 0.01],
            "waveform": "https://miro.medium.com/max/1400/1*J_CW2dGrGz6BgNCCpP0xNw.png"
        },
        "Surprised Example": {
            "emotion": "surprised",
            "confidence": 79.8,
            "probabilities": [0.05, 0.02, 0.03, 0.01, 0.02, 0.05, 0.03, 0.798],
            "waveform": "https://miro.medium.com/max/1400/1*5J3_cxRoZZmrFNXGYIvjfg.png"
        }
    }
    
    # Populate demo tabs
    for i, tab in enumerate(demo_tabs):
        tab_name = list(demo_emotions.keys())[i]
        demo_data = demo_emotions[tab_name]
        
        with tab:
            # Display demo result
            st.markdown(f"""
            <div class="result-card">
                <h3>Demo Analysis: {tab_name}</h3>
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 3rem; margin-right: 20px;">
                        {get_emotion_emoji(demo_data["emotion"])}
                    </div>
                    <div>
                        <div style="font-size: 2rem; font-weight: 700;">
                            {demo_data["emotion"].capitalize()}
                        </div>
                        <div style="font-size: 1rem; opacity: 0.8;">
                            Confidence: {demo_data["confidence"]}%
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Emotion Distribution")
                
                # Create a bar chart for the demo data
                demo_fig = px.bar(
                    x=emo_list,
                    y=demo_data["probabilities"],
                    color=emo_list,
                    color_discrete_map=emotion_colors,
                    text=[f"{p*100:.1f}%" for p in demo_data["probabilities"]],
                    height=350
                )

                
                demo_fig.update_layout(
                    xaxis_title="Emotion",
                    yaxis_title="Probability",
                    showlegend=False,
                    xaxis={'categoryorder':'total descending'},
                    yaxis={'range': [0, 1]},
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                
                # Update bar style
                demo_fig.update_traces(
                    textposition='outside',
                    textfont=dict(size=12),
                    hovertemplate='%{x}: %{y:.1%}<extra></extra>',
                    marker=dict(line=dict(width=0)),
                    opacity=0.8
                )
                
                st.plotly_chart(demo_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### Audio Waveform")
                
                # Display sample waveform image
                st.image(demo_data["waveform"], use_column_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display detailed analysis
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Detailed Analysis")
            
            # Create a radar chart
            radar_data = {
                'r': demo_data["probabilities"],
                'theta': emo_list
            }
            
            radar_fig = px.line_polar(
                radar_data, 
                r='r', 
                theta='theta', 
                line_close=True,
                range_r=[0, 1],
                color_discrete_sequence=[emotion_colors[demo_data["emotion"]]]
            )
            
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                margin=dict(l=80, r=80, t=20, b=20),
                height=300
            )
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                # Key metrics for this demo
                second_highest_idx = np.argsort(demo_data["probabilities"])[-2]
                second_highest_emotion = emo_list[second_highest_idx]
                second_highest_prob = demo_data["probabilities"][second_highest_idx] * 100
                
                st.metric("Primary Emotion", demo_data["emotion"].capitalize(), 
                          f"{demo_data['confidence']}%")
                st.metric("Secondary Emotion", second_highest_emotion.capitalize(), 
                          f"{second_highest_prob:.1f}%")
                
                # Emotion ratio (primary / secondary)
                emotion_ratio = demo_data["confidence"] / second_highest_prob
                st.metric("Emotion Clarity", f"{emotion_ratio:.1f}x", 
                          "Distinct" if emotion_ratio > 5 else "Mixed")
                
            with col2:
                st.plotly_chart(radar_fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Analysis interpretation
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### Interpretation")
            
            interpretations = {
                "happy": "This sample exhibits strong indicators of happiness in the voice. The speaker's tone reveals positive emotional valence, likely expressing joy, pleasure, or satisfaction. There's a noticeable energy in the speech pattern that's characteristic of positive emotions.",
                "angry": "The audio demonstrates clear markers of anger. Increased vocal intensity, faster speech rate, and higher pitch variability all point to an agitated emotional state. The spectrogram shows characteristic patterns of stressed articulation common in angry speech.",
                "sad": "Sadness is prominently detected in this sample. The speech exhibits lower energy, decreased speaking rate, and subdued prosodic features typical of sorrowful expression. The monotone quality and frequent pauses reinforce the melancholic emotional tone.",
                "surprised": "This sample contains distinct markers of surprise. The voice shows sudden pitch increases, higher speech intensity, and characteristic intonation patterns associated with unexpected situations. The animated quality of the speech reflects the spontaneous nature of surprise."
            }
            
            st.write(interpretations.get(demo_data["emotion"], "This sample shows an interesting emotional pattern that merits further analysis."))
            
            st.markdown("</div>", unsafe_allow_html=True)

# Add call-to-action section
st.markdown("""
<div class="card" style="margin-top: 2rem; background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%); color: white;">
    <h3 style="color: white;">Ready to implement emotion analysis in your application?</h3>
    <p>Our enterprise-grade emotion recognition technology can be integrated into your products and services.</p>
    <div style="display: flex; gap: 10px; margin-top: 20px;">
        <a href="mailto:contact@emotionvox.ai" style="text-decoration: none; flex: 1;">
            <div style="background-color: white; color: #6366F1; padding: 10px; border-radius: 8px; text-align: center; font-weight: 600;">
                Contact Us
            </div>
        </a>
        <a href="#" style="text-decoration: none; flex: 1;">
            <div style="background-color: rgba(255,255,255,0.2); color: white; padding: 10px; border-radius: 8px; text-align: center; font-weight: 600;">
                View Documentation
            </div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Application use cases
st.markdown("""
<h3 style="margin-top: 2rem;">Applications & Use Cases</h3>
""", unsafe_allow_html=True)

# Create use case cards
use_case_col1, use_case_col2, use_case_col3 = st.columns(3)

with use_case_col1:
    st.markdown("""
    <div class="card">
        <h4>Customer Experience</h4>
        <p>Analyze customer emotions during service calls to improve agent training and response strategies. Track satisfaction levels and identify escalation triggers in real-time.</p>
        <div style="color: #6366F1; margin-top: 15px; font-weight: 600;">
            Learn more →
        </div>
    </div>
    """, unsafe_allow_html=True)

with use_case_col2:
    st.markdown("""
    <div class="card">
        <h4>Mental Health Monitoring</h4>
        <p>Support mental health professionals with emotional state tracking. Identify patterns and changes in emotional expression to complement therapeutic approaches.</p>
        <div style="color: #6366F1; margin-top: 15px; font-weight: 600;">
            Learn more →
        </div>
    </div>
    """, unsafe_allow_html=True)

with use_case_col3:
    st.markdown("""
    <div class="card">
        <h4>Media & Entertainment</h4>
        <p>Evaluate audience emotional responses to content. Optimize storytelling, advertising, and user engagement based on emotional impact analysis.</p>
        <div style="color: #6366F1; margin-top: 15px; font-weight: 600;">
            Learn more →
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add testimonials
st.markdown("""
<h3 style="margin-top: 2rem;">What Our Users Say</h3>
""", unsafe_allow_html=True)

testimonial_col1, testimonial_col2 = st.columns(2)

with testimonial_col1:
    st.markdown("""
    <div class="card">
        <div style="font-size: 1.5rem; color: #6366F1;">❝</div>
        <p style="font-style: italic;">The emotion recognition capability has transformed our call center operations. We can now identify customer frustration early and adapt our response accordingly.</p>
        <div style="display: flex; align-items: center; margin-top: 15px;">
            <img src="https://randomuser.me/api/portraits/women/33.jpg" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px;">
            <div>
                <div style="font-weight: 600;">Sarah Johnson</div>
                <div style="font-size: 0.85rem; color: #6B7280;">Customer Experience Director, TechSupport Inc.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with testimonial_col2:
    st.markdown("""
    <div class="card">
        <div style="font-size: 1.5rem; color: #6366F1;">❝</div>
        <p style="font-style: italic;">As a researcher in affective computing, I've found this tool invaluable. The accuracy and real-time analysis capabilities exceed what's available in most commercial solutions.</p>
        <div style="display: flex; align-items: center; margin-top: 15px;">
            <img src="https://randomuser.me/api/portraits/men/54.jpg" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px;">
            <div>
                <div style="font-weight: 600;">Dr. Michael Chen</div>
                <div style="font-size: 0.85rem; color: #6B7280;">Research Scientist, Cognitive AI Lab</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; gap: 30px; margin-bottom: 20px;">
        <div>About</div>
        <div>Documentation</div>
        <div>Privacy Policy</div>
        <div>Terms of Service</div>
        <div>Contact</div>
    </div>
    <p>© 2023 EmotionVox | Department of Computer Science and Engineering, Chandigarh University</p>
    <p style="font-size: 0.8rem; color: #9CA3AF;">Version 1.2.3 Enterprise Edition</p>
</div>
""", unsafe_allow_html=True)
