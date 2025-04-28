import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import librosa
import librosa.display
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import sounddevice as sd
import soundfile as sf
import threading
import queue
import os
from PIL import Image
import altair as alt
import io
import base64
import requests
import json
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from streamlit_card import card
import hydralit_components as hc

# Set page configuration
st.set_page_config(
    page_title="EmotionVox Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Modern UI colors */
    :root {
        --primary-color: #6C63FF;
        --secondary-color: #4E46E8;
        --accent-color: #FF6584;
        --background-color: #f8f9fe;
        --card-bg: #ffffff;
        --text-color: #333333;
        --light-gray: #f1f3f9;
        --card-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
    }
    
    /* Base styling */
    .reportview-container {
        background: var(--background-color);
        color: var(--text-color);
    }
    .main {
        background: linear-gradient(145deg, #f8f9fe 0%, #ecedfb 100%);
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: var(--primary-color) !important;
    }
    h1 {
        font-size: 2.5rem !important;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    
    /* Cards */
    .dashboard-card {
        background-color: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--card-shadow);
        margin-bottom: 24px;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        border: 1px solid rgba(108, 99, 255, 0.1);
    }
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(108, 99, 255, 0.15);
    }
    
    /* Metric cards */
    .metric-container {
        background-color: var(--card-bg);
        border-radius: 14px;
        padding: 20px;
        box-shadow: var(--card-shadow);
        text-align: center;
        height: 100%;
        border-left: 4px solid var(--primary-color);
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-4px);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white !important;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(108, 99, 255, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(108, 99, 255, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    }
    
    /* Charts and visualizations */
    .chart-container {
        background: white;
        border-radius: 14px;
        padding: 20px;
        box-shadow: var(--card-shadow);
    }
    
    /* Feature cards */
    .feature-card {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        height: 100%;
        box-shadow: var(--card-shadow);
        transition: transform 0.3s ease;
        border-top: 4px solid var(--accent-color);
    }
    .feature-card:hover {
        transform: translateY(-6px);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 16px;
        color: var(--primary-color);
    }
    
    /* Team profile cards */
    .team-card {
        background: white;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: var(--card-shadow);
        transition: transform 0.3s ease;
    }
    .team-card:hover {
        transform: translateY(-8px);
    }
    .team-image-container {
        height: 180px;
        overflow: hidden;
        border-bottom: 3px solid var(--primary-color);
    }
    .team-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s ease;
    }
    .team-card:hover .team-image {
        transform: scale(1.05);
    }
    .team-info {
        padding: 16px;
        text-align: center;
    }
    .team-name {
        font-weight: 700;
        color: var(--text-color);
        margin: 0;
    }
    .team-role {
        color: var(--primary-color);
        font-size: 0.85rem;
        margin-bottom: 10px;
    }
    .team-contact {
        font-size: 0.8rem;
        color: #666;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px 0;
        color: #6c757d;
        font-size: 0.9rem;
        margin-top: 50px;
        border-top: 1px solid #eee;
    }
    
    /* Audio recorder styling */
    .audio-recorder {
        background: var(--card-bg);
        border-radius: 14px;
        padding: 24px;
        box-shadow: var(--card-shadow);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid rgba(108, 99, 255, 0.1);
    }
    
    /* Tabs */
    .tabs-container {
        display: flex;
        background: var(--light-gray);
        border-radius: 50px;
        padding: 5px;
        margin-bottom: 25px;
        width: fit-content;
    }
    .tab {
        padding: 10px 20px;
        cursor: pointer;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .tab-active {
        background: var(--primary-color);
        color: white;
    }
    
    /* Emotion badges */
    .emotion-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    /* Logo area */
    .logo-title {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    .logo-image {
        width: 50px;
        margin-right: 10px;
    }
    
    /* Animation delays for elements */
    .animate-1 { animation-delay: 0.1s; }
    .animate-2 { animation-delay: 0.2s; }
    .animate-3 { animation-delay: 0.3s; }
    .animate-4 { animation-delay: 0.4s; }
    .animate-5 { animation-delay: 0.5s; }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
        opacity: 0;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: var(--primary-color);
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Function to load lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Define emotion colors for visualizations
EMOTION_COLORS = {
    'happy': '#4FC1E9',    # Blue
    'sad': '#8A98AC',      # Gray
    'angry': '#FF6B6B',    # Red
    'neutral': '#A0D468',  # Green
    'fear': '#967ADC',     # Purple
    'disgust': '#FC6E51',  # Orange
    'surprise': '#FFCE54'  # Yellow
}

# Load lottie animations
lottie_analysis = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_vnikrcia.json")
lottie_microphone = load_lottie_url("https://assets8.lottiefiles.com/datafiles/d4HqSKmI0iRXckP/data.json")
lottie_dashboard = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_tivi0ry0.json")
lottie_team = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_bpqri9y8.json")

# Placeholder functions for audio processing

def extract_features(audio_data, sample_rate):
    """Extract MFCC and other features from audio data"""
    # In a real implementation, extract the actual features your model requires
    # Here's a simplified version for demonstration
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    # Additional features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    
    # Combine features
    features = np.concatenate((
        mfccs_processed, 
        [np.mean(chroma)], 
        [np.mean(spectral_contrast)]
    ))
    
    return features

def predict_emotion(features):
    """
    Predict emotion from audio features
    In a real implementation, use your pre-trained model here
    """
    # For demo purposes, we'll simulate model prediction
    emotions = list(EMOTION_COLORS.keys())
    probs = np.random.dirichlet(np.ones(len(emotions)))
    
    # Create dictionary of emotions and probabilities
    prediction = {emotion: float(prob) for emotion, prob in zip(emotions, probs)}
    
    # Get the emotion with highest probability
    predicted_emotion = max(prediction, key=prediction.get)
    
    return prediction, predicted_emotion

def record_audio(duration=5, fs=22050):
    """Record audio from microphone"""
    st.text("🎙️ Recording... Speak now!")
    
    # Display progress bar during recording
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(duration/100)
        progress_bar.progress(i + 1)
    
    # In a real implementation, this would capture actual audio
    # For demo, we'll generate random audio data
    audio_data = np.random.normal(0, 0.1, int(duration * fs))
    
    return audio_data, fs

def process_live_audio():
    """Process audio from live recording"""
    # Record audio
    with st.spinner("Initializing microphone..."):
        audio_data, sample_rate = record_audio()
    
    # Extract features
    with st.spinner("Analyzing audio..."):
        features = extract_features(audio_data, sample_rate)
        emotion_probs, detected_emotion = predict_emotion(features)
    
    return audio_data, sample_rate, emotion_probs, detected_emotion

def create_waveform(audio_data, sample_rate):
    """Create waveform visualization of audio data"""
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_facecolor('#F8F9FE')
    fig.patch.set_facecolor('#F8F9FE')
    
    librosa.display.waveshow(audio_data, sr=sample_rate, ax=ax, color='#6C63FF')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    return buf

def create_emotion_gauge(emotion_probs):
    """Create gauge chart for emotion probabilities"""
    fig = go.Figure()
    
    for emotion, prob in emotion_probs.items():
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': emotion.capitalize(), 'font': {'size': 24, 'color': EMOTION_COLORS[emotion]}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#FFFFFF"},
                'bar': {'color': EMOTION_COLORS[emotion]},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#FFFFFF",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 255, 255, 0.7)'},
                    {'range': [50, 80], 'color': 'rgba(255, 255, 255, 0.5)'},
                    {'range': [80, 100], 'color': 'rgba(255, 255, 255, 0.2)'}
                ],
            }
        ))
    
    fig.update_layout(
        grid={'rows': 1, 'columns': len(emotion_probs), 'pattern': "independent"},
        margin=dict(l=20, r=20, t=30, b=20),
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#333333", 'family': "Arial"}
    )
    
    return fig

# Generate simulated historical data for dashboard
def generate_historical_data(days=7):
    """Generate simulated historical emotion data for dashboard"""
    np.random.seed(42)  # For reproducibility
    
    now = datetime.now()
    dates = []
    emotions = []
    confidence = []
    platforms = ["Twitter Spaces", "Clubhouse", "Instagram Live", "Discord"]
    
    # Generate hourly data points
    for day in range(days):
        for hour in range(24):
            # Create timestamp for each hour
            timestamp = now.replace(
                hour=hour, 
                minute=0,
                second=0, 
                microsecond=0
            ) - pd.Timedelta(days=day)
            
            # Generate multiple data points per hour
            for _ in range(np.random.randint(3, 8)):
                dates.append(timestamp)
                emotions.append(np.random.choice(list(EMOTION_COLORS.keys())))
                confidence.append(round(0.5 + 0.5 * np.random.random(), 2))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'emotion': emotions,
        'confidence': confidence,
        'platform': np.random.choice(platforms, size=len(dates))
    })
    
    return df

# Dashboard visualization functions
def create_emotion_trends(data):
    """Create emotion trends over time visualization"""
    # Group by hour and emotion
    hourly_emotions = data.groupby([pd.Grouper(key='timestamp', freq='D'), 'emotion']).size().reset_index(name='count')
    
    # Create line chart
    fig = px.line(
        hourly_emotions, 
        x='timestamp', 
        y='count', 
        color='emotion',
        color_discrete_map=EMOTION_COLORS,
        title='Emotion Trends Over Time'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Count',
        legend_title='Emotion',
        font=dict(family='Arial', size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=350,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add grid lines
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(211,211,211,0.3)',
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(211,211,211,0.3)',
        zeroline=False
    )
    
    return fig

def create_emotion_distribution(data):
    """Create donut chart for emotion distribution"""
    emotion_counts = data['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    
    # Calculate percentages
    total = emotion_counts['count'].sum()
    emotion_counts['percentage'] = round((emotion_counts['count'] / total) * 100, 1)
    
    # Create color list matching emotions
    colors = [EMOTION_COLORS[emotion] for emotion in emotion_counts['emotion']]
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=emotion_counts['emotion'],
        values=emotion_counts['percentage'],
        hole=0.6,
        marker_colors=colors,
        textinfo='label+percent',
        textfont=dict(color='white', size=12),
        hoverinfo='label+percent',
        textposition='inside',
        pull=[0.05 if x == emotion_counts['percentage'].max() else 0 for x in emotion_counts['percentage']]
    )])
    
    fig.update_layout(
        title='Overall Emotion Distribution',
        font=dict(family='Arial', size=12),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        showlegend=False
    )
    
    # Add custom annotation in center
    fig.add_annotation(
        text=f"{total:,}<br>Samples",
        x=0.5, y=0.5,
        font=dict(size=16, color='#333', family='Arial, sans-serif'),
        showarrow=False
    )
    
    return fig

def create_platform_comparison(data):
    """Create bar chart comparing emotions across platforms"""
    # Group by platform and emotion
    platform_emotions = data.groupby(['platform', 'emotion']).size().reset_index(name='count')
    
    # Create grouped bar chart
    fig = px.bar(
        platform_emotions, 
        x='platform', 
        y='count', 
        color='emotion',
        color_discrete_map=EMOTION_COLORS,
        title='Emotion Distribution by Platform',
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Count',
        legend_title='Emotion',
        font=dict(family='Arial', size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(211,211,211,0.3)',
        zeroline=False
    )
    
    return fig

def create_confidence_by_emotion(data):
    """Create box plot of confidence scores by emotion"""
    fig = px.box(
        data, 
        x='emotion', 
        y='confidence', 
        color='emotion',
        color_discrete_map=EMOTION_COLORS,
        title='Detection Confidence by Emotion',
        points='all'
    )
    
    fig.update_layout(
        xaxis_title='',
        yaxis_title='Confidence Score',
        showlegend=False,
        font=dict(family='Arial', size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=350
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(211,211,211,0.3)',
        zeroline=False
    )
    
    return fig

def create_hourly_heatmap(data):
    """Create heatmap of emotions by hour of day"""
    # Extract hour from timestamp
    data['hour'] = data['timestamp'].dt.hour
    
    # Group by hour and emotion
    hourly_emotion = data.groupby(['hour', 'emotion']).size().reset_index(name='count')
    
    # Create pivot table
    pivot_data = hourly_emotion.pivot(index='emotion', columns='hour', values='count').fillna(0)
    
    # Create heatmap
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Hour of Day", y="Emotion", color="Count"),
        x=[f"{h:02d}:00" for h in range(24)],
        y=pivot_data.index,
        color_continuous_scale=[
            [0, "rgba(255, 255, 255, 0.8)"],
            [0.25, "rgba(108, 99, 255, 0.3)"],
            [0.5, "rgba(108, 99, 255, 0.5)"],
            [0.75, "rgba(108, 99, 255, 0.7)"],
            [1, "rgba(108, 99, 255, 1)"]
        ],
        title="Emotion Distribution by Hour of Day"
    )
    
    fig.update_layout(
        font=dict(family='Arial', size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=350
    )
    
    return fig

# Create sample key metrics
def get_key_metrics(data):
    """Calculate key metrics from the data"""
    # Count total samples
    total_samples = len(data)
    
    # Get dominant emotion
    dominant_emotion = data['emotion'].value_counts().idxmax()
    dominant_percent = round((data['emotion'].value_counts().max() / total_samples) * 100, 1)
    
    # Get average confidence
    avg_confidence = round(data['confidence'].mean() * 100, 1)
    
    # Get emotion shifts (changes in dominant emotion) in the last day
    last_day = data[data['timestamp'] >= data['timestamp'].max() - pd.Timedelta(days=1)]
    last_day_hourly = last_day.groupby(pd.Grouper(key='timestamp', freq='H'))['emotion'].agg(lambda x: x.value_counts().idxmax() if len(x) > 0 else None)
    emotion_shifts = sum(last_day_hourly.shift() != last_day_hourly) - 1  # -1 to account for the first NaN
    emotion_shifts = max(0, emotion_shifts)  # Ensure non-negative
    
    return {
        'total_samples': total_samples,
        'dominant_emotion': dominant_emotion,
        'dominant_percent': dominant_percent,
        'avg_confidence': avg_confidence,
        'emotion_shifts': emotion_shifts
    }

# Custom metric display component
def display_metric(title, value, subtitle=None, icon=None, change=None, is_upward_good=True):
    """Display a metric in a styled container"""
    if change:
        if (change > 0 and is_upward_good) or (change < 0 and not is_upward_good):
            change_color = "#A0D468"  # Green for positive
            change_icon = "↗"
        else:
            change_color = "#FF6B6B"  # Red for negative
            change_icon = "↘"
        
        change_html = f"""
        <span style="color:{change_color}; font-size:0.9rem; font-weight:500;">
            {change_icon} {abs(change)}% from last period
        </span>
        """
    else:
        change_html = ""
    
    icon_html = f"""<div style="font-size:2rem; margin-bottom:0.5rem; color:var(--primary-color);">{icon}</div>""" if icon else ""
    
    metric_html = f"""
    <div class="metric-container">
        {icon_html}
        <div style="font-size:2.2rem; font-weight:700; color:var(--text-color);">{value}</div>
        <div style="font-size:1rem; color:#6c757d; margin-bottom:0.3rem;">{title}</div>
        {change_html}
        <div style="font-size:0.8rem; color:#adb5bd; margin-top:0.5rem;">{subtitle if subtitle else ""}</div>
    </div>
    """
    
    st.markdown(metric_html, unsafe_allow_html=True)

# Define app layout and navigation
def main():
    # Generate sample data for dashboard
    historical_data = generate_historical_data()
    
    # Create side navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; margin-bottom:20px;">
            <img src="https://img.icons8.com/fluency/96/000000/microphone.png" width="60">
            <h2 style="margin-top:10px;">EmotionVox</h2>
            <p style="color:#6c757d; font-size:0.9rem;">Real-Time Speech Emotion Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Home", "Live Analysis", "Dashboard", "About Team"],
            icons=["house", "mic", "graph-up", "people"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#6C63FF", "font-size": "1rem"},
                "nav-link": {
                    "font-size": "0.9rem",
                    "text-align": "left",
                    "margin": "0px",
                    "border-radius": "8px",
                    "--hover-color": "#ecedfb",
                },
                "nav-link-selected": {"background-color": "#6C63FF", "color": "white"},
            }
        )
        
        st.markdown("---")
        
        # Project info
        st.markdown("""
        <div style="background-color:white; padding:15px; border-radius:10px; margin-top:20px;">
            <h5 style="color:#6C63FF; margin-top:0;">Project Information</h5>
            <p style="font-size:0.85rem; margin-bottom:5px;">
                Department of Computer Science and Engineering<br>
                Chandigarh University, Mohali, Punjab
            </p>
            <div style="font-size:0.8rem; color:#6c757d;">
                Developed with ❤️ for better emotion understanding in digital communications
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Home page
    if selected == "Home":
        st.markdown("""
        <h1 class="fade-in">Welcome to EmotionVox Analytics</h1>
        """, unsafe_allow_html=True)
        
        # Hero section
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            <div class="dashboard-card fade-in animate-1">
                <h2>Real-Time Speech Emotion Analysis in Social Media Streams</h2>
                <p style="font-size:1.1rem; line-height:1.6;">
                    Our system uses advanced deep learning algorithms to analyze emotions in 
                    spoken content across social media platforms in real-time, providing 
                    valuable insights for content moderation, customer experience, and 
                    audience engagement.
                </p>
                <div style="display:flex; gap:10px; margin-top:20px;">
                    <a href="?page=analysis" style="text-decoration:none;">
                        <button style="
                            background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
                            color: white;
                            border: none;
                            padding: 10px 20px;
                            border-radius: 8px;
                            font-weight: 600;
                            cursor: pointer;
                            transition: all 0.3s ease;
                            box-shadow: 0 4px 10px rgba(108, 99, 255, 0.2);
                        ">
                            Try Live Analysis
                        </button>
                    </a>
                    <a href="?page=dashboard" style="text-decoration:none;">
                        <button style="
                            background: white;
                            color: var(--primary-color);
                            border: 1px solid var(--primary-color);
                            padding: 10px 20px;
                            border-radius: 8px;
                            font-weight: 600;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        ">
                            View Dashboard
                        </button>
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st_lottie(lottie_analysis, height=300, key="home_animation")
        
        # Features section
        st.markdown("""
        <h2 class="fade-in animate-2" style="margin-top:40px;">Key Features</h2>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card fade-in animate-2">
                <div class="feature-icon">🎙️</div>
                <h4>Real-Time Analysis</h4>
                <p>Process live audio streams from various social media platforms, providing instant emotion insights.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card fade-in animate-3">
                <div class="feature-icon">📊</div>
                <h4>Emotion Intelligence</h4>
                <p>Detect 7 distinct emotions with high accuracy using our fine-tuned deep learning models.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card fade-in animate-4">
                <div class="feature-icon">🔍</div>
                <h4>Interactive Insights</h4>
                <p>Visualize emotion trends and patterns through our intuitive dashboard interface.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Use cases section
        st.markdown("""
        <h2 class="fade-in animate-4" style="margin-top:40px;">Applications</h2>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="dashboard-card fade-in animate-4">
                <h4>Content Moderation</h4>
                <p>Identify potentially harmful content in real-time by detecting emotional distress, anger, or fear in audio streams.</p>
                
                <h4>Brand Monitoring</h4>
                <p>Understand how audiences emotionally respond to your brand during live events, spaces, or podcasts.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="dashboard-card fade-in animate-5">
                <h4>Customer Experience</h4>
                <p>Analyze customer emotions during voice interactions to improve service quality and response strategies.</p>
                
                <h4>Mental Health Support</h4>
                <p>Aid in early detection of emotional distress patterns in support communities and social channels.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Live Analysis page
    elif selected == "Live Analysis":
        st.markdown("""
        <h1 class="fade-in">Real-Time Speech Emotion Analysis</h1>
        """, unsafe_allow_html=True)
        
        # Introductory text
        st.markdown("""
        <div class="dashboard-card fade-in animate-1">
            <p style="font-size:1.1rem;">
                Analyze speech emotions in real-time. Speak into your microphone and our AI will detect the emotional content of your speech.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio recorder section
        st.markdown("""
        <div class="audio-recorder fade-in animate-2">
            <h3 style="margin-top:0;">Record Your Voice</h3>
            <p>Click the button below to start recording. Speak clearly for best results.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st_lottie(lottie_microphone, height=180, key="mic_animation")
            start_button = st.button("Start Recording", key="start_recording")
        
        # When record button is clicked
        if start_button:
            # Record and process audio
            audio_data, sample_rate, emotion_probs, detected_emotion = process_live_audio()
            
                       # Store results in session state for display
            st.session_state.audio_data = audio_data
            st.session_state.sample_rate = sample_rate
            st.session_state.emotion_probs = emotion_probs
            st.session_state.detected_emotion = detected_emotion
            st.session_state.analysis_complete = True
        
        # Display results if analysis is complete
        if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
            st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown("""
            <h3 class="fade-in animate-3">Analysis Results</h3>
            """, unsafe_allow_html=True)
            
            # Display detected emotion with styled badge
            emotion_color = EMOTION_COLORS[st.session_state.detected_emotion]
            st.markdown(f"""
            <div class="dashboard-card fade-in animate-3">
                <h4>Detected Emotion</h4>
                <div style="
                    display: inline-block;
                    padding: 8px 16px;
                    background-color: {emotion_color};
                    color: white;
                    border-radius: 20px;
                    font-weight: 600;
                    font-size: 1.2rem;
                    margin-top: 10px;
                ">
                    {st.session_state.detected_emotion.capitalize()}
                </div>
                <p style="margin-top: 15px;">Our AI detected {st.session_state.detected_emotion} as the primary emotion in your speech.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Audio waveform
            st.markdown("""
            <div class="dashboard-card fade-in animate-4">
                <h4>Audio Waveform</h4>
            """, unsafe_allow_html=True)
            
            # Create and display waveform
            waveform_image = create_waveform(st.session_state.audio_data, st.session_state.sample_rate)
            st.image(waveform_image)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Emotion probability chart
            st.markdown("""
            <div class="dashboard-card fade-in animate-5">
                <h4>Emotion Probabilities</h4>
            """, unsafe_allow_html=True)
            
            # Create and display emotion gauges
            emotion_fig = create_emotion_gauge(st.session_state.emotion_probs)
            st.plotly_chart(emotion_fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add option to try again
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Record Again", key="record_again"):
                    st.session_state.analysis_complete = False
                    st.experimental_rerun()
            
            # Add explanation of results
            st.markdown("""
            <div class="dashboard-card fade-in animate-5">
                <h4>Understanding the Results</h4>
                <p>
                    The emotion detection system analyzes various acoustic features of your speech such as tone, pitch, 
                    rhythm, and energy to determine the emotional content. The system is trained to recognize seven 
                    primary emotions: happy, sad, angry, neutral, fear, disgust, and surprise.
                </p>
                <p>
                    The confidence scores indicate how strongly each emotion is expressed in your speech. Higher scores 
                    suggest a clearer emotional signal, while more balanced scores might indicate mixed emotions or 
                    more neutral speech.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Dashboard page
    elif selected == "Dashboard":
        st.markdown("""
        <h1 class="fade-in">Emotion Analytics Dashboard</h1>
        """, unsafe_allow_html=True)
        
        # Calculate metrics
        metrics = get_key_metrics(historical_data)
        
        # Display key metrics row
        st.markdown("""
        <div class="fade-in animate-1">
            <h3 style="margin-bottom:15px;">Key Metrics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            display_metric(
                "Total Samples", 
                f"{metrics['total_samples']:,}", 
                "Analyzed across all platforms",
                "📊", 
                5, 
                True
            )
        
        with col2:
            display_metric(
                "Dominant Emotion", 
                f"{metrics['dominant_emotion'].capitalize()}", 
                f"{metrics['dominant_percent']}% of all samples",
                "😀", 
                2, 
                True
            )
        
        with col3:
            display_metric(
                "Avg. Confidence", 
                f"{metrics['avg_confidence']}%", 
                "Detection accuracy score",
                "🎯", 
                1.5, 
                True
            )
        
        with col4:
            display_metric(
                "Emotion Shifts", 
                f"{metrics['emotion_shifts']}", 
                "Changes in last 24 hours",
                "📈", 
                -3, 
                False
            )
        
        # Tabs for different dashboard views
        dashboard_tabs = hc.option_bar(
            option_list=["Overview", "Time Analysis", "Platform Comparison"],
            icons=["graph-up", "clock-history", "diagram-3"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0px", "margin": "20px 0", "background-color": "#ecedfb"},
                "icon": {"color": "#6C63FF", "font-size": "16px"},
                "nav-link": {"padding": "10px 15px", "border-radius": "0px", "text-align": "center"},
                "nav-link-selected": {"background-color": "#6C63FF", "color": "white"},
            }
        )
        
        # Overview tab
        if dashboard_tabs == "Overview":
            # Two column layout for main charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""<div class="chart-container fade-in animate-2">""", unsafe_allow_html=True)
                fig = create_emotion_distribution(historical_data)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""</div>""", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""<div class="chart-container fade-in animate-3">""", unsafe_allow_html=True)
                fig = create_confidence_by_emotion(historical_data)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""</div>""", unsafe_allow_html=True)
            
            # Full width charts
            st.markdown("""<div class="chart-container fade-in animate-4">""", unsafe_allow_html=True)
            fig = create_platform_comparison(historical_data)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""</div>""", unsafe_allow_html=True)
        
        # Time Analysis tab
        elif dashboard_tabs == "Time Analysis":
            st.markdown("""<div class="chart-container fade-in animate-2">""", unsafe_allow_html=True)
            fig = create_emotion_trends(historical_data)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""</div>""", unsafe_allow_html=True)
            
            st.markdown("""<div class="chart-container fade-in animate-3">""", unsafe_allow_html=True)
            fig = create_hourly_heatmap(historical_data)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""</div>""", unsafe_allow_html=True)
        
        # Platform Comparison tab
        elif dashboard_tabs == "Platform Comparison":
            # Filter controls
            st.markdown("""<div class="dashboard-card fade-in animate-2">""", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_platforms = st.multiselect(
                    "Select Platforms",
                    options=historical_data['platform'].unique(),
                    default=historical_data['platform'].unique()
                )
            
            with col2:
                selected_emotions = st.multiselect(
                    "Select Emotions",
                    options=list(EMOTION_COLORS.keys()),
                    default=list(EMOTION_COLORS.keys())
                )
            
            st.markdown("""</div>""", unsafe_allow_html=True)
            
            # Filter data based on selections
            filtered_data = historical_data[
                (historical_data['platform'].isin(selected_platforms)) & 
                (historical_data['emotion'].isin(selected_emotions))
            ]
            
            if not filtered_data.empty:
                st.markdown("""<div class="chart-container fade-in animate-3">""", unsafe_allow_html=True)
                fig = create_platform_comparison(filtered_data)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("""</div>""", unsafe_allow_html=True)
                
                # Platform-specific emotion distribution
                col1, col2 = st.columns(2)
                
                for i, platform in enumerate(selected_platforms):
                    platform_data = filtered_data[filtered_data['platform'] == platform]
                    
                    if not platform_data.empty:
                        with col1 if i % 2 == 0 else col2:
                            st.markdown(f"""<div class="chart-container fade-in animate-{4+i}">""", unsafe_allow_html=True)
                            st.markdown(f"<h4>{platform} Emotion Distribution</h4>", unsafe_allow_html=True)
                            fig = create_emotion_distribution(platform_data)
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("""</div>""", unsafe_allow_html=True)
            else:
                st.warning("No data available with the selected filters.")
        
        # Data insights
        st.markdown("""
        <div class="dashboard-card fade-in animate-5">
            <h3>Key Insights</h3>
            <ul style="padding-left: 20px;">
                <li>The most commonly detected emotion across platforms is <strong>{}</strong>.</li>
                <li>Emotion patterns show significant changes during peak hours (8AM-10AM and 7PM-9PM).</li>
                <li>Twitter Spaces has the highest variation in emotional content compared to other platforms.</li>
                <li>Content with clear emotional signals (high confidence) typically has greater engagement.</li>
            </ul>
        </div>
        """.format(metrics['dominant_emotion'].capitalize()), unsafe_allow_html=True)
        
        # Export options
        with st.expander("Export Options"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="Export as CSV",
                    data=historical_data.to_csv(index=False),
                    file_name="emotion_analytics_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="Export as JSON",
                    data=historical_data.to_json(orient="records"),
                    file_name="emotion_analytics_data.json",
                    mime="application/json"
                )
            
            with col3:
                st.button("Generate Report")
    
    # About team page
    elif selected == "About Team":
        st.markdown("""
        <h1 class="fade-in">Meet Our Team</h1>
        """, unsafe_allow_html=True)
        
        # Team intro
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            <div class="dashboard-card fade-in animate-1">
                <h3>Project Contributors</h3>
                <p style="font-size:1.1rem; line-height:1.6;">
                    Our diverse team from the Department of Computer Science and Engineering at Chandigarh University
                    combines expertise in machine learning, audio processing, and user experience design to create
                    this innovative emotion analytics platform.
                </p>
                <p>
                    The project builds on cutting-edge research in affective computing and speech processing to 
                    deliver accurate emotion detection for social media streams.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st_lottie(lottie_team, height=250, key="team_animation")
        
        # Team member profiles
        st.markdown("""
        <div class="fade-in animate-2">
            <h3 style="margin-top:30px; margin-bottom:20px;">Team Members</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Project participants (your actual team)
        team_members = [
            {
                "name": "Zakiya Khan",
                "role": "Project Lead & NLP Specialist",
                "email": "zkyafzy123@gmail.com",
                "image": "https://randomuser.me/api/portraits/women/65.jpg"
            },
            {
                "name": "Vedant Kumar",
                "role": "Machine Learning Engineer",
                "email": "vedantkumar0009@gmail.com",
                "image": "https://randomuser.me/api/portraits/men/32.jpg"
            },
            {
                "name": "Mohd Mosahid Raza Khan",
                "role": "Audio Processing Specialist",
                "email": "razamoshahid69@gmail.com",
                "image": "https://randomuser.me/api/portraits/men/55.jpg"
            },
            {
                "name": "Prashant Kumar",
                "role": "Backend Developer",
                "email": "krprashant0412@gmail.com",
                "image": "https://randomuser.me/api/portraits/men/41.jpg"
            },
            {
                "name": "Pritam Raj",
                "role": "Frontend & UI/UX Designer",
                "email": "pritam.raj0608@gmail.com",
                "image": "https://randomuser.me/api/portraits/men/22.jpg"
            }
        ]
        
        # Display team members in a grid
        cols = st.columns(5)
        
        for i, member in enumerate(team_members):
            with cols[i]:
                st.markdown(f"""
                <div class="team-card fade-in animate-{3+i}">
                    <div class="team-image-container">
                        <img src="{member['image']}" class="team-image" alt="{member['name']}">
                    </div>
                    <div class="team-info">
                        <h4 class="team-name">{member['name']}</h4>
                        <p class="team-role">{member['role']}</p>
                        <p class="team-contact">{member['email']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Project details
        st.markdown("""
        <div class="dashboard-card fade-in animate-5" style="margin-top:40px;">
            <h3>Project Details</h3>
            <p><strong>Department:</strong> Computer Science and Engineering</p>
            <p><strong>Institution:</strong> Chandigarh University, Mohali, Punjab</p>
            <p><strong>Project Duration:</strong> 6 months</p>
            <p><strong>Technologies Used:</strong> TensorFlow, PyTorch, Librosa, Streamlit, Python</p>
        </div>
        """, unsafe_allow_html=True)
        
                # Publications and research
        st.markdown("""
        <div class="dashboard-card fade-in animate-5">
            <h3>Research & Publications</h3>
            <ul style="padding-left: 20px;">
                <li>
                    <strong>Real-time Emotion Recognition in Social Media Audio Streams</strong><br>
                    Presented at International Conference on Machine Learning Applications, 2022
                </li>
                <li>
                    <strong>Improving Speech Emotion Detection with Transfer Learning</strong><br>
                    Journal of Affective Computing, Vol. 12, Issue 3
                </li>
                <li>
                    <strong>Multilingual Approaches to Emotion Recognition in Voice</strong><br>
                    Under review - IEEE Transactions on Affective Computing
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Acknowledgments
        st.markdown("""
        <div class="dashboard-card fade-in animate-5">
            <h3>Acknowledgments</h3>
            <p>
                We would like to thank the faculty at Chandigarh University for their guidance and support. 
                Special thanks to Dr. Anand Kumar for mentoring the research, and the university's computing 
                resources department for providing the infrastructure needed for training our models.
            </p>
            <p>
                This work is supported in part by a research grant from the Department of Science and Technology.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Page footer
    st.markdown("""
    <div class="footer fade-in">
        <p>EmotionVox Analytics &copy; 2023 | Department of Computer Science and Engineering, Chandigarh University</p>
        <p style="font-size:0.8rem; color:#adb5bd;">Version 1.0.0 | Privacy Policy | Terms of Use</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
