import streamlit as st
import pickle
import os
import numpy as np
from utils.preprocessor import clean_text
from models.image_model import ImageFeatureExtractor
from models.combined_model import CombinedMentalHealthClassifier
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
from PIL import Image
import io
import asyncio
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="streamlit.util")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Mental Health Analyzer",
    page_icon="🧠",
    layout="centered"
)

# Initialize feature extractors and model
@st.cache_resource
def load_models():
    try:
        image_extractor = ImageFeatureExtractor()
        combined_model = CombinedMentalHealthClassifier.load_model()
        return image_extractor, combined_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

image_extractor, combined_model = load_models()

# Function to map mental health status to emoji
def status_to_emoji(status):
    return {
        "healthy": "😊",
        "at_risk": "⚠️",
        "needs_help": "🆘"
    }.get(status.lower(), "🤔")

# Main app
def main():
    st.markdown("<h1 style='text-align: center;'>🧠 Mental Health Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Analyze your posts and images to understand mental health indicators!</p>", unsafe_allow_html=True)
    st.markdown("---")

    if image_extractor is None or combined_model is None:
        st.error("Failed to load models. Please check if the model files exist and try again.")
        return

    # Input area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown("### 💬 Enter a Caption")
        user_input = st.text_area("What's on your mind?", height=100, 
                                placeholder="E.g. Feeling overwhelmed with work today...")

    # Analysis button
    if st.button("📈 Analyze"):
        if uploaded_file is not None and user_input.strip():
            try:
                with st.spinner("Analyzing..."):
                    # Process image
                    image_features = image_extractor.extract_features(image)
                    
                    # Process text
                    cleaned_text = clean_text(user_input)
                    text_features = combined_model.text_vectorizer.transform([cleaned_text]).toarray()
                    
                    # Get prediction
                    prediction, probabilities = combined_model.predict(text_features, image_features)
                    
                    # Display prediction with emoji
                    emoji = status_to_emoji(prediction)
                    st.success(f"🎯 **Predicted Status:** {emoji} `{prediction.upper()}`")
                    
                    # Bar chart with Altair
                    st.subheader("🔢 Probability Distribution")
                    prob_df = pd.DataFrame({
                        "Status": combined_model.classifier.classes_,
                        "Probability": np.round(probabilities * 100, 2)
                    })
                    
                    chart = alt.Chart(prob_df).mark_bar(color="#4c78a8").encode(
                        x=alt.X("Status", sort="-y"),
                        y="Probability",
                        tooltip=["Status", "Probability"]
                    ).properties(
                        width=500,
                        height=300
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")
                st.error("Please make sure you have trained the model first by running the training script.")
        else:
            st.warning("⚠️ Please upload an image and enter text to analyze.")

    st.markdown("---")

if __name__ == "__main__":
    main()
