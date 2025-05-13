import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from models.image_model import ImageFeatureExtractor
from models.combined_model import CombinedMentalHealthClassifier
import pandas as pd
import os
from utils.preprocessor import clean_text

def prepare_sample_data():
    """
    Create sample data for training
    Returns: (texts, image_paths, labels)
    """
    # Sample data - in a real application, this would come from a proper dataset
    texts = [
        "Feeling great today, went for a morning run!",
        "Having trouble sleeping lately, feeling anxious",
        "Really struggling with work pressure",
        "Enjoying time with friends and family",
        "Feeling isolated and alone",
        "Making progress on my goals",
        "Can't seem to get out of bed",
        "Excited about new opportunities",
        "Everything feels overwhelming",
        "Finding joy in small moments"
    ]
    
    # Create a sample images directory if it doesn't exist
    os.makedirs("sample_images", exist_ok=True)
    
    # For demo purposes, we'll use placeholder image paths
    # In a real application, you would have actual images
    image_paths = [f"sample_images/sample_{i}.jpg" for i in range(len(texts))]
    
    # Labels: healthy, at_risk, needs_help
    labels = ["healthy", "at_risk", "needs_help", "healthy", "needs_help",
              "healthy", "needs_help", "healthy", "at_risk", "healthy"]
    
    return texts, image_paths, labels

def train_and_evaluate():
    """
    Train the combined model and evaluate its performance
    """
    # Initialize feature extractors
    image_extractor = ImageFeatureExtractor()
    text_vectorizer = TfidfVectorizer(max_features=1000)
    
    # Prepare data
    texts, image_paths, labels = prepare_sample_data()
    
    # Process text features
    cleaned_texts = [clean_text(text) for text in texts]
    text_features = text_vectorizer.fit_transform(cleaned_texts).toarray()
    
    # Process image features
    # Note: In a real application, you would process actual images
    # For demo purposes, we'll create random features
    image_features = np.random.randn(len(texts), 512)  # ResNet18 features are 512-dimensional
    
    # Initialize and train the combined model
    model = CombinedMentalHealthClassifier()
    model.text_vectorizer = text_vectorizer
    model.train(text_features, image_features, labels)
    
    # Save the trained model
    model.save_model()
    
    return model

if __name__ == "__main__":
    train_and_evaluate() 