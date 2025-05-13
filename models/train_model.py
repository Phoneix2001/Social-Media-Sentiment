import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from models.image_model import ImageFeatureExtractor
from models.combined_model import CombinedMentalHealthClassifier
from PIL import Image
import pickle

from utils.preprocessor import clean_text
from data.download_kaggle_data import download_kaggle_dataset

def train_and_evaluate():
    """
    Train the combined model using sample data
    """
    # Initialize feature extractors
    image_extractor = ImageFeatureExtractor()
    text_vectorizer = TfidfVectorizer(max_features=1000)
    
    # Sample data (you would replace this with your actual dataset)
    sample_data = {
        'text': [
            "Feeling great today!",
            "Not sure if I can handle this anymore",
            "Just another day",
            "Everything seems so dark",
            "Excited about the future",
            "Can't stop thinking about negative things",
            "Feeling overwhelmed with work",
            "Life is beautiful",
            "Don't know what to do with my life",
            "Making progress every day"
        ],
        'image_paths': [
            'sample_images/happy.jpg',
            'sample_images/sad.jpg',
            'sample_images/neutral.jpg',
            'sample_images/depressed.jpg',
            'sample_images/excited.jpg',
            'sample_images/anxious.jpg',
            'sample_images/stressed.jpg',
            'sample_images/joyful.jpg',
            'sample_images/lost.jpg',
            'sample_images/progress.jpg'
        ],
        'labels': [
            'healthy',
            'needs_help',
            'healthy',
            'needs_help',
            'healthy',
            'at_risk',
            'at_risk',
            'healthy',
            'at_risk',
            'healthy'
        ]
    }
    
    # Extract text features
    text_features = text_vectorizer.fit_transform(sample_data['text']).toarray()
    
    # Extract image features
    image_features = []
    for img_path in sample_data['image_paths']:
        if os.path.exists(img_path):
            features = image_extractor.extract_features(img_path)
            image_features.append(features)
        else:
            # If image doesn't exist, use zeros as placeholder
            image_features.append(np.zeros(512))  # ResNet18 feature size
    
    image_features = np.array(image_features)
    
    # Initialize and train combined model
    combined_model = CombinedMentalHealthClassifier()
    combined_model.text_vectorizer = text_vectorizer
    combined_model.train(text_features, image_features, sample_data['labels'])
    
    # Save the model
    combined_model.save_model()
    
    # Save the text vectorizer separately
    with open('text_vectorizer.pkl', 'wb') as f:
        pickle.dump(text_vectorizer, f)
    
    return combined_model

if __name__ == "__main__":
    train_and_evaluate() 