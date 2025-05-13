import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class CombinedMentalHealthClassifier:
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(max_features=1000)
        self.image_scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=1000)
        self.is_trained = False
    
    def train(self, text_features, image_features, labels):
        """
        Train the combined classifier
        """
        # Ensure image features are 2D
        if len(image_features.shape) == 1:
            image_features = image_features.reshape(1, -1)
        
        # Combine features
        combined_features = np.hstack([text_features, image_features])
        
        # Scale features
        combined_features = self.image_scaler.fit_transform(combined_features)
        
        # Train classifier
        self.classifier.fit(combined_features, labels)
        self.is_trained = True
    
    def predict(self, text_features, image_features):
        """
        Make predictions using both text and image features
        """
        if not self.is_trained:
            raise ValueError("Model needs to be trained before making predictions")
        
        # Ensure image features are 2D
        if len(image_features.shape) == 1:
            image_features = image_features.reshape(1, -1)
        
        # Combine features
        combined_features = np.hstack([text_features, image_features])
        
        # Scale features
        combined_features = self.image_scaler.transform(combined_features)
        
        # Make prediction
        prediction = self.classifier.predict(combined_features)
        probabilities = self.classifier.predict_proba(combined_features)
        
        return prediction[0], probabilities[0]
    
    def save_model(self, path="combined_model.pkl"):
        """
        Save the trained model
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'classifier': self.classifier,
            'image_scaler': self.image_scaler,
            'text_vectorizer': self.text_vectorizer,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, path="combined_model.pkl"):
        """
        Load a trained model
        """
        if not os.path.exists(path):
            return cls()
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls()
        model.classifier = model_data['classifier']
        model.image_scaler = model_data['image_scaler']
        model.text_vectorizer = model_data['text_vectorizer']
        model.is_trained = model_data['is_trained']
        
        return model 