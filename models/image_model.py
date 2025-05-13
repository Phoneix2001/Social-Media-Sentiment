import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageFeatureExtractor:
    def __init__(self):
        # Load pretrained ResNet model with updated syntax
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the last layer to get features
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image):
        """
        Extract features from an image using ResNet
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Transform image
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
        
        # Flatten features and ensure 2D array
        features = features.squeeze().numpy()
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        return features 