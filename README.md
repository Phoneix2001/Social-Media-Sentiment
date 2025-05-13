# Mental Health Analyzer

A Streamlit application that analyzes both images and text to assess mental health indicators. The application uses a combination of computer vision (ResNet18) and natural language processing (TF-IDF) to provide insights into mental health status.

## Features

- Image upload and analysis
- Text caption analysis
- Combined image and text feature extraction
- Mental health status prediction
- Visual probability distribution
- User-friendly interface

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mental-health-analyzer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
mental-health-analyzer/
├── app.py                 # Main Streamlit application
├── models/
│   ├── image_model.py     # Image feature extraction
│   ├── combined_model.py  # Combined classifier
│   └── train_model.py     # Training script
├── utils/
│   └── preprocessor.py    # Text preprocessing utilities
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Model Architecture

The application uses a combination of:
- ResNet18 for image feature extraction
- TF-IDF vectorization for text features
- Logistic Regression for final classification

## Usage

1. Upload an image using the file uploader
2. Enter a caption describing your current state or thoughts
3. Click "Analyze" to get the prediction
4. View the probability distribution for different mental health statuses

## Note

This application is for educational and research purposes only. It should not be used as a substitute for professional mental health advice or diagnosis.

## License

MIT License # Social-Media-Sentiment
