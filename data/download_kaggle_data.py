import kagglehub
import pandas as pd

def download_kaggle_dataset():
    # Download the dataset
    path = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")
    # Read the CSV file
    print(path)
    df = pd.read_csv(f"{path}/Combined Data.csv")
    return df

if __name__ == "__main__":
    df = download_kaggle_dataset()
    print("First 5 records:", df.head())

