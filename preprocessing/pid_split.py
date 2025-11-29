import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Homogenized train-test split across ML models
BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR.parent / "data" / "train.csv"
test_csv_path = BASE_DIR.parent / "data" / "test.csv"

unique_ids = pd.read_csv(csv_path)["Patient"].unique()
test_ids = pd.read_csv(test_csv_path)["Patient"].unique()
unique_ids = [pid for pid in unique_ids if pid not in test_ids]
train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=3244)