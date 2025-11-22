import pandas as pd
from sklearn.model_selection import train_test_split

# Homogenized train-test split across ML models
# unique_ids = pd.read_csv("C:/Users/rlaal/Documents/GitHub/pulmonary_fibrosis/data/train.csv")["Patient"].unique()
unique_ids = pd.read_csv("C:/Coding/pulmonary_fibrosis/osic-pulmonary-fibrosis-progression/train.csv")["Patient"].unique()
train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=3244)