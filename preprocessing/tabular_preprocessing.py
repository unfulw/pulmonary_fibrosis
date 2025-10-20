import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Preprocessing for row-coerced tabular models

BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR.parent / "data" / "train.csv"

df = pd.read_csv(csv_path)

rows = []
for pid, g in df.groupby("Patient", sort=False):
    g = g.reset_index(drop=True)
    baseline = g.iloc[0]
    pre_last = g.iloc[-2]
    last = g.iloc[-1]

    feature_row = {
        "Patient": pid,
        "Age": baseline["Age"],               
        "Sex": baseline["Sex"],
        "SmokingStatus": baseline["SmokingStatus"],
        "dFVC": pre_last["FVC"] - baseline["FVC"],
        "dPercent": pre_last["Percent"] - baseline["Percent"],
        "dweeks": pre_last["Weeks"] - baseline["Weeks"],
        "weeks_next": last["Weeks"] - pre_last["Weeks"], 
    }                                       # Columns of features
    feature_row["y_FVC_last"] = last["FVC"] # The label to predict

    rows.append(feature_row)

collapsed = pd.DataFrame(rows)
collapsed = collapsed.reset_index(drop=True)

# X = collapsed.drop(columns=["dPercent", "y_FVC_last", "Patient"]) #dPercent removed to eliminate multicollinearity
# y = collapsed["y_FVC_last"]

######################################################################################################################

# Preprocessing for gaussian process models

ids = df['Patient'].unique()
train_ids, val_ids = train_test_split(ids, test_size=0.2, random_state=3244)

train_df = df[df['Patient'].isin(train_ids)].reset_index(drop=True)
val_df   = df[df['Patient'].isin(val_ids)].reset_index(drop=True)

time_scaler = StandardScaler()
fvc_scaler = StandardScaler()
train_df["Weeks_scaled"] = time_scaler.fit_transform(train_df[["Weeks"]])
train_df["FVC_scaled"] = fvc_scaler.fit_transform(train_df[["FVC"]])
val_df["Weeks_scaled"] = time_scaler.transform(val_df[["Weeks"]])
val_df["FVC_scaled"] = fvc_scaler.transform(val_df[["FVC"]])
train_df = train_df.sort_values(["Patient", "Weeks"]).reset_index(drop=True)

#tidying validation dataframe (collapsing rows)
first_week = val_df.loc[val_df.groupby("Patient")["Weeks"].idxmin(), ["Patient", "Age", "Sex", "SmokingStatus", "Weeks_scaled", "FVC_scaled"]]
last_week  = val_df.loc[val_df.groupby("Patient")["Weeks"].idxmax(), ["Patient", "Weeks_scaled"]]
first_week = first_week.sort_values("Patient").reset_index(drop=True)
last_week  = last_week.sort_values("Patient").reset_index(drop=True)
dweek = last_week["Weeks_scaled"].values - first_week["Weeks_scaled"].values

val_df = val_df.loc[val_df.groupby("Patient")["Weeks_scaled"].idxmin(), ["Patient", "Age", "Sex", "SmokingStatus", "Weeks_scaled", "FVC_scaled"]]
val_df["Weeks_scaled"] = dweek
val_df = val_df.sort_values(["Patient"]).reset_index(drop=True)

