'''
Joint_fusion model script that uses trained model (from .ipynb version) 
to output validation and test results.
For full analysis of validation/test results, refer to notebook version

Final variables:
- jf_validation_results
- jf_test_results
'''

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import pandas as pd
import random
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

random.seed(3244)

# --------------------------- #
# ------- DATA PREP --------- #
# --------------------------- # 

## PART 1: Image data

# Process dicom ct-scans into npy files
from preprocessing.image_preprocessing import preprocess_scans

scans_dir_pkl = "data/preprocessed_scans.pkl"
if not os.path.exists(scans_dir_pkl):
    print("Starting image preprocessing for the first time...")
    preprocessed_scans = preprocess_scans("data")
    pickle.dump(preprocessed_scans, open(scans_dir_pkl, 'wb'))

# load scan data if already preprocessed before
print("Loading in preprocessed image scans...")
preprocessed_scans = pickle.load(open(scans_dir_pkl, 'rb'))

# function to prepare preprocessed scans into standard volumes for 3D ResNet/CNN encoder
def build_image_volumes(preprocessed_scans):
    scan_data = {}
    for patientID, volume in preprocessed_scans.items():
        dimensions = volume.shape
        if dimensions[0] == 0:  # exclude patient scans with 0 depth
            continue
        else:
            # change from (D,1,256,256) -> (1,D,256,256)
            volume = volume.transpose(1, 0, 2, 3)
            volume = resample_depth_with_padding(volume)
            scan_data[patientID] = volume
    return scan_data


# helper: resample to fixed depth per scan volume
def resample_depth_with_padding(volume, target_d=128):
    # from volume: (1, D, H, W) -> (1, target_d, H, W)
    C, D, H, W = volume.shape
    if D >= target_d:
        # downsample evenly
        indices = np.linspace(0, D-1, target_d).astype(int)
        return volume[:, indices, :, :]
    else:
        # pad before and after along depth axis
        pad_before = (target_d - D) // 2
        pad_after = target_d - D - pad_before
        padded = np.pad(volume, ((0,0),(pad_before,pad_after),(0,0),(0,0)), mode='constant', constant_values=0)
        return padded

new_scans = build_image_volumes(preprocessed_scans)

## PART 2: Tabular data

from preprocessing.tabular_preprocessing import train_df, val_df, time_scaler, fvc_scaler
# create smoking status encoder by fitting on training dataframe
smoking_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
smoking_encoder.fit(train_df[["SmokingStatus"]])

# function to prepare tabular data for LSTM encoder
def prepare_tabular_dataset(dataframe):
    # encode Sex
    dataframe["Sex_is_male"] = (dataframe["Sex"] == "Male").astype(int)
    # one-hot encode smoking status
    smoking_encoded = smoking_encoder.transform(train_df[["SmokingStatus"]])
    smoking_dummies = pd.DataFrame(
        smoking_encoded, 
        columns=smoking_encoder.get_feature_names_out(['SmokingStatus'])
    )
    # join smoking status df back to original df
    dataframe = pd.concat([dataframe, smoking_dummies], axis=1)
    # choose X features
    feature_cols = [
        "Weeks_scaled",
        "Age",
        "Sex_is_male",
        "SmokingStatus_Never smoked",
        "SmokingStatus_Ex-smoker",
        "SmokingStatus_Currently smokes",
        "Baseline_FVC"
    ]
    # this is Y target
    target_col = "FVC_scaled"

    # group data by patient ID
    patient_features = {}
    for pid, df in dataframe.groupby("Patient"):
        df = df.sort_values("Weeks")   # ensure time-series order
        
        features = df[feature_cols].values.astype(np.float32)  # shape = (T, F)
        targets = df[target_col].values.astype(np.float32)     # shape = (T,)
        # store structured patient data
        patient_features[pid] = {
            "features": features,
            "targets": targets,
            "times": df["Weeks_scaled"].values.astype(np.float32)
        }
    return patient_features


X_tabular_train = prepare_tabular_dataset(train_df)
X_tabular_val = prepare_tabular_dataset(val_df)


X_tabular_train = prepare_tabular_dataset(train_df)
X_tabular_val = prepare_tabular_dataset(val_df)

## PART 3: Data Loader

X_tabular_train = prepare_tabular_dataset(train_df)
X_tabular_val = prepare_tabular_dataset(val_df)


# 1. Extract patient IDs from tabular train/val sets
train_patient_ids = set(train_df["Patient"].unique())
val_patient_ids   = set(val_df["Patient"].unique())

# 2. Remove patients whose images are removed due to 0 slices/depth
valid_ids = set(new_scans.keys())   # built image volumes

train_patient_ids = train_patient_ids & valid_ids
val_patient_ids   = val_patient_ids & valid_ids

# 3. Build image splits
X_image_train = {pid: new_scans[pid] for pid in train_patient_ids}
X_image_val   = {pid: new_scans[pid] for pid in val_patient_ids}

# 4. Ensure tabular splits
X_tabular_train = {pid: X_tabular_train[pid] for pid in train_patient_ids}
X_tabular_val = {pid: X_tabular_val[pid] for pid in val_patient_ids}

# custom Dataset class for Joint-level-fusion model
class JointFusion_Dataset(Dataset):
    def __init__(self, patient_ids, tabular_dict, image_dict):
            self.patient_ids = list(patient_ids)
            self.tabular_dict = tabular_dict
            self.image_dict = image_dict

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # for tabular data
        tab = self.tabular_dict[pid]["features"]      # shape (T, F)
        target = self.tabular_dict[pid]["targets"]     # shape (T,)    
        # convert to tensor
        tab = torch.tensor(tab, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # for image data (same for all timesteps, since we only have base CT-scan)
        img = self.image_dict[pid]       # numpy array (1, 64, 256, 256)
        img = torch.tensor(img, dtype=torch.float32)

        return {
            "patient_id": pid,
            "tabular": tab,
            "image": img,
            "target": target,
            "seq_len": tab.size(0)
        }

# custom collate function to pad tabular time series into equal sizes 
def collate_fn(batch):
    tabs = [item["tabular"] for item in batch]
    imgs = torch.stack([item["image"] for item in batch])  # images already fixed size
    targets = [item["target"] for item in batch]
    seq_lens = torch.tensor([len(t) for t in tabs], dtype=torch.long)

    # Pad tabular sequences and targets
    tabs_padded = pad_sequence(tabs, batch_first=True, padding_value=0.0)       # (B, T_max, F)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0.0) # (B, T_max)

    # Create mask for loss calculation: 1 for real timesteps, 0 for padded
    mask = torch.zeros_like(targets_padded, dtype=torch.float32)
    for i, l in enumerate(seq_lens):
        mask[i, :l] = 1.0

    return {
        "tabular": tabs_padded,
        "image": imgs,
        "target": targets_padded,
        "seq_len": seq_lens,
        "mask": mask
    }

# build dataloaders
train_dataset = JointFusion_Dataset(
    patient_ids=train_patient_ids,
    tabular_dict=X_tabular_train,
    image_dict=X_image_train
)
val_dataset = JointFusion_Dataset(
    patient_ids=val_patient_ids,
    tabular_dict=X_tabular_val,
    image_dict=X_image_val
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# --------------------------- #
# --------- MODELS ---------- #
# --------------------------- # 

# ---------------------------------
# 2A. LSTM encoder for tabular data
# ---------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=64, num_layers=2, out_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: (B, T, in_dim)
        out, _ = self.lstm(x)       # out: (B, T, hidden_dim)
        out = self.fc(out)          # (B, T, out_dim)
        return out


# -------------------------------------------
# 2B. ResNet3D encoder for image data
# -------------------------------------------
# 3D basic residual block for ResNet3D
class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # projection if channel count or stride changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# Actual ResNet3D encoder
class ResNet3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = BasicBlock3D(1, 16, stride=2)   # output: 16×32×128×128
        self.layer2 = BasicBlock3D(16, 32, stride=2)  # output: 32×16×64×64
        self.layer3 = BasicBlock3D(32, 64, stride=2)  # output: 64×8×32×32
        self.layer4 = BasicBlock3D(64, 128, stride=2) # output: 128×4×16×16

        self.pool = nn.AdaptiveAvgPool3d(1)  # -> 128×1×1×1
        self.fc = nn.Linear(128, 64)         # Final embedding

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)    # (B, 128)
        x = self.fc(x)               # (B, 64)
        return x

# --------------------------------------------------------
# 2C. Fusion RNN Model that concatenates output 
#    from previous two model to be trained on
# --------------------------------------------------------
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.tabular_encoder = LSTMEncoder(
            in_dim=7,          # tabular features per time step
            hidden_dim=64,
            num_layers=2,
            out_dim=64,
        )  
        self.image_encoder = ResNet3DEncoder()  # unchanged (B,64)

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)    # 2 -> output is shape: (mean, sigma)
        )

    def forward(self, tabular, image):
        tab_feat = self.tabular_encoder(tabular)  # (B, T, 64)
        img_feat = self.image_encoder(image)      # (B, 64)

        # Broadcast image features across T
        img_feat_exp = img_feat.unsqueeze(1).expand(-1, tab_feat.size(1), -1)  # (B, T, 64)
        fused = torch.cat([tab_feat, img_feat_exp], dim=2)  # (B, T, 128)
        params = self.fusion(fused)  # (B, T, 2)

        mu = params[..., 0]                # (B, T)
        sigma_raw = params[..., 1]             # (B, T)
        sigma = F.softplus(sigma_raw) + 1e-6  # ensure positivity

        return mu, sigma

# --------------------------- #
# ------- INFERENCE --------- #
# --------------------------- # 

# Load the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/final_jointFusion_model.pth"
model = FusionModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
print(f"Loaded final jointFusion_model on device: {DEVICE}")

# function to get predictions on validation set
def get_predictions(model, val_loader, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_pred_list = []
    y_true_list = []
    y_sigma_list = []
    with torch.no_grad():
        for batch in val_loader:
            tab = batch["tabular"].to(device)
            img = batch["image"].to(device)
            target = batch["target"].to(device)
            seq_len = batch["seq_len"].to(device)
            # Forward pass
            mu_pred, sigma_pred = model(tab, img)  # (B, T_max)
            # Mask padded timesteps
            mask = torch.arange(target.size(1), device=device)[None, :] < seq_len[:, None]
            # Only keep valid timesteps
            for i in range(mu_pred.size(0)):
                y_pred_list.append(mu_pred[i, :seq_len[i]].cpu().numpy())
                y_true_list.append(target[i, :seq_len[i]].cpu().numpy())
                y_sigma_list.append(sigma_pred[i, :seq_len[i]].cpu().numpy())

    y_pred_list = np.concatenate(y_pred_list)
    y_true_list = np.concatenate(y_true_list)
    y_sigma_list = np.concatenate(y_sigma_list)
    y_pred_list = fvc_scaler.inverse_transform(y_pred_list.reshape(-1, 1)).flatten()
    y_true_list = fvc_scaler.inverse_transform(y_true_list.reshape(-1, 1)).flatten()
    # Important: sigma only scales, does NOT shift
    y_sigma_list = y_sigma_list * fvc_scaler.scale_[0]
    
    return y_pred_list, y_true_list, y_sigma_list

print("Computing results on validation set...")
y_pred_list, y_true_list, y_sigma_list = get_predictions(model, val_loader)


# create a dict object for each validation patient and their respective pred, true, weeks, standard deviation values
jf_validation_results = {}
counter = 0
for id, item in X_tabular_val.items():
    prev = counter
    duration = len(item['times'])
    weeks_scaled = item['times']
    weeks = time_scaler.inverse_transform(weeks_scaled.reshape(-1, 1)).flatten()
    counter += duration
    curr_patient_pred = y_pred_list[prev: counter]
    curr_patient_true = y_true_list[prev: counter]
    curr_patient_sigma = y_sigma_list[prev: counter]
    jf_validation_results[id] = {
        "weeks": weeks, 
        "y_pred": curr_patient_pred, 
        "y_true": curr_patient_true,
        "y_sigma": curr_patient_sigma
        }
    
print("Results computed!")
    

# --------------------------- #
# ------- Test data --------- #
# --------------------------- # 

# Read in and prepare test data for tabular
test_path = "data/test.csv"
sample_path = "data/sample_submission.csv"

test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)  

testData = []
patient_Week = np.array(list(sample['Patient_Week'].apply(lambda x: x.split('_')).values))

## retrieve baseline measure for current patient
for p in np.unique(patient_Week[:, 0]):
    currPatient = test[test['Patient'] == p]
    firstMeasure = list(currPatient.iloc[0, :].values)
    ## for current patient, retrieve their subsequent measures in later weeks (no FVC since we need to predict that)
    for week in patient_Week[patient_Week[:, 0] == p, 1]:
        testDataPoint = firstMeasure + [week]
        testData.append(testDataPoint)
testData = pd.DataFrame(testData)
testData.columns = ['Patient', 'StartWeeks', 'FVC', 'start_Percent', 'Age', 'Sex', 'SmokingStatus'] + ['Weeks']
testData.drop(columns = ['start_Percent', 'StartWeeks'], inplace = True)
## scale weeks and baseline fvc
testData["Weeks_scaled"] = time_scaler.transform(testData[["Weeks"]])
testData["Baseline_FVC"] = fvc_scaler.transform(testData[["FVC"]])
testData["FVC_scaled"] = fvc_scaler.transform(testData[["FVC"]])    # this is just a placeholder, not actual targets

test_patient_ids = set(testData["Patient"].unique())

# Preprocess test images the same way as training images
from preprocessing.image_preprocessing import get_preprocessed_scan

def preprocess_test_scans(data_path: str) -> dict[str, np.ndarray]:
    preprocessed_scans = dict()
    for patient_id in os.listdir(os.path.join(data_path, 'test')):
        patient_scans = []
        for scan_idx in range(1, len(os.listdir(os.path.join(data_path, 'test', patient_id))) + 1):
            scan = get_preprocessed_scan(data_path, patient_id, scan_idx)
            if scan is not None:
                patient_scans.append(scan)
        patient_scans = np.array(patient_scans, dtype=np.float32)
        preprocessed_scans[patient_id] = patient_scans
    return preprocessed_scans

# load test scan data
test_scans_dir_pkl = "data/preprocessed_test_scans.pkl"
preprocessed_test_scans = pickle.load(open(test_scans_dir_pkl, 'rb'))

test_scans = build_image_volumes(preprocessed_test_scans)
X_image_test = {pid: test_scans[pid] for pid in test_patient_ids}
X_tabular_test = prepare_tabular_dataset(testData)

# finally, create test loader
test_dataset = JointFusion_Dataset(
    patient_ids=test_patient_ids,
    tabular_dict=X_tabular_test,
    image_dict=X_image_test
)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

# retrieve test results
test_pred_list, _, test_sigma_list = get_predictions(model, test_loader)

# create a dict object for each test patient and their respective pred, true, weeks, standard deviation values
jf_test_results = {}
counter = 0
for id, item in X_tabular_test.items():
    prev = counter
    duration = len(item['times'])
    weeks_scaled = item['times']
    weeks = time_scaler.inverse_transform(weeks_scaled.reshape(-1, 1)).flatten()
    counter += duration
    curr_patient_pred = test_pred_list[prev: counter]
    curr_patient_sigma = test_sigma_list[prev: counter]
    jf_test_results[id] = {
        "weeks": weeks, 
        "test_pred": curr_patient_pred, 
        "test_sigma": curr_patient_sigma
        }


