import sys
import os
import math
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Any
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle


# Add preprocessing module to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "preprocessing", "scan")))
from preprocess import preprocess_scans, get_preprocessed_scan

random.seed(42)

data_dir = r'C:\Users\rlaal\Documents\NUS\AY2526S1\CS3244\Project\osic-pulmonary-fibrosis-progression'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if not os.path.exists(f'{data_dir}/preprocessed_scans.pkl'):
    preprocessed_scans = preprocess_scans(data_dir)
    pickle.dump(preprocessed_scans, open(f'{data_dir}/preprocessed_scans.pkl', 'wb'))

# Input: List of patient_id
# Output: Tuple of (train_patient_ids, val_patient_ids)
# train_ratio: Ratio of training set
def train_val_split(patients: pd.DataFrame, train_ratio: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
  patient_ids = patients['Patient'].unique()
  random.shuffle(patient_ids)

  train_patients = patients[patients['Patient'].isin(patient_ids[:int(len(patient_ids) * train_ratio)])]
  val_patients = patients[patients['Patient'].isin(patient_ids[int(len(patient_ids) * train_ratio):])]

  return train_patients, val_patients

# Prepare train and val data
train_datas = pd.read_csv(data_dir + '/train.csv')
test_datas = pd.read_csv(data_dir + '/test.csv')

test_patient_ids = test_datas['Patient'].unique()

# Remove row in train data if patient_id is in test_patient_ids
test_datas = train_datas[train_datas['Patient'].isin(test_patient_ids)]
train_datas = train_datas[~train_datas['Patient'].isin(test_patient_ids)]

# Sort df by patient and then by 'Weeks'
train_datas = train_datas.sort_values(by=['Patient', 'Weeks'])

# Group by patient and get the first FVC value and weeks
patient_id_to_initial_FVC = train_datas.groupby('Patient')['FVC'].first().to_dict()
patient_id_to_initial_weeks = train_datas.groupby('Patient')['Weeks'].first().to_dict()

train_datas, val_datas = train_val_split(train_datas)

# Count the number of scans for each patient
scan_count = {}
test_scan_count = {}
for patient_id in os.listdir(os.path.join(data_dir, 'train')):
    scan_count[patient_id] = len(os.listdir(os.path.join(data_dir, 'train', patient_id)))
    
    if patient_id in test_patient_ids:
        test_scan_count[patient_id] = len(os.listdir(os.path.join(data_dir, 'test', patient_id)))

test_patient_id_to_initial_FVC = test_datas.groupby('Patient')['FVC'].first().to_dict()
test_patient_id_to_initial_weeks = test_datas.groupby('Patient')['Weeks'].first().to_dict()

# Calculate normalization statistics for tabular features
all_weeks = train_datas['Weeks'].values
all_initial_fvc = train_datas.groupby('Patient')['FVC'].first().values
all_initial_weeks = train_datas.groupby('Patient')['Weeks'].first().values

tabular_stats = {
    'weeks_mean': float(np.mean(all_weeks)),
    'weeks_std': float(np.std(all_weeks)),
    'initial_fvc_mean': float(np.mean(all_initial_fvc)),
    'initial_fvc_std': float(np.std(all_initial_fvc)),
    'initial_weeks_mean': float(np.mean(all_initial_weeks)),
    'initial_weeks_std': float(np.std(all_initial_weeks))
}

print(f"Tabular normalization stats: {tabular_stats}")

train_x, train_y = defaultdict(list), defaultdict(list)

for idx, row in train_datas.iterrows():
    train_x[row['Patient']].append({
        'Weeks': row['Weeks'],
        'initial_FVC': patient_id_to_initial_FVC[row['Patient']],
        'initial_weeks': patient_id_to_initial_weeks[row['Patient']],
    })
    train_y[row['Patient']].append(row['FVC'])

val_x, val_y = defaultdict(list), defaultdict(list)

for idx, row in val_datas.iterrows():
    val_x[row['Patient']].append({
        'Weeks': row['Weeks'],
        'initial_FVC': patient_id_to_initial_FVC[row['Patient']],
        'initial_weeks': patient_id_to_initial_weeks[row['Patient']],
    })
    val_y[row['Patient']].append(row['FVC'])

test_x, test_y = defaultdict(list), defaultdict(list)

for idx, row in test_datas.iterrows():
    test_x[row['Patient']].append({
        'Weeks': row['Weeks'],
        'initial_FVC': test_patient_id_to_initial_FVC[row['Patient']],
        'initial_weeks': test_patient_id_to_initial_weeks[row['Patient']],
    })
    test_y[row['Patient']].append(row['FVC'])

# window = 0 to remove smoothing
def plot_loss(training_loss, val_loss, window=20):
    # Use sliding window to smooth the loss
    training_loss = [sum(training_loss[i:i+window]) / window for i in range(len(training_loss)-window)]
    val_loss = [sum(val_loss[i:i+window]) / window for i in range(len(val_loss)-window)]

    # Create one row, two col subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].plot(range(len(training_loss)), [math.log(x) for x in training_loss], label='Training Loss')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss')
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(range(len(val_loss)), [math.log(x) for x in val_loss], label='Validation Loss', color='orange')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Validation Loss')
    axs[1].legend()



def get_scans(patient_id: str, patient_scan_count: int, scan_batch_size: int) -> torch.Tensor:
    skip_size = max(1, round(patient_scan_count / scan_batch_size))
    scans = []
    for j in range(1, patient_scan_count + 1, skip_size):
        scan = get_preprocessed_scan(data_dir, patient_id, j)
        if scan is None:
            continue
        scan = torch.tensor(scan, dtype=torch.float32, device=device)
        scans.append(scan)
    scans = torch.stack(scans)
    return scans



def test_model(cnn_model, fc_model, scan_batch_size=64):
    cnn_model.to(device)
    fc_model.to(device)

    cnn_model.eval()
    fc_model.eval()

    test_predictions = []
    test_target = []

    for patient_id in tqdm(test_patient_ids):
        patient_scan_count = test_scan_count[patient_id]
        scans = get_scans(patient_id, patient_scan_count, scan_batch_size)
        features = cnn_model.forward(scans)
        features = torch.mean(features, dim=0) # 1024,

        x = test_x[patient_id]
        y = test_y[patient_id]

        for i in range(len(x)):
            weeks = torch.tensor(x[i]['Weeks'], dtype=torch.float32, device=device).unsqueeze(0)
            initial_FVC = torch.tensor(x[i]['initial_FVC'], dtype=torch.float32, device=device).unsqueeze(0)
            initial_FVC_weeks = torch.tensor(x[i]['initial_weeks'], dtype=torch.float32, device=device).unsqueeze(0)

            output = fc_model.forward(features, weeks, initial_FVC, initial_FVC_weeks).squeeze()
            test_predictions.append(output.item())
            test_target.append(y[i])

        del x, y, features, scans, weeks, initial_FVC, initial_FVC_weeks, output
        torch.cuda.empty_cache()
    
    return torch.tensor(test_predictions), torch.tensor(test_target)

# Quick visualization
def plot_test_results(targets, predictions, losses):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Scatter
    ax[0].scatter(targets, predictions, alpha=0.5)
    ax[0].plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    ax[0].set_xlabel('Target')
    ax[0].set_ylabel('Prediction')
    ax[0].set_title('Predictions vs Targets')

    # Right: Loss
    ax[1].plot(losses)
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Loss')
    ax[1].set_yscale('log')
    ax[1].set_title('Loss per Sample')

    plt.tight_layout()
    plt.show()


def plot_patient_predict_samples(cnn_model, fc_model):
    cnn_model.eval()
    fc_model.eval()

    fig, axes = plt.subplots(2, 2, figsize=(10, 5))

    for patient_order, patient in enumerate(list(train_x.keys())[10:14]):
        scans = get_scans(patient, scan_count[patient], 64)
        features = torch.mean(cnn_model.forward(scans), dim=0)
        x = train_x[patient]
        y = train_y[patient]

        predictions = []
        for i in range(len(x)):
            weeks = torch.tensor(x[i]['Weeks'], dtype=torch.float32, device=device).unsqueeze(0)
            initial_FVC = torch.tensor(x[i]['initial_FVC'], dtype=torch.float32, device=device).unsqueeze(0)
            initial_FVC_weeks = torch.tensor(x[i]['initial_weeks'], dtype=torch.float32, device=device).unsqueeze(0)
            output = fc_model.forward(features, weeks, initial_FVC, initial_FVC_weeks).squeeze()
            predictions.append(output.item())

        # Visualize activation levels as a line plot
        # norm_weeks = (weeks - tabular_stats['weeks_mean']) / tabular_stats['weeks_std']
        # norm_initial_FVC = (initial_FVC - tabular_stats['initial_fvc_mean']) / tabular_stats['initial_fvc_std']
        # norm_initial_FVC_weeks = (initial_FVC_weeks - tabular_stats['initial_weeks_mean']) / tabular_stats['initial_weeks_std']
        # tab_features = torch.cat([norm_weeks, norm_initial_FVC, norm_initial_FVC_weeks])
        # tab_features = F.relu(fc_model.tabular_expansion_fc1(tab_features))
        # tab_features = F.relu(fc_model.tabular_expansion_fc2(tab_features))
        # combined_features = torch.cat([features, tab_features])
        # print(combined_features)

        # Plot prediction vs actual for each patient
        axes[patient_order // 2][patient_order % 2].scatter(range(len(y)), y, color='blue', alpha=0.6, label='Actual')
        axes[patient_order // 2][patient_order % 2].scatter(range(len(predictions)), predictions, color='red', alpha=0.6, label='Predicted')
        axes[patient_order // 2][patient_order % 2].set_xlabel('Timepoints')
        axes[patient_order // 2][patient_order % 2].set_ylabel('FVC')
        axes[patient_order // 2][patient_order % 2].set_title(f'Patient {patient}')
        axes[patient_order // 2][patient_order % 2].legend()
    plt.show()



# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(cnn_model, fc_model, log_file, epoch=6, learning_rate=0.0001, scan_batch_size=32):
    torch.cuda.empty_cache()

    cnn_model.to(device)
    fc_model.to(device)

    training_loss = []
    val_loss = []

    criterion = nn.MSELoss()

    with open(log_file, 'w') as f:
        f.write(f"Training started: {datetime.now()}\n")
        f.write("="*70 + "\n\n")
    
    cnn_model.train()
    fc_model.train()

    optimizer = torch.optim.Adam(
        list(cnn_model.parameters()) + list(fc_model.parameters()), 
        lr=learning_rate,
        # weight_decay=1e-5  # L2 regularization
    )

    for epoch in range(epoch):
        print(f"Epoch {epoch}")
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}\n")

        patient_count = 0
        patient_list = list(train_x.keys())
        random.shuffle(patient_list)
        for patient_id in tqdm(patient_list):
            x = train_x[patient_id]
            y = train_y[patient_id]

            # Accumulate gradient for 8 datapoints for each patient
            total_loss = 0
            patient_scan_count = scan_count[patient_id]
            scans = get_scans(patient_id, patient_scan_count, scan_batch_size)
            for i in range(0, len(x), 4):
                optimizer.zero_grad()
                features = cnn_model.forward(scans)
                features = torch.mean(features, dim=0) # 1024,
                
                chunk_loss = 0
                for j in range(i, min(len(x), i+4)):
                    weeks = torch.tensor(x[j]['Weeks'], dtype=torch.float32, device=device).unsqueeze(0)
                    initial_FVC = torch.tensor(x[j]['initial_FVC'], dtype=torch.float32, device=device).unsqueeze(0)
                    initial_FVC_weeks = torch.tensor(x[j]['initial_weeks'], dtype=torch.float32, device=device).unsqueeze(0)

                    # Forward Pass
                    output = fc_model.forward(features, weeks, initial_FVC, initial_FVC_weeks).squeeze()
                    loss = criterion(output, torch.tensor(y[j], dtype=torch.float32, device=device))
                    chunk_loss += loss
                    total_loss += loss.item()

                chunk_loss.backward()
                optimizer.step()
            
            total_loss = 0
            patient_scan_count = scan_count[patient_id]
            scans = get_scans(patient_id, patient_scan_count, scan_batch_size)
            for i in range(len(x)):
                optimizer.zero_grad()
                features = cnn_model.forward(scans)
                features = torch.mean(features, dim=0) # 1024,

                weeks = torch.tensor(x[i]['Weeks'], dtype=torch.float32, device=device).unsqueeze(0)
                initial_FVC = torch.tensor(x[i]['initial_FVC'], dtype=torch.float32, device=device).unsqueeze(0)
                initial_FVC_weeks = torch.tensor(x[i]['initial_weeks'], dtype=torch.float32, device=device).unsqueeze(0)

                # Forward Pass
                output = fc_model.forward(features, weeks, initial_FVC, initial_FVC_weeks).squeeze()
                loss = criterion(output, torch.tensor(y[i], dtype=torch.float32, device=device))
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            patient_count += 1
            
            with open(log_file, "a") as f:
                f.write(f"Epoch {epoch}, Loss: {total_loss / max(len(x), 1)}\n")
            if len(x) == 0:
                print(f"Patient {patient_id} has no data during training")
            training_loss.append(total_loss / max(len(x), 1))
        
            del x, y, features, scans, weeks, initial_FVC, initial_FVC_weeks, output
            torch.cuda.empty_cache()

        # Validation
        with torch.no_grad():
            for patient_id in list[Any](val_x.keys()):
                optimizer.zero_grad()

                patient_scan_count = scan_count[patient_id]
                # Only take a look at "scan_batch_size" number of representive slices from each patient
                scans = get_scans(patient_id, patient_scan_count, scan_batch_size)

                features = cnn_model.forward(scans)
                features = torch.mean(features, dim=0)
                
                x = val_x[patient_id]
                y = val_y[patient_id]

                # Accumulate for all datapoints of patient
                accum_loss = 0
                for i in range(len(x)):
                    weeks = torch.tensor(x[i]['Weeks'], dtype=torch.float32, device=device).unsqueeze(0)
                    initial_FVC = torch.tensor(x[i]['initial_FVC'], dtype=torch.float32, device=device).unsqueeze(0)
                    initial_FVC_weeks = torch.tensor(x[i]['initial_weeks'], dtype=torch.float32, device=device).unsqueeze(0)

                    # Forward Pass
                    output = fc_model.forward(features, weeks, initial_FVC, initial_FVC_weeks).squeeze()
                    loss = criterion(output, torch.tensor(y[i], dtype=torch.float32, device=device))
                    accum_loss += loss.item()

                with open(log_file, "a") as f:
                    f.write(f"Epoch {epoch}, Val Loss: {accum_loss / max(len(x), 1)}\n")
                val_loss.append(accum_loss / max(len(x), 1))

                if len(x) == 0:
                    print(f"Patient {patient_id} has no data during validation")

                del x, y, features, scans, weeks, initial_FVC, initial_FVC_weeks, output, loss
                torch.cuda.empty_cache()

        tqdm.write(f"Epoch {epoch} completed")
        # Last 10 training loss, last 10 validation loss
        tqdm.write(f"Training Loss: {sum(training_loss[-10:]) / 10}")
        tqdm.write(f"Validation Loss: {sum(val_loss[-10:]) / 10}")
    return training_loss, val_loss



class CNN(nn.Module):
    def __init__(self, input_dim=256):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, scans: torch.Tensor):
        # Scans: List of num_slices, 1, 256, 256
        x = F.relu(self.bn1(self.conv1(scans)))
        x = self.pool(x) # num_slices, 32, 128, 128
    
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x) # num_slices, 64, 64, 64
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x) # num_slices, 128, 32, 32

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x) # num_slices, 256, 16, 16

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x) # num_slices, 512, 8, 8

        x = self.global_pool(x) # num_slices, 512, 1, 1
        x = x.view(-1, 512)

        return x

class FCLayer(nn.Module):
    def __init__(self, input_dim=512, tabular_dim=30, tabular_norm_stats=None):
        super(FCLayer, self).__init__()
        self.tabular_norm_stats = tabular_norm_stats
        if tabular_norm_stats is not None:
            # Register as buffers (not parameters, but saved with model)
            self.register_buffer('weeks_mean', torch.tensor(tabular_norm_stats['weeks_mean']))
            self.register_buffer('weeks_std', torch.tensor(tabular_norm_stats['weeks_std']))
            self.register_buffer('initial_fvc_mean', torch.tensor(tabular_norm_stats['initial_fvc_mean']))
            self.register_buffer('initial_fvc_std', torch.tensor(tabular_norm_stats['initial_fvc_std']))
            self.register_buffer('initial_weeks_mean', torch.tensor(tabular_norm_stats['initial_weeks_mean']))
            self.register_buffer('initial_weeks_std', torch.tensor(tabular_norm_stats['initial_weeks_std']))

        self.fc1 = nn.Linear(input_dim + tabular_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.tabular_expansion_fc1 = nn.Linear(3, tabular_dim)
        self.tabular_expansion_fc2 = nn.Linear(tabular_dim, tabular_dim)


        nn.init.kaiming_normal_(
            self.tabular_expansion_fc1.weight, 
            mode='fan_in',
            nonlinearity='relu')
        nn.init.constant_(self.tabular_expansion_fc1.bias, 0.01)
        nn.init.kaiming_normal_(
            self.tabular_expansion_fc2.weight, 
            mode='fan_in',
            nonlinearity='relu')
        
        nn.init.kaiming_normal_(
            self.fc1.weight, 
            mode='fan_in',
            nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0.01)
        
        nn.init.kaiming_normal_(
            self.fc2.weight, 
            mode='fan_in',
            nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.01)
        
        nn.init.kaiming_normal_(
            self.fc3.weight, 
            mode='fan_in',
            nonlinearity='relu')        

    def forward(self, features, weeks, initial_FVC, initial_FVC_weeks):
        if self.tabular_norm_stats is not None:
            weeks = (weeks - self.weeks_mean) / self.weeks_std
            initial_FVC = (initial_FVC - self.initial_fvc_mean) / self.initial_fvc_std
            initial_FVC_weeks = (initial_FVC_weeks - self.initial_weeks_mean) / self.initial_weeks_std
        
        tabular_features = torch.cat([weeks, initial_FVC, initial_FVC_weeks])
        tabular_features = F.relu(self.tabular_expansion_fc1(tabular_features))
        tabular_features = self.tabular_expansion_fc2(tabular_features)

        x = torch.cat([features, tabular_features])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
cnn_model = CNN().to(device)
fc_model = FCLayer(tabular_norm_stats=tabular_stats).to(device)
log_file = 'training_log_simple_cnn.txt'

training_loss, val_loss = train_model(cnn_model, fc_model, log_file, epoch=10, scan_batch_size=64)

plot_loss(training_loss, val_loss)
plot_patient_predict_samples(cnn_model, fc_model)
