import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Now import the data
from preprocessing.tabular_preprocessing import train_df, val_df

# feature engineering for gru-t: dt in raw weeks
train_df["dt"] = train_df.groupby("Patient")["Weeks"].diff().fillna(0.0)
val_df["dt"]   = val_df.groupby("Patient")["Weeks"].diff().fillna(0.0)

# feature engineering for gru-t: dt in scaled weeks
train_df["dt_scaled"] = train_df.groupby("Patient")["Weeks_scaled"].diff().fillna(0.0)
val_df["dt_scaled"]   = val_df.groupby("Patient")["Weeks_scaled"].diff().fillna(0.0)

# Check the processed data
# print(val_df.head())
print(train_df.head())

def build_sequences(df, feature_cols, target_col):
    X_seqs, week_seqs, dt_seqs, y_seqs = [], [], [], []
    patients = []

    for pid, g in df.groupby("Patient"):
        g = g.sort_values("Weeks")  # safety

        X = g[feature_cols].values.astype(np.float32)      # (T, F)
        dt = g["dt"].values.astype(np.float32)            # (T,)
        y = g[target_col].values.astype(np.float32)       # (T,)
        weeks = g["Weeks"].values.astype(np.float32)    # (T,)


        X_seqs.append(X)
        dt_seqs.append(dt)
        y_seqs.append(y)
        patients.append(pid)
        week_seqs.append(weeks)


    return X_seqs, dt_seqs, y_seqs, patients, week_seqs

feature_cols = [
    "Weeks_scaled",
    "Age",
    "Sex_id",
    "Smk_id",
    "Baseline_FVC"
    ]

target_col_M1 = "FVC_scaled"
target_col_M2 = "FVC"

# Build sequences for Model 1 (using FVC_scaled as target)
X_tr_seqs_M1, dt_tr_seqs_M1, y_tr_seqs_M1, train_patients_M1, weeks_tr_seqs_M1 = build_sequences(train_df, feature_cols, target_col_M1)
X_val_seqs_M1, dt_val_seqs_M1, y_val_seqs_M1, val_patients_M1, weeks_val_seqs_M1 = build_sequences(val_df, feature_cols, target_col_M1)

print(len(X_tr_seqs_M1), "train patients")
print("Example sequence shapes:", X_tr_seqs_M1[0].shape, dt_tr_seqs_M1[0].shape, y_tr_seqs_M1[0].shape)

# Build sequences for Model 2 (using raw FVC as target)
X_tr_seqs_M2, dt_tr_seqs_M2, y_tr_seqs_M2, train_patients_M2, weeks_tr_seqs_M2 = build_sequences(train_df, feature_cols, target_col_M2)
X_val_seqs_M2, dt_val_seqs_M2, y_val_seqs_M2, val_patients_M2, weeks_val_seqs_M2 = build_sequences(val_df, feature_cols, target_col_M2)

class OSICSequenceDataset(Dataset):
    def __init__(self, X_seqs, dt_seqs, y_seqs):
        self.X_seqs = X_seqs
        self.dt_seqs = dt_seqs
        self.y_seqs = y_seqs

    def __len__(self):
        return len(self.X_seqs)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X_seqs[idx])          # (T, F)
        dt = torch.from_numpy(self.dt_seqs[idx])        # (T,)
        y = torch.from_numpy(self.y_seqs[idx])          # (T,)
        return X, dt, y

class TimeAwareGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # standard GRUCell
        self.gru_cell = nn.GRUCell(input_size, hidden_size)

        # map dt -> hidden_size decay rates
        self.dt2decay = nn.Linear(1, hidden_size)

    # add dt_t as input
    def forward(self, x_t, h_prev, dt_t):
        """
        x_t:   (batch, input_size)
        h_prev:(batch, hidden_size)
        dt_t:  (batch,) or (batch, 1)   # time gap since previous step
        """
        if dt_t.dim() == 1:
            dt_t = dt_t.unsqueeze(-1)    # (batch, 1)

        # ensure non-negative decay_rate, then gamma in (0,1]
        decay_rate = F.relu(self.dt2decay(dt_t))          # (batch, hidden_size)
        gamma = torch.exp(-decay_rate)                    # (batch, hidden_size)
        h_tilde = gamma * h_prev                          # decayed hidden state -> core of time-aware GRU

        # standard GRU update using decayed h
        h_new = self.gru_cell(x_t, h_tilde)               # (batch, hidden_size)
        return h_new

class GRUT1(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TimeAwareGRUCell(input_size, hidden_size) 
        self.out = nn.Linear(hidden_size, 1)   # predict FVC (scaled)

    def forward(self, X, dt):
        """
        X:  (batch, T, F) -> use batch size of 1 to process different sequence lengths
        dt: (batch, T)
        Returns:
            preds: (batch, T, 1)
        """
        batch_size, T, F = X.shape
        device = X.device

        h = torch.zeros(batch_size, self.hidden_size, device=device) # initial hidden state of GRU -> no memory so start with zeros
        outputs = []

        for t in range(T):
            x_t = X[:, t, :]    # (batch, F) -> feature vector at time t (for all patients in batch)
            dt_t = dt[:, t]     # (batch,) -> dt at time t 
            h = self.cell(x_t, h, dt_t) # update hidden state by passing in current input, previous memory, and time gap (dt)
            outputs.append(h.unsqueeze(1))  # (batch, 1, H) -> [h_at_t0, h_at_t1, h_at_t2, ...]

        H_all = torch.cat(outputs, dim=1)       # (batch, T, H) -> stack all hidden states over time to make predictions
        preds = self.out(H_all)                 # (batch, T, 1) -> final FVC predictions
        return preds


train_dataset_M1 = OSICSequenceDataset(X_tr_seqs_M1, dt_tr_seqs_M1, y_tr_seqs_M1)
val_dataset_M1   = OSICSequenceDataset(X_val_seqs_M1, dt_val_seqs_M1, y_val_seqs_M1)

# use batch_size=1 to handle variable-length sequences -> prevents padding
train_loader_M1 = DataLoader(train_dataset_M1, batch_size=1, shuffle=True)
val_loader_M1   = DataLoader(val_dataset_M1, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = len(feature_cols)
model1 = GRUT1(input_size=input_size, hidden_size=64).to(device)

optimizer_M1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
criterion = nn.MSELoss()  

def run_epoch_m1(loader, train=True):
    if train:
        model1.train() # for training dataset
    else:
        model1.eval() # for validation dataset

    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for X, dt, y in loader: # one patient at a time
            # shapes: X:(1,T,F), dt:(1,T), y:(1,T) -> batch_size=1
            X = X.to(device)                  # (1, T, F)
            dt = dt.to(device)                # (1, T)
            y = y.to(device).unsqueeze(-1)    # (1, T, 1)

            preds = model1(X, dt)              # (1, T, 1)

            loss = criterion(preds, y)

            if train:
                optimizer_M1.zero_grad()
                loss.backward()
                optimizer_M1.step()

            total_loss += loss.item()

    return total_loss / len(loader)

class GRUT2(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TimeAwareGRUCell(input_size, hidden_size) 
        self.out = nn.Linear(hidden_size, 2)   # output [mu, log_sigma]
        
    def forward(self, X, dt):
        """
        X:  (batch, T, F) -> use batch size of 1 to process different sequence lengths
        dt: (batch, T)
        Returns:
            preds: (batch, T, 1)
        """
        batch_size, T, F = X.shape
        device = X.device

        h = torch.zeros(batch_size, self.hidden_size, device=device) # initial hidden state of GRU -> no memory so start with zeros
        outputs = []

        for t in range(T):
            x_t = X[:, t, :]    # (batch, F) -> feature vector at time t (for all patients in batch)
            dt_t = dt[:, t]     # (batch,) -> dt at time t 
            h = self.cell(x_t, h, dt_t) # update hidden state by passing in current input, previous memory, and time gap (dt)
            outputs.append(h.unsqueeze(1))  # (batch, 1, H) -> [h_at_t0, h_at_t1, h_at_t2, ...]

        H_all = torch.cat(outputs, dim=1)       # (batch, T, H) -> stack all hidden states over time to make predictions
        preds = self.out(H_all)                 # (batch, T, 2) -> final mu, log_sigma predictions
        return preds

def laplace_nll(y_true, y_pred):
    mu = y_pred[..., 0]
    log_sigma = y_pred[..., 1]
    sigma = F.softplus(log_sigma) + 1e-6
    diff = torch.abs(y_true[..., 0] - mu)
    loss = torch.log(2.0 * sigma) + diff / sigma
    return loss.mean()


train_dataset_M2 = OSICSequenceDataset(X_tr_seqs_M2, dt_tr_seqs_M2, y_tr_seqs_M2)
val_dataset_M2  = OSICSequenceDataset(X_val_seqs_M2, dt_val_seqs_M2, y_val_seqs_M2)

# use batch_size=1 to handle variable-length sequences -> prevents padding
train_loader_M2 = DataLoader(train_dataset_M2, batch_size=1, shuffle=True)
val_loader_M2   = DataLoader(val_dataset_M2, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = len(feature_cols)
model2 = GRUT2(input_size=input_size, hidden_size=64).to(device)

optimizer_M2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

def run_epoch_m2(loader, train=True):
    if train:
        model2.train() # for training dataset
    else:
        model2.eval() # for validation dataset

    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for X, dt, y in loader: # one patient at a time
            # shapes: X:(1,T,F), dt:(1,T), y:(1,T) -> batch_size=1
            X = X.to(device)                  # (1, T, F)
            dt = dt.to(device)                # (1, T)
            y = y.to(device).unsqueeze(-1)    # (1, T, 1)

            preds = model2(X, dt)              # (1, T, 2) -> [mu, log_sigma]

            loss = laplace_nll(y, preds)

            if train:
                optimizer_M2.zero_grad()
                loss.backward()
                optimizer_M2.step()

            total_loss += loss.item()

    return total_loss / len(loader)

def best_epoch(model_name, train_loader,val_loader, model, optimizer, num_epochs=500, patience=20):
    optimizer.zero_grad()

    best_val_loss = float('inf')
    best_epoch = -1
    best_state = None
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        if model_name == "M1":
            train_loss = run_epoch_m1(train_loader, train=True)
            val_loss   = run_epoch_m1(val_loader, train=False)
        else:  # model_name == "M2"
            train_loss = run_epoch_m2(train_loader, train=True)
            val_loss   = run_epoch_m2(val_loader, train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": best_epoch
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return best_epoch, best_state, train_losses, val_losses

best_epoch_M1, best_state_M1, train_losses_M1, val_losses_M1 = best_epoch("M1", train_loader_M1, val_loader_M1, model1, optimizer_M1, num_epochs=100, patience=50)
best_epoch_M2, best_state_M2, train_losses_M2, val_losses_M2 = best_epoch("M2", train_loader_M2, val_loader_M2, model2, optimizer_M2, num_epochs=500, patience=50)

epochs_M1 = range(1, 100 + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs_M1, train_losses_M1, label="Train loss")
plt.plot(epochs_M1, val_losses_M1,   label="Validation loss")
plt.axvline(best_epoch_M1, color="gray", linestyle="--", alpha=0.7,
            label=f"Best epoch = {best_epoch_M1}")
plt.xlabel("Epoch")
plt.ylabel("Negative Log-likelihood loss")
plt.title("M1 (GRU-T) training vs validation loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


epochs_M2 = range(1, 500 + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs_M2, train_losses_M2, label="Train loss")
plt.plot(epochs_M2, val_losses_M2,   label="Validation loss")
plt.axvline(best_epoch_M2, color="gray", linestyle="--", alpha=0.7,
            label=f"Best epoch = {best_epoch_M2}")
plt.xlabel("Epoch")
plt.ylabel("Negative Log-likelihood loss")
plt.title("M2 (GRU-T) training vs validation loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.metrics import mean_absolute_error, r2_score

def eval_mae_r2_M1(model1, loader, device, fvc_scaler):
    """
    model1: GRU or any model producing FVC_scaled (shape: (1,T,1))
    loader: DataLoader returning (X, dt, y_scaled)
    fvc_scaler: StandardScaler used to scale FVC
    """
    model1.eval()

    true_scaled_all = []
    pred_scaled_all = []

    with torch.no_grad():
        for X, dt, y_scaled in loader:
            X = X.to(device)            # (1, T, F)
            dt = dt.to(device)          # (1, T)

            y_scaled = y_scaled.squeeze(0).cpu().numpy()   # (T,)

            preds = model1(X, dt)                            # (1, T, 1)
            preds = preds.squeeze(0).squeeze(-1).cpu().numpy()  # (T,)

            true_scaled_all.extend(y_scaled.tolist())
            pred_scaled_all.extend(preds.tolist())

    # convert to arrays
    true_scaled_all = np.array(true_scaled_all)
    pred_scaled_all = np.array(pred_scaled_all)

    # ===== Metrics in scaled FVC space =====
    mae_scaled = mean_absolute_error(true_scaled_all, pred_scaled_all)
    r2_scaled  = r2_score(true_scaled_all, pred_scaled_all)

    # ===== Convert to RAW FVC in mL =====
    true_raw = fvc_scaler.inverse_transform(true_scaled_all.reshape(-1,1)).ravel()
    pred_raw = fvc_scaler.inverse_transform(pred_scaled_all.reshape(-1,1)).ravel()

    mae_raw = mean_absolute_error(true_raw, pred_raw)
    r2_raw  = r2_score(true_raw, pred_raw)

    return {
        "mae_scaled": mae_scaled,
        "r2_scaled": r2_scaled,
        "mae_raw": mae_raw,
        "r2_raw": r2_raw,
    }

metrics_train_M1 = eval_mae_r2_M1(model1, train_loader_M1, device, fvc_scaler)
metrics_val_M1   = eval_mae_r2_M1(model1, val_loader_M1, device, fvc_scaler)

print("=== M1 TRAIN ===")
print("MAE (scaled):", metrics_train_M1["mae_scaled"])
print("R2   (scaled):", metrics_train_M1["r2_scaled"])
print("MAE (mL):", metrics_train_M1["mae_raw"])
print("R2   (mL):", metrics_train_M1["r2_raw"])

print("\n=== M1 VAL ===")
print("MAE (scaled):", metrics_val_M1["mae_scaled"])
print("R2   (scaled):", metrics_val_M1["r2_scaled"])
print("MAE (mL):", metrics_val_M1["mae_raw"])
print("R2   (mL):", metrics_val_M1["r2_raw"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_patient_pred(model_name, model, dataset, weeks_seqs, idx):
    """
    Returns weeks, true FVC (scaled), predicted FVC (scaled) for one patient.
    """
    model.eval()

    X, dt, y = dataset[idx]          # X:(T,F), dt:(T,), y:(T,)
    weeks = weeks_seqs[idx]          # (T,)

    # add batch dimension: (1,T,F), (1,T)
    X_in = X.unsqueeze(0).to(device)
    dt_in = dt.unsqueeze(0).to(device)

    if model_name == "M1":
        with torch.no_grad():
            pred_scaled = model(X_in, dt_in)                      # (1,T,1)
            pred_scaled = pred_scaled.cpu().numpy().reshape(-1)   # (T,)

        true_scaled = y.numpy().reshape(-1)                       # (T,)
        return weeks, true_scaled, pred_scaled
    else: # for M2
        with torch.no_grad():
            preds = model(X_in, dt_in)             # (1,T,2)
            mu = preds[..., 0].squeeze(0)          # (T,)
            log_sigma = preds[..., 1].squeeze(0)   # (T,)
            sigma = F.softplus(log_sigma) + 1e-6   # (T,)
        
        mu_raw = mu.cpu().numpy()            # (T,)
        sigma_raw = sigma.cpu().numpy()      # (T,)
        true_raw = y.numpy().reshape(-1)     # (T,)

        return weeks, true_raw, mu_raw, sigma_raw
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_patient_pred_scaled(model, dataset, weeks_seqs, idx):
    """
    Returns weeks, true FVC (scaled), predicted FVC (scaled) for one patient.
    """
    model.eval()

    X, dt, y = dataset[idx]          # X:(T,F), dt:(T,), y:(T,)
    weeks = weeks_seqs[idx]          # (T,)

    # add batch dimension: (1,T,F), (1,T)
    X_in = X.unsqueeze(0).to(device)
    dt_in = dt.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model(X_in, dt_in)      # (1,T,1)

    pred_scaled = pred_scaled.cpu().numpy().reshape(-1)  # (T,)
    true_scaled = y.numpy().reshape(-1)                  # (T,)

    return weeks, true_scaled, pred_scaled


def plot_grut_grid_M1(model, dataset, weeks_seqs, patient_ids, indices=None, n_rows=2, n_cols=5, ci_dict=None):

    model.eval()

    figsize=(18, 6)

    n_plots = n_rows * n_cols

    if indices is None:
        indices = list(range(min(n_plots, len(dataset))))
    else:
        indices = indices[:n_plots]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        pid = patient_ids[idx]
        weeks, true_fvc_scaled, pred_fvc_scaled = get_patient_pred(
            "M1", model, dataset, weeks_seqs, idx
        )

        # true FVC (scaled)
        ax.plot(
            weeks, true_fvc_scaled,
            marker="o", color="C0",
            label="True FVC" if idx == indices[0] else None
        )

        # GRU-T predicted FVC (scaled)
        ax.plot(
            weeks, pred_fvc_scaled,
            marker="s", linestyle="--", color="C1",
            label="Predicted FVC" if idx == indices[0] else None
        )

        ax.set_title(f"Patient {pid[:8]}...")
        ax.set_xlabel("Weeks")
        ax.set_ylabel("FVC (scaled)")
        ax.grid(True)

    # Put shared legend in first axes (or figure-level)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

plot_grut_grid_M1(
    model=model1,
    dataset=val_dataset_M1,
    weeks_seqs= weeks_val_seqs_M1,
    patient_ids=val_patients_M1,
    n_rows=2,
    n_cols=5,
    ci_dict=None)

def laplace_log_likelihood(y_true, y_pred, sigma):
    sigma_clipped = np.maximum(sigma, 70)
    delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
    score = - np.sqrt(2) * delta_clipped / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
    return np.mean(score)

def get_mu_sigma(model, X, dt):
    model.eval()
    with torch.no_grad():
        preds = model(X, dt)          # (1, T, 2)
        mu_t = preds[..., 0]          # (1, T)
        log_sigma_t = preds[..., 1]   # (1, T)

        # numerically stable softplus in torch (avoid overflow) -> without this the laplace score might be -inf
        sigma_t = F.softplus(log_sigma_t) + 1e-6   # (1, T)

    mu = mu_t.squeeze(0).cpu().numpy()             # (T,)
    sigma = sigma_t.squeeze(0).cpu().numpy()       # (T,)

    return mu, sigma


def eval_lapace_on_loader(model, loader, device):
    model.eval()
    scores = []

    with torch.no_grad():
        for X, dt, y in loader:
            # X: (1, T, F), dt: (1, T), y: (1, T)
            X = X.to(device)
            dt = dt.to(device)

            # y is target FVC in mL
            y_true = y.squeeze(0).cpu().numpy()   # (T,)

            # get mu, sigma from the model
            mu, sigma = get_mu_sigma(model, X, dt)  # both (T,)

            # compute laplace log likelihood for this patient
            score = laplace_log_likelihood(
                y_true=y_true,
                y_pred=mu,
                sigma=sigma
            )
            scores.append(score)

    # mean OSIC score across all patients in this loader
    return float(np.mean(scores))


from sklearn.metrics import mean_absolute_error, r2_score

def eval_mae_r2(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for X, dt, y in loader:
            X = X.to(device)        # (1, T, F)
            dt = dt.to(device)      # (1, T)

            y_true = y.squeeze(0).cpu().numpy()       # (T,)

            # get predicted mu
            mu, _ = get_mu_sigma(model, X, dt)   # mu: (T,)

            all_true.extend(y_true.tolist())
            all_pred.extend(mu.tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    mae = mean_absolute_error(all_true, all_pred)
    r2 = r2_score(all_true, all_pred)

    return mae, r2

model2.eval()

train_osic = eval_lapace_on_loader(model2, train_loader_M2, device)
val_osic   = eval_lapace_on_loader(model2, val_loader_M2, device)

print(f"Train Laplace Log Likelihood score: {train_osic:.4f}")
print(f"Val Laplace Log Likelihood score: {val_osic:.4f}")

train_mae, train_r2 = eval_mae_r2(model2, train_loader_M2, device)
val_mae, val_r2     = eval_mae_r2(model2, val_loader_M2, device)

print("\nTrain R2:", train_r2)
print("Val R2:", val_r2)

print("\nTrain MAE:", train_mae)
print("Val MAE:", val_mae)


import matplotlib.pyplot as plt

def plot_patient_M2(ax, weeks, true, pred, sigma, title=None, ci=1.96):
    """
    ax    : matplotlib Axes
    weeks : (T,)
    true  : (T,)  true FVC (scaled or raw)
    pred  : (T,)  mu
    sigma : (T,)  std dev in same units as true/pred
    ci    : float, e.g. 1.96 for ~95% CI
    """

    # CI bounds
    upper = pred + ci * sigma
    lower = pred - ci * sigma

    # shade CI
    ax.fill_between(weeks, lower, upper, color="C0", alpha=0.2, label=f"{int(ci*100/1.96)}% CI")

    # true points
    ax.plot(weeks, true, marker="o", linestyle="-", color="C0", label="True FVC")

    # predicted mean
    ax.plot(weeks, pred, marker="s", linestyle="--", color="C1", label="Predicted FVC")

    ax.set_xlabel("Weeks")
    ax.set_ylabel("FVC (scaled)" if title and "scaled" in title.lower() else "FVC")
    if title:
        ax.set_title(title)
    ax.grid(True)

n_patients_to_plot = 10
fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharey=True)
axes = axes.flatten()

for i in range(n_patients_to_plot):
    weeks, true, pred, sigma = get_patient_pred(
        model_name="M2",
        model=model2,
        dataset=val_dataset_M2,        # your M2 dataset
        weeks_seqs= weeks_val_seqs_M2,     # list of week arrays
        idx=i,
    )

    ax = axes[i]
    # pid = val_patients_M2[i] if 'patient_ids_val' in globals() else f"Patient {i}"
    pid = val_patients_M2[i]
    plot_patient_M2(ax, weeks, true, pred, sigma, title=f"Patient {pid[:8]}...")
    
# common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.show()

def plot_grut_grid_M2(model, dataset, weeks_seqs, patient_ids, indices=None, n_rows=2, n_cols=5, ci=1.96):
    """
    Plots a grid of GRU-T predictions (μ ± CI·σ) for multiple patients.

    Parameters
    ----------
    model         : GRU-T model (predicts (mu, logsigma))
    dataset       : Dataset returning (X, dt, y_raw)
    weeks_seqs    : list of arrays of week indices per patient
    patient_ids   : list of patient IDs
    fvc_scaler    : StandardScaler for FVC (optional)
                    - if provided → convert raw FVC to scaled
                    - if None     → plot raw FVC in mL
    indices       : list of patient indices to plot
    ci            : float, confidence interval multiplier
    """

    model.eval()

    figsize = (18, 6)
    n_plots = n_rows * n_cols

    if indices is None:
        indices = list(range(min(n_plots, len(dataset))))
    else:
        indices = indices[:n_plots]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        pid = patient_ids[idx]

        # Weeks, true_y, mu, sigma (scaled or raw depending on fvc_scaler)
        weeks, true_fvc, pred_mu, sigma = get_patient_pred(
            model_name="M2",
            model=model,
            dataset=dataset,
            weeks_seqs=weeks_seqs,
            idx=idx
        )

        # Confidence interval bounds
        upper = pred_mu + ci * sigma
        lower = pred_mu - ci * sigma

        # CI shaded area
        ax.fill_between(
            weeks, lower, upper,
            color="C0", alpha=0.15,
            label="95% CI" if idx == indices[0] else None
        )

        # true FVC
        ax.plot(
            weeks, true_fvc,
            marker="o", color="C0",
            label="True FVC" if idx == indices[0] else None
        )

        # predicted mean FVC
        ax.plot(
            weeks, pred_mu,
            marker="s", linestyle="--", color="C1",
            label="Predicted FVC" if idx == indices[0] else None
        )

        ax.set_title(f"Patient {pid[:8]}...")
        ax.set_xlabel("Weeks")

        ylabel = "FVC (mL)"
        ax.set_ylabel(ylabel)

        ax.grid(True)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.show()

plot_grut_grid_M2(
    model=model2,
    dataset=val_dataset_M2,
    weeks_seqs= weeks_val_seqs_M2,
    patient_ids=val_patients_M2,
)
