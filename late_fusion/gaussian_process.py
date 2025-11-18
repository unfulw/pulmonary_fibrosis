import numpy as np
import pandas as pd
from scipy import stats
import sys
from pathlib import Path
import torch
import torch.nn as nn
import gpytorch
import matplotlib.pyplot as plt
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, IndexKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.metrics import mean_squared_error, r2_score

# Set up path BEFORE importing from preprocessing
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Now import the data
from preprocessing.tabular_preprocessing import train_df, val_df



# Data preprocessing
X_time_train = torch.tensor(train_df[["Weeks_scaled"]].values, dtype=torch.float32)
X_age_train = torch.tensor(train_df[["Age"]].values, dtype=torch.float32)

sex_map = {k:i for i,k in enumerate(train_df["Sex"].astype("category").cat.categories)}
smk_map = {k:i for i,k in enumerate(train_df["SmokingStatus"].astype("category").cat.categories)}
sex_train = torch.tensor(train_df["Sex"].map(sex_map).values, dtype=torch.long).unsqueeze(-1)
smk_train = torch.tensor(train_df["SmokingStatus"].map(smk_map).values, dtype=torch.long).unsqueeze(-1)
y_train = torch.tensor(train_df["FVC_scaled"].values, dtype=torch.float32)

# M1: GP with no inter-patient variance explored (pid dropped)
class MixedKernel(nn.Module):
    has_lengthscale = False
    def __init__(self):
        super().__init__()
        self.time = ScaleKernel(MaternKernel(nu=1.5))
        self.age  = ScaleKernel(RBFKernel())

        self.sex  = IndexKernel(num_tasks=len(sex_map), rank=2)
        self.smk  = IndexKernel(num_tasks=len(smk_map), rank=2)

    def forward(self, x_time, x_age, x_sex, x_smk, diag=False, **params):
        Kt = self.time(x_time, x_time, diag=diag)
        Ka = self.age(x_age,  x_age,  diag=diag)

        Ks  = self.sex(x_sex,  x_sex)
        Km  = self.smk(x_smk,  x_smk)

        Kdemo = Ka + Ks + Km
        return Kt * Kdemo 
    

class GPM1(gpytorch.models.ExactGP):
    def __init__(self, Xt, Xage, Xsex, Xsmk, y, likelihood):
        super().__init__(train_inputs=(Xt, Xage, Xsex, Xsmk), train_targets=y, likelihood=likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = MixedKernel()

    def forward(self, Xt, Xage, Xsex, Xsmk):
        mean = self.mean_module(torch.cat([Xt, Xage], dim=-1)) 
        covar = self.covar_module(Xt, Xage, Xsex, Xsmk)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

likelihood = GaussianLikelihood()
M1 = GPM1(X_time_train, X_age_train, sex_train, smk_train, y_train, likelihood)

M1.train(); likelihood.train()
optimizer = torch.optim.Adam(M1.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, M1)

for i in range(200):  
    optimizer.zero_grad()
    output = M1(*M1.train_inputs)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()


# M1 evaluation
M1.eval()
likelihood.eval()

X_time_val = torch.as_tensor(val_df[["Weeks_scaled"]].values, dtype=torch.float32)
X_age_val  = torch.as_tensor(val_df[["Age"]].values,          dtype=torch.float32)
sex_val    = torch.as_tensor(val_df["Sex"].map(sex_map).values, dtype=torch.long).unsqueeze(-1)
smk_val    = torch.as_tensor(val_df["SmokingStatus"].map(smk_map).values, dtype=torch.long).unsqueeze(-1)

y_true = torch.as_tensor(val_df["FVC_scaled"].values, dtype=torch.float32)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(M1(X_time_val, X_age_val, sex_val, smk_val))

y_mean = pred.mean.cpu().numpy()
y_std  = pred.variance.sqrt().cpu().numpy()
y_true = y_true.cpu().numpy()

rmse = np.sqrt(mean_squared_error(y_true, y_mean))
r2   = r2_score(y_true, y_mean)
print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

plt.figure()
plt.errorbar(y_true, y_mean, yerr=2*y_std, fmt='o', alpha=0.5, ecolor='lightgray')
plt.plot([-3,3],[-3,3],'r--')  # 1:1 line if still scaled
plt.xlabel("True FVC (scaled)")
plt.ylabel("Predicted FVC (scaled)")
plt.title("Population-level GP Predictions (95% CI)")
plt.show()

# M2: GP with patient-specific effects
# Add patient IDs and baseline for M2
pid_map = {k:i for i,k in enumerate(train_df["Patient"].astype("category").cat.categories)}
pid_train = torch.tensor(train_df["Patient"].map(pid_map).values, dtype=torch.long).unsqueeze(-1)

sex_card = train_df["Sex"].nunique()
smk_card = train_df["SmokingStatus"].nunique()
n_patients = train_df["Patient"].nunique()

# Baseline_FVC is now calculated in preprocessing, so we just use it directly
X_baseline_train = torch.tensor(train_df[["Baseline_FVC"]].values, dtype=torch.float32)

class MixedKernelM2(nn.Module):
    def __init__(self, n_patients, sex_card, smk_card, emb_dim=2):
        super().__init__()
        self.beta_w = nn.Linear(1, 1, bias=False)  # weeks effect
        self.beta_a = nn.Linear(1, 1, bias=False)  # age effect

        self.alpha = nn.Embedding(n_patients, 1)
        self.gamma = nn.Embedding(n_patients, 1)

        self.sex_emb = nn.Embedding(sex_card, emb_dim)
        self.smk_emb = nn.Embedding(smk_card, emb_dim)
        self.cat_lin = nn.Linear(2*emb_dim, 1, bias=False)

    def forward(self, weeks, age, baseline_fvc, pid, sex_id, smk_id, use_patient_effects=True):
        # START FROM BASELINE! Predict CHANGE from baseline
        # Population-level effects (relative to baseline)
        delta = self.beta_w(weeks) + self.beta_a(age)
        delta = delta.squeeze()
        
        # Demographic effects
        sex_emb = self.sex_emb(sex_id.squeeze(-1))
        smk_emb = self.smk_emb(smk_id.squeeze(-1))
        cat = self.cat_lin(torch.cat([sex_emb, smk_emb], dim=-1))
        cat = cat.squeeze()
        
        # Patient-specific effects
        if use_patient_effects and pid is not None:
            alpha = self.alpha(pid.squeeze(-1)).squeeze()
            gamma = self.gamma(pid.squeeze(-1)).squeeze()
            weeks_1d = weeks.squeeze()
            pat = alpha + gamma * weeks_1d
            # CRITICAL: Add baseline to the delta/change
            return baseline_fvc.squeeze() + delta + pat + cat
        else:
            # CRITICAL: Add baseline to the delta/change
            return baseline_fvc.squeeze() + delta + cat

    
class GPM2(gpytorch.models.ExactGP):
    def __init__(self, weeks, age, baseline_fvc, pid, sex, smk, y, likelihood, 
                 n_patients, sex_card, smk_card, emb_dim=2):
        # Add baseline_fvc to training inputs
        super().__init__((weeks, age, baseline_fvc, sex, smk), y, likelihood)
        
        self.mean_module = MixedKernelM2(n_patients, sex_card, smk_card, emb_dim=emb_dim)
        self.time_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=1)
        )
        self.emb_dim = emb_dim
        self.n_patients = n_patients
        self.register_buffer('train_pid', pid.squeeze(-1))

    def forward(self, weeks, age, baseline_fvc, sex_id, smk_id, pid=None, use_patient_effects=True):
        if pid is None and self.training:
            pid = self.train_pid.unsqueeze(-1)
        
        mean_x = self.mean_module(weeks, age, baseline_fvc, pid, sex_id, smk_id, use_patient_effects)
        covar = self.time_kernel(weeks)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

likelihood_m2 = GaussianLikelihood()
M2 = GPM2(X_time_train, X_age_train, X_baseline_train, pid_train, sex_train, smk_train, 
          y_train, likelihood_m2, n_patients, sex_card, smk_card, emb_dim=2)

M2.train()
likelihood_m2.train()
optimizer_m2 = torch.optim.Adam(M2.parameters(), lr=0.05)
mll_m2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_m2, M2)

for i in range(500):  
    optimizer_m2.zero_grad()
    weeks, age, baseline_fvc, sex, smk = M2.train_inputs  # Now 5 inputs
    output = M2(weeks, age, baseline_fvc, sex, smk, use_patient_effects=True)
    loss = -mll_m2(output, y_train)
    loss.backward()
    optimizer_m2.step()
    
    if (i + 1) % 50 == 0:
        print(f"Iter {i+1}/500 - Loss: {loss.item():.3f}")

M2.eval()
likelihood.eval()

X_time_val = torch.tensor(val_df[["Weeks_scaled"]].values, dtype=torch.float32)
X_age_val = torch.tensor(val_df[["Age"]].values, dtype=torch.float32)
X_baseline_val = torch.tensor(val_df[["Baseline_FVC"]].values, dtype=torch.float32)
sex_val = torch.tensor(val_df["Sex"].map(sex_map).values, dtype=torch.long).unsqueeze(-1)
smk_val = torch.tensor(val_df["SmokingStatus"].map(smk_map).values, dtype=torch.long).unsqueeze(-1)


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(M2(X_time_val, X_age_val, X_baseline_val, sex_val, smk_val, 
                         use_patient_effects=False))
    y_pred = pred.mean.numpy()
    y_std = pred.variance.sqrt().numpy()

y_true = val_df["FVC_scaled"].values
residuals = y_true - y_pred

mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals**2))
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"\nResidual Statistics:")
print(f"Mean: {residuals.mean():.4f}")
print(f"Std: {residuals.std():.4f}")
print(f"Min: {residuals.min():.4f}")
print(f"Max: {residuals.max():.4f}")

print(f"R squared: {r2_score(y_true, y_pred)}")

plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, alpha=0.5, s=20)
plt.plot([y_true.min(), y_true.max()], 
         [y_true.min(), y_true.max()], 
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('True FVC (scaled)')
plt.ylabel('Predicted FVC (scaled)')
plt.title('Predicted vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()

# Select 10 patient IDs from validation set
sample_patients = val_df['Patient'].unique()[:10]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

M2.eval()
likelihood.eval()

for idx, patient_id in enumerate(sample_patients):
    patient_data = val_df[val_df['Patient'] == patient_id].sort_values('Weeks')
    
    # Prepare patient-specific data
    X_time_patient = torch.tensor(patient_data[["Weeks_scaled"]].values, dtype=torch.float32)
    X_age_patient = torch.tensor(patient_data[["Age"]].values, dtype=torch.float32)
    X_baseline_patient = torch.tensor(patient_data[["Baseline_FVC"]].values, dtype=torch.float32)
    sex_patient = torch.tensor(patient_data["Sex"].map(sex_map).values, dtype=torch.long).unsqueeze(-1)
    smk_patient = torch.tensor(patient_data["SmokingStatus"].map(smk_map).values, dtype=torch.long).unsqueeze(-1)
    
    # Get predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_patient = likelihood(M2(X_time_patient, X_age_patient, X_baseline_patient, 
                                      sex_patient, smk_patient, use_patient_effects=False))
        y_pred_patient = pred_patient.mean.numpy()
        y_std_patient = pred_patient.variance.sqrt().numpy()
    
    y_true_patient = patient_data["FVC_scaled"].values
    weeks = patient_data["Weeks"].values
    
    # Plot - use consistent 2D indexing
    row = idx // 5
    col = idx % 5
    axes[row, col].plot(weeks, y_true_patient, 'o-', label='True FVC', linewidth=2, markersize=6)
    axes[row, col].plot(weeks, y_pred_patient, 's--', label='Predicted FVC', linewidth=2, markersize=6)
    axes[row, col].fill_between(weeks, 
                           y_pred_patient - 2*y_std_patient, 
                           y_pred_patient + 2*y_std_patient, 
                           alpha=0.2, label='95% CI')
    
    axes[row, col].set_xlabel('Weeks')
    axes[row, col].set_ylabel('FVC (scaled)')
    axes[row, col].set_title(f'Patient {patient_id[:8]}...')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('longitudinal_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
