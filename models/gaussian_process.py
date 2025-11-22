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
    def __init__(self, n_patients, sex_card, smk_card, emb_dim=4):  # Increased from 2 to 4
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
                 n_patients, sex_card, smk_card, emb_dim=4):  # Increased from 2 to 4
        # Add baseline_fvc to training inputs
        super().__init__((weeks, age, baseline_fvc, sex, smk), y, likelihood)
        
        self.mean_module = MixedKernelM2(n_patients, sex_card, smk_card, emb_dim=emb_dim)
        self.time_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=1)  # Changed from 2.5 to 1.5 for tighter fit
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

    def reset_parameters(self):
        with torch.no_grad():
            self.time_kernel.base_kernel.raw_lengthscale.normal_(0., 0.3)  # Tighter lengthscale
            self.time_kernel.raw_outputscale.normal_(0., 0.1)  # Lower outputscale = lower variance
            self.likelihood.raw_noise.normal_(-4., 0.2)  # Lower noise initialization
        
        # Add tight prior on noise to keep predictions confident
        self.likelihood.noise_covar.register_prior(
            "noise_prior",
            gpytorch.priors.NormalPrior(0.01, 0.005),
            lambda m: m.noise
        )
        return self


def build_M2(X_time_train, X_age_train, X_baseline_train,
             pid_train, sex_train, smk_train, y_train, n_patients,
             sex_card, smk_card, emb_dim=4):  # Increased from 2 to 4
    likelihood = GaussianLikelihood()
    model = GPM2(
        X_time_train, X_age_train, X_baseline_train,
        pid_train, sex_train, smk_train,
        y_train, likelihood, n_patients, sex_card, smk_card, emb_dim
    )
    model.reset_parameters()
    return model, likelihood