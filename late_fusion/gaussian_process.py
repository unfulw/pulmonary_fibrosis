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

# Import the data
repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root))

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


M1.eval(); likelihood.eval()

Xc_val  = torch.tensor(val_df[["Weeks_scaled","Age"]].values, dtype=torch.float32)
sex_val = torch.tensor(val_df["Sex"].map(sex_map).values, dtype=torch.long).unsqueeze(-1)
smk_val = torch.tensor(val_df["SmokingStatus"].map(smk_map).values, dtype=torch.long).unsqueeze(-1)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    X_time_val = torch.tensor(val_df[["Weeks_scaled"]].values, dtype=torch.float32)
    X_age_val  = torch.tensor(val_df[["Age"]].values,          dtype=torch.float32)
    pred = likelihood(M1(X_time_val, X_age_val, sex_val, smk_val))   
    y_mean = pred.mean.numpy()
    y_std  = pred.variance.sqrt().numpy()

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
