import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def eval_metrics(y_true, y_pred):
    """
    Evaluate the model using a custom metric.
    
    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted values.
    
    Returns:
    float: Computed evaluation metric.
    """
    # Example: Mean Squared Errorresiduals = y_true_flat - y_pred_flat
    residuals = y_true - y_pred
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# Generalized plotting function for ANY model
# The function can be used after performing train-validation split
def plot_n_patients(model, X_val, y_val, patients_val, feature_cols, n=5):
    """
    Generalized plotting function for ANY model.
    Plots True FVC vs Predicted FVC across time,
    for the first n patients in the validation set.

    Parameters:
    model: Trained model with a predict method
    X_val (np.ndarray): Validation features of shape (num_patients, T_max, num_features).
    y_val (np.ndarray): Validation targets of shape (num_patients, T_max, 1).
    patients_val (list): List of patient IDs corresponding to the validation set.
    feature_cols (list): List of feature column names.
    n (int): Number of patients to plot.
    """

    n_patients = min(n, X_val.shape[0])

    # Determine if "Weeks" column exists (optional)
    has_weeks = "Weeks" in feature_cols
    weeks_idx = feature_cols.index("Weeks") if has_weeks else None

    # Model prediction
    y_val_pred = model.predict(X_val[:n_patients])

    for i in range(n_patients):
        x_seq = X_val[i]                    # (T_max, num_features)
        y_true_seq = y_val[i, :, 0]         # (T_max,)
        y_pred_seq = y_val_pred[i, :, 0]    # (T_max,)

        # Mask padded timesteps (assuming padded y == 0)
        mask = y_true_seq != 0

        # X-axis for plotting
        if has_weeks:
            x_axis = x_seq[mask, weeks_idx]
            x_label = "Weeks"
        else:
            x_axis = np.arange(mask.sum())
            x_label = "Time Step"

        plt.figure(figsize=(7, 4))

        # Plot true and predicted values
        plt.plot(x_axis, y_true_seq[mask], marker="o", label="True FVC")
        plt.plot(x_axis, y_pred_seq[mask], marker="x", label="Predicted FVC")

        plt.title(f"Patient {patients_val[i]}")
        plt.xlabel(x_label)
        plt.ylabel("FVC (mL)")
        plt.legend()
        plt.tight_layout()
        plt.show()


# OSIC custom Laplace log-likelihood metric
def laplace_log_likelihood(y_true, y_pred, y_sd):
    """
    Parameters:
    y_true (np.ndarray): True values.
    y_pred (np.ndarray): Predicted (mean) values.
    y_sd (np.ndarray): Predicted Standard deviation values
    
    Returns:
    float: Computed laplace_log_likelihood metric.
    """
    sd_clipped = np.maximum(y_sd, 70)
    delta = np.minimum(np.abs(y_true-y_pred), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric)

