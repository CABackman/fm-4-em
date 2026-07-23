from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from tabpfn_extensions.interpretability.pdp import partial_dependence_plots
from tabpfn_extensions.interpretability.feature_selection import feature_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import sys
from tabpfn import TabPFNRegressor
project_root = Path.cwd()
print("SCRIPT STARTED", flush=True)

modules_path = project_root / "modules"
sys.path.append(str(modules_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

embedding = "maceoff_large"

embds = np.load(f"{project_root}/02_EMBEDDINGS/log(h50)_exp/{embedding}_embeddings.npy")
dataset = pd.read_csv(f"{project_root}/01_DATASETS/log(h50)_exp/log(h50)_exp.csv")
ys = dataset["val"].values
test_split_idx = np.load(f"{project_root}/01_DATASETS/log(h50)_exp/random_seed0_split_test.npy")

train_idx = np.setdiff1d(np.arange(len(dataset)), test_split_idx)
train_x = embds[train_idx]
train_y = ys[train_idx]
test_x = embds[test_split_idx]
test_y = ys[test_split_idx]

y_scaler = MinMaxScaler()
train_y = y_scaler.fit_transform(train_y.reshape(-1, 1)).ravel()
test_y = y_scaler.transform(test_y.reshape(-1, 1)).ravel()

model = TabPFNRegressor(device=device)
model.fit(train_x, train_y)
predictions = model.predict(test_x)

predictions = y_scaler.inverse_transform(predictions)
# Evaluate the model
mae = mean_absolute_error(test_y, predictions)
r2 = r2_score(test_y, predictions)
print(f"Mean Absolute Error: {mae}", flush=True)
print(f"R^2 Score: {r2}", flush=True)

print("Generating partial dependence plots...", flush=True)
# select the most important features according to TabPFN's internal feature selection
result = feature_selection(model, train_x, train_y, n_features_to_select=10)
top_features = result.selected_indices
print(f"Top features: {top_features}", flush=True)

# generate partial dependence plots for selected features
partial_dependence_plots(model, test_x, features=top_features, kind="average", target_class=1)
plt.tight_layout()
plt.savefig(f"{project_root}/05_PY_SCRIPTS/{embedding}_pdp_plot.png")