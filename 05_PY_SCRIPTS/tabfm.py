from __future__ import annotations
from pathlib import Path
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import sys, os
from tabpfn import TabPFNRegressor
project_root = Path.cwd()

modules_path = project_root / "modules"
sys.path.append(str(modules_path))
mace_embds = np.load(f"{project_root}/02_EMBEDDINGS/log(h50)_exp/maceoff_large_embeddings.npy")
dataset = pd.read_csv(f"{project_root}/01_DATASETS/log(h50)_exp/log(h50)_exp.csv")
ys = dataset["val"].values

model = TabPFNRegressor()

preds = {}
for split in Path(f"{project_root}/01_DATASETS/log(h50)_exp").glob("random*.npy"):
    test_idx = np.load(split)
    train_idx = np.setdiff1d(np.arange(len(dataset)), test_idx)

    # split datasets 
    train_x = mace_embds[train_idx]
    train_y = ys[train_idx]

    test_x = mace_embds[test_idx]
    test_y = ys[test_idx]

    # train model
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)

    # save predictions
    preds[split.stem] = predictions

