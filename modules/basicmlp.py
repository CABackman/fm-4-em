"""
Vanilla MLP baseline for probing frozen molecular embeddings.

This module implements a simple feed-forward multilayer perceptron (MLP)
used as a nonlinear regression probe on top of frozen foundation-model
embeddings. The goal is to assess how much task-relevant signal is present
in the embeddings without any fine-tuning of the underlying representation.

The MLP is intentionally kept simple and conservative:
- Fixed hidden sizes (512 → 256) across all embeddings
- LayerNorm instead of BatchNorm for stability in small-data regimes
- GELU activations to match Transformer-style embedding distributions
- Early stopping based on validation loss to prevent overfitting

This setup is intended as a *baseline* for out-of-distribution, low-data
evaluation, not as a high-capacity or optimized predictive model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import seaborn as sns
import torch.nn as nn
import copy
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Utilities
def split_train_into_train_val(
    train_idx: np.ndarray,
    val_ratio: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split an existing train index array into new train and validation indices.

    Parameters
    ----------
    train_idx : np.ndarray
        Array of training indices.
    val_ratio : float, default=0.2
        Fraction of train_idx to use for validation.
    random_state : int or None
        Seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle before splitting.

    Returns
    -------
    new_train_idx, val_idx : np.ndarray
        Updated training indices and validation indices.
    """

    train_idx = np.asarray(train_idx)
    n = len(train_idx)

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    indices = train_idx.copy()

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    val_size = int(val_ratio * n)

    val_idx = indices[:val_size]
    new_train_idx = indices[val_size:]

    return new_train_idx, val_idx

def get_tvt(
    emb: torch.Tensor,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    ) -> dict:
    """
    Prepare train, validation, and test splits from embeddings and labels.

    Parameters
    ----------
    emb : torch.Tensor
        A 2D tensor or array-like of shape (N, D), where N is the number of
        samples and D is the embedding dimension.
    y : np.ndarray
        Array-like of shape (N,) or (N, 1) containing target labels.
    train_idx : np.ndarray
        Indices specifying which samples belong to the training set.
    test_idx : np.ndarray
        Indices specifying which samples belong to the test set.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        
        - "X_train": standardized training features (numpy array)
        - "X_val": standardized validation features (numpy array)
        - "X_test": standardized test features (numpy array)
        - "y_train": training labels as a torch.FloatTensor of shape (N_train, 1)
        - "y_val": validation labels as a torch.FloatTensor of shape (N_val, 1)
        - "y_test": test labels as a torch.FloatTensor of shape (N_test, 1)
    scaler object
        Standard scaler fit on the X_train split 

    """

    y = np.asarray(y).ravel()
    
    X = np.asarray(emb)
    if X.shape[0] != len(y):
        raise ValueError(f"Emb: X has {X.shape[0]} rows but y has {len(y)}")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx],   y[val_idx]
    X_test, y_test   = X[test_idx],  y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    train_val_test = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

    return train_val_test, scaler 

def seed_all(seed: int = 42) -> None:
    """
    Set random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VanillaMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden1: int | None = None,
        hidden2: int | None = None,
        dropout: float = 0.05,
        proj_dim: int = 256,
    ):
        super().__init__()

        # Resolve defaults
        if hidden1 is None:
            hidden1 = 256
        if hidden2 is None:
            hidden2 = max(1, 256 // 2)

        # Validate
        assert hidden1 > 0, "hidden1 must be positive"
        assert hidden2 > 0, "hidden2 must be positive"
        assert proj_dim > 0, "proj_dim must be positive"

        # set class var
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.proj_dim = proj_dim
        

        # Project all embeddings to same dimension (capacity scaling)
        self.proj = nn.Linear(input_size, proj_dim, bias=False)

        self.net = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.net(x)

def mae_from_pred(pred, target):
    return torch.mean(torch.abs(pred - target))

def mse_from_pred(pred, target):
    return torch.mean((pred - target) ** 2)

def r2_from_pred(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - ss_res / ss_tot

@torch.no_grad()
def eval_metrics(model, X, y, y_scaler=None):
    model.eval()
    pred = model(X)

    if y_scaler is not None:
        pred = torch.tensor(
            y_scaler.inverse_transform(pred.cpu().numpy()),
            device=pred.device,
            dtype=pred.dtype,
        )
        y = torch.tensor(
            y_scaler.inverse_transform(y.cpu().numpy()),
            device=y.device,
            dtype=y.dtype,
        )

    mse = mse_from_pred(pred, y).item()
    mae = mae_from_pred(pred, y).item()
    rmse = mse ** 0.5
    r2 = r2_from_pred(pred, y).item()

    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

def train_with_early_stopping(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    y_scaler: MinMaxScaler | None = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 400,
    patience: int = 20,
    min_delta: float = 0.0,
    print_every: int = 20,
    early_stop_on: str = "mse",
) -> dict:
    """
    Train an MLP AdamW and early stopping.

    The model is optimized with MSE loss (option for MAE). Each epoch logs and stores MSE/RMSE/MAE
    on train and validation sets. Early stopping based on MSE with a required improvement of
    `min_delta` within `patience` epochs.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    X_train, y_train : torch.Tensor
        Training inputs and targets.
    X_val, y_val : torch.Tensor
        Validation inputs and targets.
    X_test, y_test : torch.Tensor
        Test inputs and targets (evaluation only).
    lr : float
        AdamW learning rate.
    y_scaler : 
        Method that scaled target (fit on y train)
    weight_decay : float
        AdamW weight decay.
    num_epochs : int
        Max epochs. Default = 20
    patience : int
        Early-stopping patience (epochs).
    min_delta : float
        Minimum validation improvement to reset patience.
    print_every : int
        Print interval in epochs (0 disables).
    early_stop_on : {"mse","mae"}
        Validation metric used to choose the best checkpoint.

    Returns
    -------
    dict
        Keys: best_val, early_stop_on, final_epoch, epochs_run,
        history (train/val mse, rmse, mae per epoch),
        train/val/test (final metrics on best checkpoint).
    """

    criterion = nn.MSELoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_MSE": [],
        "val_MSE": [],
        "train_MAE": [],
        "val_MAE": [],
        "train_RMSE": [],
        "val_RMSE": [],
        "train_R2": [],
        "val_R2": [],
    }

    # Early stopping tracker
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    final_epoch = 0

    for epoch in range(num_epochs):
        final_epoch = epoch

        # train 
        model.train()
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)  # MSE
        train_loss.backward()
        optimizer.step()

        # compute train metrics
        train_mse = float(train_loss.item())
        train_rmse = train_mse ** 0.5
        train_mae = float(mae_from_pred(y_pred.detach(), y_train).item())
        train_r2 = float(r2_from_pred(y_pred.detach(), y_train).item())

        # validate 
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse_t = criterion(val_pred, y_val)  # MSE tensor
            val_mse = float(val_mse_t.item())
            val_rmse = val_mse ** 0.5
            val_mae = float(mae_from_pred(val_pred, y_val).item())
            val_r2 = float(r2_from_pred(val_pred, y_val).item())

        # save histories
        history["train_MSE"].append(train_mse)
        history["val_MSE"].append(val_mse)
        history["train_MAE"].append(train_mae)
        history["val_MAE"].append(val_mae)
        history["train_RMSE"].append(train_rmse)
        history["val_RMSE"].append(val_rmse)
        history["train_R2"].append(train_r2)
        history["val_R2"].append(val_r2)

        # early stopping metric 
        current_val = val_mse if early_stop_on.lower() == "mse" else val_mae

        if best_val - current_val > min_delta:
            best_val = current_val
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if print_every and (epoch % print_every == 0):
            print(
                f"Epoch {epoch:03d} | "
                f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
                f"Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch:03d}. "
                f"Best val ({early_stop_on.upper()}): {best_val:.6f}"
            )
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics (on best checkpoint)
    train_metrics = eval_metrics(model, X_train, y_train, y_scaler)
    val_metrics   = eval_metrics(model, X_val, y_val, y_scaler)
    test_metrics  = eval_metrics(model, X_test, y_test, y_scaler)

    if print_every:
        print(
            f"[BEST CKPT] | Scaled Space"
            f"Train MAE: {train_metrics['MAE']:.4f} | Val MAE: {val_metrics['MAE']:.4f} | Test MAE: {test_metrics['MAE']:.4f}"
        )

    return {
        "best_val_scaled": best_val,
        "early_stop_on": early_stop_on.lower(),
        "final_epoch": final_epoch,
        "epochs_run": final_epoch + 1,
        "history_scaled": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "y_scaler": y_scaler
    }


def fit_mlp_with_fixed_split(
    X: dict,
    ys,
    train_idx: np.array,
    test_idx: np.array,
    *,
    val_idx: np.array | None = None,
    seed: int = 42,
    dropout: float = 0.05,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 400,
    patience: int = 20,
    min_delta: float = 1e-5,
    device: str | None = None,
    print_every: int = 20,
    hidden1: int | None = None,
    hidden2: int | None = None,
) -> dict:
    """
    Train and evaluate a vanilla MLP on multiple embedding sets.

    Parameters
    ----------
    X : np.ndarrays
        embedding matrix (n_samples x n_features).
    ys : array-like
        Target values.
    train_idx, test_idx : array-like
        Indices defining train/test splits.
    get_tvt_fn : callable
        Function that constructs train/val/test tensors from embeddings and indices.
    seed : int, default=42
        Random seed for reproducibility.
    dropout : float, default=0.05
        Dropout probability in the MLP.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-4
        Weight decay for AdamW.
    num_epochs : int, default=400
        Maximum training epochs.
    patience : int, default=20
        Early stopping patience.
    min_delta : float, default=1e-5
        Minimum validation improvement threshold.
    device : str or None, optional
        Device to run on ('cpu' or 'cuda'). Auto-detected if None.
    print_every : int, default=20
        Logging frequency.
    hidden1, hidden2 : ignored
        Present for API compatibility; hidden sizes are fixed.

    Returns
    -------
    dict : training & ckpt info
    """

    seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Working on {device}")

    if val_idx is None:
        # val indecies are taken out from the train data - Test set remains untouched 
        train_idx, val_idx = split_train_into_train_val(
        train_idx,
        val_ratio=0.2,
        random_state=seed,
        )

    tvt, scaler = get_tvt(X, ys, train_idx, val_idx, test_idx)

    X_train = tvt["X_train"].to(device)
    y_train = tvt["y_train"].to(device)
    X_val   = tvt["X_val"].to(device)
    y_val   = tvt["y_val"].to(device)
    X_test  = tvt["X_test"].to(device)
    y_test  = tvt["y_test"].to(device)
    
    # Ensure float32 everywhere (sklearn/scipy often produce float64)
    X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)

    X_val   = torch.as_tensor(X_val,   dtype=torch.float32, device=device)
    y_val   = torch.as_tensor(y_val,   dtype=torch.float32, device=device)

    X_test  = torch.as_tensor(X_test,  dtype=torch.float32, device=device)
    y_test  = torch.as_tensor(y_test,  dtype=torch.float32, device=device)

    input_size = X_train.shape[1]

    model = VanillaMLP(
        input_size=input_size,
        hidden1=hidden1 if hidden1 is not None else input_size,
        hidden2=hidden2 if hidden2 is not None else max(1, input_size // 2),
        dropout=dropout,
    ).to(device=device, dtype=torch.float32)

    train_info = train_with_early_stopping(
        model,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        patience=patience,
        min_delta=min_delta,
        print_every=print_every
    )

    test_info = eval_metrics(
        model,
        X_test, y_test
    )

    return {
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "model_config": {
            "input_size": int(model.input_size),
            "hidden1": model.hidden1,
            "hidden2": model.hidden2,
            "dropout": float(model.dropout),
        },
        "train_info": train_info,
        "test_info": test_info,
        "seed": seed,
        "save_type": "torch",
        "torch_version": torch.__version__,
    }

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression

    # Generate synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    train_idx = np.arange(800)
    test_idx = np.arange(800, 1000)

    results = fit_mlp_with_fixed_split(
        X=X,
        ys=y,
        train_idx=train_idx,
        test_idx=test_idx,
        seed=42,
        dropout=0.05,
        lr=1e-3,
        weight_decay=1e-4,
        num_epochs=100,
        patience=10,
        min_delta=1e-5,
        print_every=10
    )

    print("Training and evaluation completed.")