#!/usr/bin/env python
"""
Stage 3 Survival Modeling: Stacking CoxMLP Ensemble (CpG + Clinical Biomarkers)

This script implements the CoxMLP stacking ensemble model using out-of-fold base-model
predictions generated on the combined CpG + clinical biomarkers feature set. It standardizes
predictions using Quantile Normalization and fits the PyTorch CoxMLP model.
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from joblib import dump, load

# Scikit-Learn
from sklearn.preprocessing import QuantileTransformer

# Lifelines
from lifelines.utils import concordance_index

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Configuration matching the paper settings
CONFIG = {
    "model": {
        "hidden_layers": [256, 128, 64],
        "dropout": 0.4,
        "lr": 4.396e-5,
        "weight_decay": 6.943e-5
    },
    "training": {
        "max_epochs": 150,
        "patience": 15,
        "seed": 42
    }
}

np.random.seed(CONFIG["training"]["seed"])
torch.manual_seed(CONFIG["training"]["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG["training"]["seed"])

RUN_ID = "V3_OPTUNA_SS"

# Resolve directories relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "saved_models"
PRED_DIR = REPO_ROOT / "predictions"
SCALER_DIR = REPO_ROOT / "scalers"
LOG_DIR = REPO_ROOT / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "stacking_combined.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

class SurvivalDataset(Dataset):
    """
    Dataset wrapper mapping features to time and event outcomes.
    """
    def __init__(self, df, feature_cols, time_col, event_col):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.time = torch.tensor(df[time_col].values, dtype=torch.float32)
        self.event = torch.tensor(df[event_col].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]

class CoxMLP(nn.Module):
    """
    Multilayer Perceptron for Cox Proportional Hazards survival prediction.
    """
    def __init__(self, input_dim, hidden_layers=[128, 64, 128], dropout=0.1):
        super().__init__()
        layers_list = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers_list.append(nn.Linear(in_dim, h_dim))
            layers_list.append(nn.ReLU())
            if dropout > 0:
                layers_list.append(nn.Dropout(dropout))
            in_dim = h_dim
        output_layer = nn.Linear(in_dim, 1, bias=False)
        nn.init.zeros_(output_layer.weight)
        layers_list.append(output_layer)
        self.network = nn.Sequential(*layers_list)
        
    def forward(self, x):
        return self.network(x)

def cox_ph_loss_sorted(risk_scores, events):
    """
    Calculates Cox partial likelihood loss on sorted samples.
    """
    risk_scores = risk_scores.view(-1)
    events = events.view(-1)
    exp_risk = torch.exp(risk_scores)
    risk_set_sum = torch.cumsum(exp_risk, dim=0)
    log_risk_set_sum = torch.log(risk_set_sum + 1e-7)
    uncensored_loss = events * (log_risk_set_sum - risk_scores)
    num_events = torch.sum(events) + 1e-7
    return torch.sum(uncensored_loss) / num_events

def cox_ph_loss(risk_scores, times, events):
    """
    Sorts datasets and calculates Cox PH loss.
    """
    times_sorted, indices = torch.sort(times, descending=True)
    risk_scores_sorted = risk_scores[indices]
    events_sorted = events[indices]
    return cox_ph_loss_sorted(risk_scores_sorted, events_sorted)

def compute_cindex(model, loader, device):
    """
    Evaluates model C-index performance.
    """
    model.eval()
    all_risk, all_time, all_event = [], [], []
    with torch.no_grad():
        for X, t, e in loader:
            X = X.to(device, non_blocking=True)
            risk_scores = model(X).detach().cpu().numpy().ravel()
            all_risk.extend(risk_scores)
            all_time.extend(t.numpy())
            all_event.extend(e.numpy())
    all_risk = np.array(all_risk)
    all_time = np.array(all_time)
    all_event = np.array(all_event)
    valid = ~np.isnan(all_risk)
    if valid.sum() < 10:
        return 0.5
    return concordance_index(all_time[valid], -all_risk[valid], all_event[valid])

def train_coxmlp_model(params, train_loader, val_loader, device, n_features, max_epochs=100, patience=10, whi_loader=None):
    """
    Trains the PyTorch model using early stopping.
    """
    model = CoxMLP(
        input_dim=n_features,
        hidden_layers=params["hidden_layers"],
        dropout=params["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_c = -np.inf
    best_state = None
    no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for X, t, e in train_loader:
            X = X.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            risk_scores = model(X)
            if torch.isnan(risk_scores).any():
                continue
            loss = cox_ph_loss(risk_scores, t, e)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()

        train_c = compute_cindex(model, train_loader, device)
        val_c = compute_cindex(model, val_loader, device)

        if val_c > best_val_c:
            best_val_c = val_c
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_c

def main():
    logger.info("Initializing Stacking Ensemble training...")
    try:
        train_df_stack = pd.read_csv(PRED_DIR / f"stacking_train_{RUN_ID}.csv")
        val_df_stack = pd.read_csv(PRED_DIR / f"stacking_val_{RUN_ID}.csv")
        test_df_stack = pd.read_csv(PRED_DIR / f"stacking_test_{RUN_ID}.csv")
        whi_df_stack = pd.read_csv(PRED_DIR / f"stacking_whi_{RUN_ID}.csv")
    except Exception as e:
        logger.error(f"Failed to load stacking input predictions: {e}. Please train base models first.")
        sys.exit(1)

    TIME_COL = "time"
    EVENT_COL = "event"
    EXCLUDE_COLS = [TIME_COL, EVENT_COL, "patient_id", "age", "sex"]
    FEATURE_COLS = [c for c in train_df_stack.columns if c not in EXCLUDE_COLS]

    logger.info("Applying Quantile Normalization to base model out-of-fold risk scores...")
    qn_transformer = QuantileTransformer(output_distribution="normal", random_state=CONFIG["training"]["seed"])
    train_df_stack[FEATURE_COLS] = qn_transformer.fit_transform(train_df_stack[FEATURE_COLS].values)
    val_df_stack[FEATURE_COLS] = qn_transformer.transform(val_df_stack[FEATURE_COLS].values)
    test_df_stack[FEATURE_COLS] = qn_transformer.transform(test_df_stack[FEATURE_COLS].values)
    whi_df_stack[FEATURE_COLS] = qn_transformer.transform(whi_df_stack[FEATURE_COLS].values)

    dump(qn_transformer, SCALER_DIR / f"qn_transformer_{RUN_ID}.joblib")

    # DataLoader setups
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = SurvivalDataset(train_df_stack, FEATURE_COLS, TIME_COL, EVENT_COL)
    val_dataset = SurvivalDataset(val_df_stack, FEATURE_COLS, TIME_COL, EVENT_COL)
    test_dataset = SurvivalDataset(test_df_stack, FEATURE_COLS, TIME_COL, EVENT_COL)
    whi_dataset = SurvivalDataset(whi_df_stack, FEATURE_COLS, TIME_COL, EVENT_COL)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, pin_memory=True)
    whi_loader = DataLoader(whi_dataset, batch_size=256, shuffle=False, pin_memory=True)

    # Train model
    model, best_val = train_coxmlp_model(
        CONFIG["model"], train_loader, val_loader, device, len(FEATURE_COLS),
        max_epochs=CONFIG["training"]["max_epochs"], patience=CONFIG["training"]["patience"]
    )

    # Save ensemble checkpoint
    torch.save(model.state_dict(), MODEL_DIR / f"coxmlp_surv_best_{RUN_ID}.pt")
    
    # Log results
    train_c = compute_cindex(model, train_loader, device)
    val_c = compute_cindex(model, val_loader, device)
    test_c = compute_cindex(model, test_loader, device)
    whi_c = compute_cindex(model, whi_loader, device)

    logger.info("=== STACKING ENSEMBLE RESULTS ===")
    logger.info(f"Train C-index : {train_c:.4f}")
    logger.info(f"Val C-index   : {val_c:.4f}")
    logger.info(f"Test C-index  : {test_c:.4f}")
    logger.info(f"WHI C-index   : {whi_c:.4f}")

    # Export predictions
    def save_predictions(loader, df, output_filename):
        model.eval()
        preds = []
        with torch.no_grad():
            for X, _, _ in loader:
                X = X.to(device)
                preds.extend(model(X).cpu().numpy().ravel())
        out_df = df.copy()
        out_df["risk_score"] = preds
        out_df.to_csv(PRED_DIR / output_filename, index=False)

    save_predictions(train_loader, train_df_stack, f"ensemble_train_{RUN_ID}.csv")
    save_predictions(val_loader, val_df_stack, f"ensemble_val_{RUN_ID}.csv")
    save_predictions(test_loader, test_df_stack, f"ensemble_test_{RUN_ID}.csv")
    save_predictions(whi_loader, whi_df_stack, f"ensemble_whi_{RUN_ID}.csv")

if __name__ == "__main__":
    main()
