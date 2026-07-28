#!/usr/bin/env python
"""
Stage 3 Survival Modeling: Base Models Training (CpG + Clinical Biomarkers)

This script trains five base survival models (MTLR, CoxNet, Weibull-AFT, RSF, SVM)
on the combined CpG + clinical feature set. It uses Optuna for hyperparameter optimization,
performs cross-validation to generate out-of-fold risk scores, and fits the
final base models on the entire training cohort.
"""

import os
import sys
import gc
import json
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from joblib import dump, load

# Scikit-Learn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Scikit-Survival
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored

# Lifelines
from lifelines import WeibullAFTFitter

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Pycox
import torch
import torch.nn as nn
import torchtuples as tt
from torchtuples import optim as ttoptim
from torchtuples.callbacks import EarlyStopping
from pycox.models import MTLR as PycoxMTLR
from torchtuples.practical import MLPVanilla

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

RUN_ID = "V3_OPTUNA_SS"

# Resolve directories relative to repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
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
        logging.FileHandler(LOG_DIR / "pipeline_combined.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

N_SPLITS = 5

def to_structured_array(y_df):
    """
    Converts event and time columns to structured array format for scikit-survival.
    """
    return np.array(
        list(zip(y_df["event"], y_df["time"])),
        dtype=[("event", "?"), ("time", "<f8")]
    )

class SurvivalPreprocessor:
    """
    Handles training features standardization.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        self.scaler.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X):
        assert self.is_fitted, "Preprocessor must be fitted first."
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path):
        dump(self.scaler, path)

@dataclass
class TrainingResult:
    model_name: str
    oof_risk: np.ndarray
    fold_cis: List[float]
    refit_risk_train: np.ndarray
    refit_risk_val: np.ndarray
    refit_risk_test: np.ndarray
    refit_risk_whi: np.ndarray
    model_adapter: Any
    preprocessor: Any

class BaseSurvivalModelAdapter:
    def fit(self, X, y_df):
        raise NotImplementedError

    def predict_risk(self, X):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

class SkSurvAdapter(BaseSurvivalModelAdapter):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y_df):
        y_surv = to_structured_array(y_df)
        self.model.fit(X, y_surv)
        return self

    def predict_risk(self, X):
        return self.model.predict(X)

    def save(self, path):
        dump(self.model, path)

class WeibullAdapter(BaseSurvivalModelAdapter):
    def __init__(self, **kwargs):
        self.model = WeibullAFTFitter(**kwargs)

    def fit(self, X, y_df):
        df = X.copy()
        df["time"] = y_df["time"]
        df["event"] = y_df["event"].astype(int)
        try:
            self.model.fit(df, duration_col="time", event_col="event", fit_options={"step_size": 0.1})
        except Exception as e:
            logger.warning(f"Weibull model fitting failed: {e}")
        return self

    def predict_risk(self, X):
        if not hasattr(self.model, "params_"):
            return np.zeros(len(X))
        return -self.model.predict_median(X).values.ravel()

    def save(self, path):
        dump(self.model, path)

class MTLRAdapter(BaseSurvivalModelAdapter):
    def __init__(self, num_durations=35, layers=(32, 16), dropout=0.35, batchnorm=False, lr=3e-4, batch_size=32, epochs=400, patience=50, verbose=False, weight_decay=0.0, use_quantile_cuts=True):
        self.num_durations = num_durations
        self.layers = layers
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.use_quantile_cuts = use_quantile_cuts
        self.labtrans = None
        self.net = None
        self.model = None

    def _get_mtlr_target(self, y_df, labtrans=None):
        durations = y_df["time"].values.astype(np.float32)
        events = y_df["event"].values.astype(np.float32)
        if labtrans is None:
            if self.use_quantile_cuts:
                event_times = durations[events == 1]
                if len(event_times) > 0:
                    cuts = np.unique(np.quantile(event_times, np.linspace(0, 1, self.num_durations)))
                else:
                    cuts = self.num_durations
                labtrans = PycoxMTLR.label_transform(cuts)
            else:
                labtrans = PycoxMTLR.label_transform(self.num_durations)
            targets = labtrans.fit_transform(durations, events)
            return targets, labtrans
        else:
            targets = labtrans.transform(durations, events)
            return targets

    def fit(self, X, y_df, X_val=None, y_val_df=None):
        import random
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        X_np = X.values.astype(np.float32)
        y_mtlr, self.labtrans = self._get_mtlr_target(y_df)
        in_features = X_np.shape[1]
        self.in_features = in_features
        out_features = self.labtrans.out_features
        self.net = MLPVanilla(in_features, self.layers, out_features, self.batchnorm, self.dropout)
        optimizer = ttoptim.Adam(lr=self.lr, weight_decay=self.weight_decay)
        self.model = PycoxMTLR(self.net, optimizer)

        callbacks = []
        val_data = None
        if X_val is not None and y_val_df is not None:
            X_val_np = X_val.values.astype(np.float32)
            y_val_mtlr = self._get_mtlr_target(y_val_df, self.labtrans)
            val_data = (X_val_np, y_val_mtlr)
            callbacks = [EarlyStopping(patience=self.patience)]

        self.model.fit(X_np, y_mtlr, self.batch_size, self.epochs, callbacks, self.verbose, val_data=val_data)
        return self

    def predict_risk(self, X):
        X_np = X.values.astype(np.float32)
        surv_df = self.model.predict_surv_df(X_np)
        return -surv_df.mean().values

    def save(self, path):
        bundle = {
            'state_dict': self.net.state_dict(),
            'labtrans_cuts': self.labtrans.cuts,
            'config': {
                'num_durations': self.num_durations,
                'layers': self.layers,
                'dropout': self.dropout,
                'batchnorm': self.batchnorm,
                'in_features': self.in_features,
                'out_features': self.labtrans.out_features,
                'use_quantile_cuts': self.use_quantile_cuts
            }
        }
        torch.save(bundle, path)

def tune_model_hyperparameters(model_name: str, X_train: pd.DataFrame, y_train: pd.DataFrame, n_trials: int = 12) -> Dict[str, Any]:
    """
    Optimizes base model hyper-parameters via Optuna framework.
    """
    logger.info(f"Optimizing hyperparameters for {model_name.upper()} via Optuna...")
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        if model_name == "mtlr":
            lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.4)
            num_durations = trial.suggest_int("num_durations", 35, 70, step=5)
            adapter = MTLRAdapter(
                num_durations=num_durations, layers=(), dropout=dropout, batchnorm=True,
                lr=lr, batch_size=64, epochs=120, patience=15, weight_decay=weight_decay,
                use_quantile_cuts=True, verbose=False
            )
        elif model_name == "coxnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 1.0)
            alpha_min_ratio = trial.suggest_float("alpha_min_ratio", 0.001, 0.05, log=True)
            adapter = SkSurvAdapter(CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=alpha_min_ratio, fit_baseline_model=True))
        elif model_name == "weibull_aft":
            penalizer = trial.suggest_float("penalizer", 0.0005, 0.5, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
            adapter = WeibullAdapter(penalizer=penalizer, l1_ratio=l1_ratio)
        elif model_name == "rsf":
            max_depth = trial.suggest_int("max_depth", 2, 5)
            min_samples_split = trial.suggest_int("min_samples_split", 40, 80)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 20, 45)
            adapter = SkSurvAdapter(RandomSurvivalForest(
                n_estimators=100, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                n_jobs=-1, random_state=SEED
            ))
        elif model_name == "svm":
            gamma = trial.suggest_float("gamma", 0.005, 0.05, log=True)
            alpha = trial.suggest_float("alpha", 0.00005, 0.005, log=True)
            adapter = SkSurvAdapter(FastKernelSurvivalSVM(
                kernel="rbf", gamma=gamma, alpha=alpha, rank_ratio=1.0, max_iter=1500,
                tol=1e-4, random_state=SEED
            ))
        else:
            raise ValueError(f"Model {model_name} not supported.")

        fold_cis = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train["event"]), 1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            preprocessor = SurvivalPreprocessor()
            X_tr_s = preprocessor.fit_transform(X_tr)
            X_va_s = preprocessor.transform(X_va)

            if isinstance(adapter, MTLRAdapter):
                adapter.fit(X_tr_s, y_tr, X_val=X_va_s, y_val_df=y_va)
            else:
                adapter.fit(X_tr_s, y_tr)

            risk_va = adapter.predict_risk(X_va_s)
            ci_fold = concordance_index_censored(y_va["event"].values.astype(bool), y_va["time"].values, risk_va)[0]
            fold_cis.append(ci_fold)

        return np.mean(fold_cis)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f" -> Optuna optimization complete. Best fold C-index: {study.best_value:.4f}")
    return study.best_params

def train_evaluate_and_refit(model_name: str, best_params: Dict[str, Any], X_train, y_train, X_val, y_val, X_test, y_test, X_whi, y_whi):
    """
    Fits model via CV on training cohort, logs metrics, and fits final base learner on entire train set.
    """
    logger.info(f"=== Final cross-validation and fit for {model_name.upper()} ===")

    def adapter_factory():
        if model_name == "mtlr":
            return MTLRAdapter(
                num_durations=int(best_params.get("num_durations", 60)),
                layers=(),
                dropout=float(best_params.get("dropout", 0.0)),
                batchnorm=True,
                lr=float(best_params.get("lr", 0.00019)),
                batch_size=64,
                epochs=400,
                patience=32,
                weight_decay=float(best_params.get("weight_decay", 0.00008)),
                use_quantile_cuts=True
            )
        elif model_name == "coxnet":
            return SkSurvAdapter(CoxnetSurvivalAnalysis(
                l1_ratio=float(best_params.get("l1_ratio", 0.5)),
                alpha_min_ratio=float(best_params.get("alpha_min_ratio", 0.01)),
                fit_baseline_model=True
            ))
        elif model_name == "weibull_aft":
            return WeibullAdapter(
                penalizer=float(best_params.get("penalizer", 0.01)),
                l1_ratio=float(best_params.get("l1_ratio", 0.5))
            )
        elif model_name == "rsf":
            return SkSurvAdapter(RandomSurvivalForest(
                n_estimators=200,
                max_depth=int(best_params.get("max_depth", 3)),
                min_samples_split=int(best_params.get("min_samples_split", 60)),
                min_samples_leaf=int(best_params.get("min_samples_leaf", 30)),
                n_jobs=-1,
                random_state=SEED
            ))
        elif model_name == "svm":
            return SkSurvAdapter(FastKernelSurvivalSVM(
                kernel="rbf",
                gamma=float(best_params.get("gamma", 0.01)),
                alpha=float(best_params.get("alpha", 0.0001)),
                rank_ratio=1.0,
                max_iter=5000,
                tol=1e-4,
                random_state=SEED
            ))
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_risk = np.zeros(len(X_train), dtype=float)
    fold_cis = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train["event"]), 1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        preprocessor = SurvivalPreprocessor()
        X_tr_s = preprocessor.fit_transform(X_tr)
        X_va_s = preprocessor.transform(X_va)

        adapter = adapter_factory()
        if isinstance(adapter, MTLRAdapter):
            adapter.fit(X_tr_s, y_tr, X_val=X_va_s, y_val_df=y_va)
        else:
            adapter.fit(X_tr_s, y_tr)

        risk_va = adapter.predict_risk(X_va_s)
        oof_risk[va_idx] = risk_va

        ci_fold = concordance_index_censored(y_va["event"].values.astype(bool), y_va["time"].values, risk_va)[0]
        fold_cis.append(ci_fold)

    mean_ci = np.mean(fold_cis)
    std_ci = np.std(fold_cis)
    oof_ci = concordance_index_censored(y_train["event"].values.astype(bool), y_train["time"].values, oof_risk)[0]

    logger.info(f"[{model_name.upper()}] Mean CV C-index : {mean_ci:.4f} \u00b1 {std_ci:.4f}")
    logger.info(f"[{model_name.upper()}] OOF Train C-index : {oof_ci:.4f}")

    # Full Train Refit
    logger.info(f"Refitting {model_name.upper()} on full train cohort...")
    final_preprocessor = SurvivalPreprocessor()
    X_train_s = final_preprocessor.fit_transform(X_train)
    X_val_s = final_preprocessor.transform(X_val)
    X_test_s = final_preprocessor.transform(X_test)
    X_whi_s = final_preprocessor.transform(X_whi)

    final_adapter = adapter_factory()
    final_adapter.fit(X_train_s, y_train)

    risk_train = final_adapter.predict_risk(X_train_s)
    risk_val = final_adapter.predict_risk(X_val_s)
    risk_test = final_adapter.predict_risk(X_test_s)
    risk_whi = final_adapter.predict_risk(X_whi_s)

    ci_train = concordance_index_censored(y_train["event"].values.astype(bool), y_train["time"].values, risk_train)[0]
    ci_val = concordance_index_censored(y_val["event"].values.astype(bool), y_val["time"].values, risk_val)[0]
    ci_test = concordance_index_censored(y_test["event"].values.astype(bool), y_test["time"].values, risk_test)[0]
    ci_whi = concordance_index_censored(y_whi["event"].values.astype(bool), y_whi["time"].values, risk_whi)[0]

    logger.info(f"[{model_name.upper()}] Refit Train C-index: {ci_train:.4f}")
    logger.info(f"[{model_name.upper()}] Val C-index       : {ci_val:.4f}")
    logger.info(f"[{model_name.upper()}] Test C-index      : {ci_test:.4f}")
    logger.info(f"[{model_name.upper()}] WHI C-index       : {ci_whi:.4f}")

    # Save model artifacts into saved_models/
    out_dir = MODEL_DIR / f"{model_name}_{RUN_ID}"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_adapter.save(out_dir / f"{model_name}_{RUN_ID}.model")
    final_preprocessor.save(SCALER_DIR / f"{model_name}_scaler_{RUN_ID}.joblib")

    return TrainingResult(
        model_name=model_name, oof_risk=oof_risk, fold_cis=fold_cis,
        refit_risk_train=risk_train, refit_risk_val=risk_val,
        refit_risk_test=risk_test, refit_risk_whi=risk_whi,
        model_adapter=final_adapter, preprocessor=final_preprocessor
    )

def main():
    logger.info("Loading preprocessed datasets...")
    try:
        fhs_data = pd.read_csv(DATA_DIR / "FHS_non_normalized.csv", low_memory=False)
        fhs_data = fhs_data.rename(columns={"Share_ID": "patient_id"})
        whi_data = pd.read_csv(DATA_DIR / "WHI_non_normalized.csv", low_memory=False)
        whi_data = whi_data.rename(columns={"SUBJECT_ID": "patient_id"})
    except Exception as e:
        logger.error(f"Failed to load data files: {e}")
        sys.exit(1)

    DROP_COLS = ["patient_id", "SUBJECT_ID", "Share_ID", "dbGaP_Subject_ID", "time", "event", "cohort", "batch", "age", "sex"]
    fhs_features = [c for c in fhs_data.columns if c not in DROP_COLS]
    whi_features = [c for c in whi_data.columns if c not in DROP_COLS]
    common_features = sorted(list(set(fhs_features).intersection(set(whi_features))))
    logger.info(f"Found {len(common_features)} common features.")

    FHS = fhs_data[["patient_id", "time", "event", "age", "sex"] + common_features].copy()
    WHI = whi_data[["patient_id", "time", "event", "age", "sex"] + common_features].copy()

    FHS["event_stratify"] = FHS["event"].astype(int)
    train_val_df, test_df = train_test_split(FHS, test_size=0.20, random_state=SEED, stratify=FHS["event_stratify"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=SEED, stratify=train_val_df["event_stratify"])

    train_df = train_df.drop(columns=["event_stratify"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["event_stratify"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["event_stratify"]).reset_index(drop=True)
    FHS = FHS.drop(columns=["event_stratify"])

    def extract_components(df):
        meta = df[["patient_id", "age", "sex"]].copy()
        y_df = df[["event", "time"]].copy()
        y_df["event"] = y_df["event"].astype(bool)
        X = df[common_features].copy()
        return X, y_df, meta

    X_train, y_train, meta_train = extract_components(train_df)
    X_val, y_val, meta_val = extract_components(val_df)
    X_test, y_test, meta_test = extract_components(test_df)
    X_whi, y_whi, meta_whi = extract_components(WHI)

    results = {}
    best_configs = {}

    model_names = ["mtlr", "coxnet", "weibull_aft", "rsf", "svm"]
    for m_name in model_names:
        n_trials = 10 if m_name == "mtlr" else 15 if m_name in ["coxnet", "weibull_aft"] else 12
        best_configs[m_name] = tune_model_hyperparameters(m_name, X_train, y_train, n_trials=n_trials)
        results[m_name] = train_evaluate_and_refit(
            m_name, best_configs[m_name], X_train, y_train, X_val, y_val, X_test, y_test, X_whi, y_whi
        )

    # Save best parameters to predictions/ folder
    best_configs_path = PRED_DIR / f"optuna_best_configs_{RUN_ID}.json"
    serializable_configs = {}
    for k, v in best_configs.items():
        serializable_configs[k] = {pk: float(pv) if isinstance(pv, (np.floating, float)) else int(pv) if isinstance(pv, (np.integer, int)) else pv for pk, pv in v.items()}
    with open(best_configs_path, "w") as f:
        json.dump(serializable_configs, f, indent=2)

    # Export Stacking datasets
    def export_stacking_datasets(results_dict, meta, y_df, split_name):
        logger.info(f"Exporting {split_name} stacking datasets...")
        preds = {}
        for m_name, res in results_dict.items():
            if split_name == "train":
                preds[m_name] = res.oof_risk
            elif split_name == "val":
                preds[m_name] = res.refit_risk_val
            elif split_name == "test":
                preds[m_name] = res.refit_risk_test
            elif split_name == "whi":
                preds[m_name] = res.refit_risk_whi

        pred_df = pd.DataFrame(preds)
        meta_df = meta.reset_index(drop=True).copy()
        y_df_clean = y_df.reset_index(drop=True)
        out_df = pd.concat([meta_df, y_df_clean, pred_df], axis=1)
        out_path = PRED_DIR / f"stacking_{split_name}_{RUN_ID}.csv"
        out_df.to_csv(out_path, index=False)

        for m_name in preds.keys():
            model_out_df = pd.concat([meta_df, y_df_clean, pred_df[[m_name]]], axis=1)
            model_out_df = model_out_df.rename(columns={m_name: "risk_score"})
            m_out_path = PRED_DIR / f"{m_name}_{split_name}_{RUN_ID}.csv"
            model_out_df.to_csv(m_out_path, index=False)

    export_stacking_datasets(results, meta_train, y_train, "train")
    export_stacking_datasets(results, meta_val, y_val, "val")
    export_stacking_datasets(results, meta_test, y_test, "test")
    export_stacking_datasets(results, meta_whi, y_whi, "whi")

    # Feature Importance
    feature_importance_records = []
    final_preprocessor = results["coxnet"].preprocessor
    X_train_s = final_preprocessor.transform(X_train)

    class _AdapterWrapper:
        def __init__(self, adapter):
            self.adapter = adapter
        def fit(self, X, y):
            return self
        def predict(self, X):
            return self.adapter.predict_risk(X)
        def score(self, X, y):
            risk = self.predict(X)
            return concordance_index_censored(y["event"], y["time"], risk)[0]

    for m_name, res in results.items():
        adapter = res.model_adapter
        if isinstance(adapter, SkSurvAdapter):
            estimator = adapter.model
        else:
            estimator = _AdapterWrapper(adapter)

        try:
            n_jobs = 1 if m_name == "mtlr" else -1
            imp = permutation_importance(
                estimator, X_train_s, to_structured_array(y_train),
                n_repeats=5, random_state=SEED, n_jobs=n_jobs
            )
            for i, col in enumerate(X_train.columns):
                feature_importance_records.append({
                    "Model": m_name.upper(),
                    "Feature": col,
                    "Importance": imp.importances_mean[i],
                    "Std": imp.importances_std[i]
                })
        except Exception as e:
            logger.warning(f"Permutation importance failed for {m_name}: {e}")

    if feature_importance_records:
        df_fi = pd.DataFrame(feature_importance_records)
        df_fi.to_csv(PRED_DIR / f"base_model_feature_importance_{RUN_ID}.csv", index=False)

    # Performance Summary
    summary_records = []
    for m_name, res in results.items():
        summary_records.append({
            "Model": m_name.upper(),
            "Train": concordance_index_censored(y_train["event"].values.astype(bool), y_train["time"].values, res.refit_risk_train)[0],
            "Val": concordance_index_censored(y_val["event"].values.astype(bool), y_val["time"].values, res.refit_risk_val)[0],
            "Test": concordance_index_censored(y_test["event"].values.astype(bool), y_test["time"].values, res.refit_risk_test)[0],
            "WHI": concordance_index_censored(y_whi["event"].values.astype(bool), y_whi["time"].values, res.refit_risk_whi)[0]
        })
    df_summary = pd.DataFrame(summary_records).set_index("Model")
    df_summary.to_csv(PRED_DIR / f"base_model_cindex_summary_{RUN_ID}.csv")

if __name__ == "__main__":
    main()
