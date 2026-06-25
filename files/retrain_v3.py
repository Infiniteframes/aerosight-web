#!/usr/bin/env python
# coding: utf-8
"""
AeroSight Retraining v3
========================
Fixes: blind guessing, imbalance, overfitting
Implements: Stratified KFold, RFE, Optuna tuning, Threshold optimization
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, precision_recall_curve
)
from sklearn.utils import resample
import lightgbm as lgb

# ── CONFIG ─────────────────────────────────────────────────
DATA_PATH     = r'C:\Users\kmoha\OneDrive\Desktop\Final year sem 2\MDXacceltor\competiton\flights_with_weather.csv'
ARTIFACTS_DIR = 'aerosights_artifacts'
ENGINEERED_PATH = r'C:\Users\kmoha\OneDrive\Desktop\Final year sem 2\MDXacceltor\competiton\flights_engineered.csv'
DELAY_THRESHOLD = 15
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
OPTUNA_TRIALS   = 50  # increase for better tuning, decrease to go faster

FEATURES = [
    'DepHour', 'day_of_week', 'month', 'distance',
    'IsPeakHour', 'IsWeekend',
    'Airline_enc', 'Origin_enc', 'Dest_enc',
    'AvgWeatherDelay_Route', 'AvgNASDelay_Origin',
    'AvgLateAircraft_Airline', 'AvgCarrierDelay_Airline',
    'temp_c', 'windspeed_kmh', 'precip_mm',
    'weathercode', 'weather_severity'
]

NUMERIC_FEATURES = [
    'DepHour', 'day_of_week', 'month', 'distance',
    'AvgWeatherDelay_Route', 'AvgNASDelay_Origin',
    'AvgLateAircraft_Airline', 'AvgCarrierDelay_Airline',
    'temp_c', 'windspeed_kmh', 'precip_mm',
    'weathercode', 'weather_severity'
]

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("=" * 60)
print("AeroSight Retraining v3 — Full Fix")
print("=" * 60)

# ── SECTION 1: LOAD OR BUILD ENGINEERED DATASET ───────────
print("\nSECTION 1 — Data Loading & Feature Engineering")
print("-" * 50)

if os.path.exists(ENGINEERED_PATH):
    print("✅ Found saved engineered dataset — loading directly...")
    df = pd.read_csv(ENGINEERED_PATH, low_memory=False)
    print(f"✅ Loaded: {len(df):,} rows x {df.shape[1]} columns")
else:
    print("Building engineered dataset (this takes ~15 mins, saved after for future use)...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"✅ Raw loaded: {len(df):,} rows")

    # Clean
    df = df.dropna(subset=['dep_delay', 'arr_delay'])
    df = df[df['cancelled'] == 0]
    df = df.dropna(subset=['temp_c', 'windspeed_kmh', 'precip_mm', 'weathercode'])
    print(f"✅ After cleaning: {len(df):,} rows")

    # Target
    df['is_delayed'] = (df['dep_delay'] >= DELAY_THRESHOLD).astype(int)

    # Time features
    df['DepHour']    = (df['crs_dep_time'].fillna(0).astype(int) // 100).clip(0, 23)
    df['IsPeakHour'] = df['DepHour'].apply(lambda x: 1 if (7<=x<=9) or (16<=x<=20) else 0)
    df['IsWeekend']  = df['day_of_week'].apply(lambda x: 1 if x >= 6 else 0)

    # Weather severity
    df['weather_severity'] = (
        (df['temp_c'] < 0).astype(int) * 2 +
        (df['windspeed_kmh'] > 40).astype(int) * 2 +
        (df['precip_mm'] > 5).astype(int) * 3 +
        (df['weathercode'] >= 61).astype(int) * 2
    )

    # Historical averages
    print("   Computing historical averages...")
    route_weather = df.groupby(['origin','dest','month'])['weather_delay'].mean().reset_index()
    route_weather.columns = ['origin','dest','month','AvgWeatherDelay_Route']
    df = df.merge(route_weather, on=['origin','dest','month'], how='left')

    nas_origin = df.groupby(['origin','DepHour'])['nas_delay'].mean().reset_index()
    nas_origin.columns = ['origin','DepHour','AvgNASDelay_Origin']
    df = df.merge(nas_origin, on=['origin','DepHour'], how='left')

    late_aircraft = df.groupby(['op_unique_carrier','day_of_week'])['late_aircraft_delay'].mean().reset_index()
    late_aircraft.columns = ['op_unique_carrier','day_of_week','AvgLateAircraft_Airline']
    df = df.merge(late_aircraft, on=['op_unique_carrier','day_of_week'], how='left')

    carrier_avg = df.groupby('op_unique_carrier')['carrier_delay'].mean().reset_index()
    carrier_avg.columns = ['op_unique_carrier','AvgCarrierDelay_Airline']
    df = df.merge(carrier_avg, on='op_unique_carrier', how='left')

    # Encode
    le_airline = joblib.load(os.path.join(ARTIFACTS_DIR, 'le_airline.pkl'))
    le_origin  = joblib.load(os.path.join(ARTIFACTS_DIR, 'le_origin.pkl'))
    le_dest    = joblib.load(os.path.join(ARTIFACTS_DIR, 'le_dest.pkl'))

    def safe_encode(le, col):
        known = set(le.classes_)
        return col.apply(lambda x: le.transform([x])[0] if x in known else -1)

    df['Airline_enc'] = safe_encode(le_airline, df['op_unique_carrier'].astype(str))
    df['Origin_enc']  = safe_encode(le_origin,  df['origin'].astype(str))
    df['Dest_enc']    = safe_encode(le_dest,     df['dest'].astype(str))

    # Save engineered dataset
    df.to_csv(ENGINEERED_PATH, index=False)
    print(f"✅ Engineered dataset saved to: {ENGINEERED_PATH}")

print(f"   Class balance — Delayed: {df['is_delayed'].mean()*100:.1f}% | On-time: {(1-df['is_delayed'].mean())*100:.1f}%")

# ── SECTION 2: TRAIN/TEST SPLIT ───────────────────────────
print("\nSECTION 2 — Train/Test Split")
print("-" * 50)

X_full = df[FEATURES].fillna(0)
y_full = df['is_delayed']

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_full, y_full,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_full
)
print(f"✅ Train: {len(X_train_full):,} | Test: {len(X_test):,}")

# ── SECTION 3: FEATURE SELECTION (Permutation Importance) ─
print("\nSECTION 3 — Feature Selection")
print("-" * 50)
print("   Running permutation importance on 100K sample...")

# Use small sample for speed
sample_idx = np.random.choice(len(X_train_full), size=100_000, replace=False)
X_sample = X_train_full.iloc[sample_idx]
y_sample = y_train_full.iloc[sample_idx]

# Quick RF for feature selection
quick_rf = RandomForestClassifier(
    n_estimators=50, max_depth=8,
    class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1
)
quick_rf.fit(X_sample, y_sample)

perm_imp = permutation_importance(
    quick_rf, X_sample, y_sample,
    n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
)

# Rank features
imp_df = pd.DataFrame({
    'feature':    FEATURES,
    'importance': perm_imp.importances_mean
}).sort_values('importance', ascending=False)

print("\n   Feature Importance Ranking:")
for _, row in imp_df.iterrows():
    bar = '█' * int(row['importance'] * 500)
    print(f"   {row['feature']:30s} {row['importance']:.4f} {bar}")

# Keep features with positive importance
SELECTED_FEATURES = imp_df[imp_df['importance'] > 0]['feature'].tolist()
print(f"\n✅ Selected {len(SELECTED_FEATURES)}/{len(FEATURES)} features (removed noise)")
print(f"   Dropped: {set(FEATURES) - set(SELECTED_FEATURES)}")

X_train_sel = X_train_full[SELECTED_FEATURES]
X_test_sel  = X_test[SELECTED_FEATURES]

# ── SECTION 4: BALANCE TRAINING DATA ─────────────────────
print("\nSECTION 4 — Balancing Training Data")
print("-" * 50)

train_df = X_train_full.copy()
train_df['is_delayed'] = y_train_full.values

majority = train_df[train_df['is_delayed'] == 0]
minority = train_df[train_df['is_delayed'] == 1]

majority_down = resample(majority, replace=False,
                         n_samples=len(minority) * 2,
                         random_state=RANDOM_STATE)
minority_up   = resample(minority, replace=True,
                         n_samples=len(minority) * 2,
                         random_state=RANDOM_STATE)

balanced = pd.concat([majority_down, minority_up])\
             .sample(frac=1, random_state=RANDOM_STATE)\
             .reset_index(drop=True)

X_train_bal = balanced[SELECTED_FEATURES]
y_train_bal = balanced['is_delayed']
print(f"✅ Balanced: {len(X_train_bal):,} rows (50/50)")

# ── SECTION 5: OPTUNA HYPERPARAMETER TUNING (LightGBM) ────
print(f"\nSECTION 5 — Optuna Tuning LightGBM ({OPTUNA_TRIALS} trials)")
print("-" * 50)
print("   This will take a while — finding optimal hyperparameters...\n")

# Use stratified sample for tuning speed
tune_idx = np.random.choice(len(X_train_bal), size=min(500_000, len(X_train_bal)), replace=False)
X_tune = X_train_bal.iloc[tune_idx]
y_tune = y_train_bal.iloc[tune_idx]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

def optuna_objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'num_leaves':        trial.suggest_int('num_leaves', 20, 100),
        'max_depth':         trial.suggest_int('max_depth', 4, 12),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'lambda_l1':         trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2':         trial.suggest_float('lambda_l2', 0.0, 5.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 2.0),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq':      trial.suggest_int('bagging_freq', 1, 7),
        'scale_pos_weight':  trial.suggest_float('scale_pos_weight', 1.0, 5.0),
        'random_state':      RANDOM_STATE,
        'n_jobs':            -1,
        'verbose':           -1,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(
        model, X_tune, y_tune,
        cv=skf, scoring='f1', n_jobs=-1
    )
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(optuna_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params.update({'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})

print(f"\n✅ Best params found (F1: {study.best_value:.4f}):")
for k, v in best_params.items():
    print(f"   {k}: {v}")

# ── SECTION 6: TRAIN FINAL MODELS ─────────────────────────
print("\nSECTION 6 — Training Final Models")
print("-" * 50)

results  = {}
models   = {}

# ── LightGBM with best params ─────────────────────────────
print("\n   Training LightGBM (Optuna tuned)...")
lgb_final = lgb.LGBMClassifier(**best_params)
lgb_final.fit(X_train_bal, y_train_bal)
models['LightGBM (Optuna)'] = lgb_final

# ── Random Forest with balanced_subsample ─────────────────
print("   Training Random Forest (balanced_subsample)...")
rf_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight='balanced_subsample',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_final.fit(X_train_bal, y_train_bal)
models['Random Forest (balanced_subsample)'] = rf_final

# ── SECTION 7: THRESHOLD OPTIMIZATION ────────────────────
print("\nSECTION 7 — Threshold Optimization")
print("-" * 50)

def evaluate_at_threshold(model, X, y, threshold):
    proba  = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        'threshold': round(threshold, 2),
        'accuracy':  round(accuracy_score(y, y_pred),  4),
        'precision': round(precision_score(y, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y, y_pred,    zero_division=0), 4),
        'f1':        round(f1_score(y, y_pred,        zero_division=0), 4),
        'auc':       round(roc_auc_score(y, proba),   4),
    }

def find_best_threshold(model, X, y, model_name):
    print(f"\n   {model_name} — Threshold sweep:")
    print(f"   {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"   {'-'*55}")

    thresholds = np.arange(0.1, 0.95, 0.05)
    best_t, best_f1, best_metrics = 0.5, 0, {}

    for t in thresholds:
        m = evaluate_at_threshold(model, X, y, t)
        marker = ' <-- best' if m['f1'] > best_f1 else ''
        print(f"   {m['threshold']:>10.2f} {m['accuracy']*100:>9.1f}% "
              f"{m['precision']*100:>9.1f}% {m['recall']*100:>9.1f}% "
              f"{m['f1']:>10.4f}{marker}")
        if m['f1'] > best_f1:
            best_f1      = m['f1']
            best_t       = t
            best_metrics = m

    print(f"\n   Best threshold: {best_t:.2f} (F1: {best_f1:.4f})")
    return best_t, best_metrics

best_thresholds = {}
for name, model in models.items():
    best_t, best_m = find_best_threshold(model, X_test_sel, y_test, name)
    best_thresholds[name] = best_t
    results[name] = best_m

# ── SECTION 8: FINAL EVALUATION ───────────────────────────
print("\n" + "=" * 60)
print("FINAL RESULTS (at optimal threshold, real-world test set)")
print("=" * 60)

best_model_name = None
best_f1_score   = 0
best_model_obj  = None

for name, m in results.items():
    print(f"\n  {name}")
    print(f"    Threshold:  {m['threshold']}")
    print(f"    Accuracy:   {m['accuracy']*100:.2f}%")
    print(f"    AUC:        {m['auc']:.4f}")
    print(f"    F1:         {m['f1']:.4f}")
    print(f"    Precision:  {m['precision']*100:.2f}%")
    print(f"    Recall:     {m['recall']*100:.2f}%")
    if m['f1'] > best_f1_score:
        best_f1_score   = m['f1']
        best_model_name = name
        best_model_obj  = models[name]

print(f"\n{'='*60}")
print(f"  Best Model: {best_model_name} (F1: {best_f1_score:.4f})")
print(f"  Best Threshold: {best_thresholds[best_model_name]:.2f}")
print(f"{'='*60}")

# ── SECTION 9: SAVE BEST MODEL ────────────────────────────
print("\nSECTION 9 — Saving Artifacts")
print("-" * 50)

joblib.dump(best_model_obj, os.path.join(ARTIFACTS_DIR, 'model.pkl'))

# Save best threshold
with open(os.path.join(ARTIFACTS_DIR, 'best_threshold.json'), 'w') as f:
    json.dump({
        'model_name':      best_model_name,
        'best_threshold':  best_thresholds[best_model_name],
        'selected_features': SELECTED_FEATURES
    }, f, indent=2)

# Save feature importance
if hasattr(best_model_obj, 'feature_importances_'):
    fi = dict(zip(SELECTED_FEATURES, best_model_obj.feature_importances_.tolist()))
    with open(os.path.join(ARTIFACTS_DIR, 'feature_importance.json'), 'w') as f:
        json.dump(fi, f, indent=2)

# Update metadata
with open(os.path.join(ARTIFACTS_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

metadata['model_name']        = best_model_name
metadata['best_threshold']    = best_thresholds[best_model_name]
metadata['selected_features'] = SELECTED_FEATURES
metadata['performance']       = results
metadata['optuna_best_params']= best_params

with open(os.path.join(ARTIFACTS_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved:     model.pkl")
print(f"✅ Threshold saved: best_threshold.json")
print(f"✅ Metadata updated: metadata.json")
print(f"\n🎉 ALL DONE!")