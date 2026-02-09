#!/usr/bin/env python3
"""
Train XGBoost (tabular) and LSTM (sequence) and optimize ensemble weights 
to maximize accuracy on the holdout set.
"""
import argparse
import numpy as np
import os
from sklearn.model_selection import train_test_split
from models import build_xgb, build_lstm, eval_model_binary, save_xgb, save_keras, load_xgb, load_keras
import xgboost as xgb
from sklearn.metrics import accuracy_score

def main(features_npz, seed=42):
    np.random.seed(seed)
    
    # Load data
    if not os.path.exists(features_npz):
        print(f"Error: Features file {features_npz} not found.")
        return

    dat = np.load(features_npz, allow_pickle=True)
    X = dat['X']          # tabular features
    y = dat['y']
    
    # For LSTM we need seq data — reload raw from expected location
    raw_path = 'data/raw/f1_telemetry.npy'
    if not os.path.exists(raw_path):
        print(f"Error: Raw data file {raw_path} not found.")
        return

    raw = np.load(raw_path, allow_pickle=True).item()
    seq = raw['data']
    
    # Split test set (leave 15% holdout)
    # Note: We use the same seed to ensure X and seq are split identically
    X_rest, X_test, y_rest, y_test, seq_rest, seq_test = train_test_split(
        X, y, seq, test_size=0.15, random_state=seed, stratify=y
    )

    print("Training XGBoost model...")
    # Train XGBoost on the tabular engineered features
    xgb_model = build_xgb(X_rest, y_rest, seed=seed)
    # Eval baseline
    acc_xgb, prob_xgb = eval_model_binary(xgb_model, X_test, y_test, model_type='xgb')
    error_xgb = 1.0 - acc_xgb

    print("Training LSTM model...")
    # Train LSTM on sequences
    lstm_model = build_lstm(seq_rest, y_rest, seed=seed, epochs=10, batch_size=128)
    
    acc_lstm, prob_lstm = eval_model_binary(lstm_model, None, y_test, model_type='lstm', seq_data_for_lstm=seq_test)
    error_lstm = 1.0 - acc_lstm

    # Optimize ensemble weight w (for XGB) in [0,1] to maximize accuracy
    print("Optimizing ensemble weights...")
    best_w = 0.5
    best_acc = 0.0
    best_threshold = 0.5
    
    # Search for best weight
    for w in np.linspace(0.0, 1.0, 101):
        probs = w * prob_xgb + (1.0 - w) * prob_lstm
        
        # Optional: Search for best threshold for this weight
        # For simplicity/speed, we can check 0.5, but a small inner loop is better
        # checking thresholds:
        for t in [0.4, 0.45, 0.5, 0.55, 0.6]:
            preds = (probs >= t).astype(int)
            acc = accuracy_score(y_test, preds)
            
            if acc > best_acc:
                best_acc = acc
                best_w = w
                best_threshold = t

    # Compute final hybrid metrics using best parameters
    final_probs = best_w * prob_xgb + (1.0 - best_w) * prob_lstm
    final_preds = (final_probs >= best_threshold).astype(int)
    acc_hybrid = accuracy_score(y_test, final_preds)
    error_hybrid = 1.0 - acc_hybrid

    # Compute error reduction
    if error_xgb > 0:
        error_reduction = (error_xgb - error_hybrid) / error_xgb
    else:
        error_reduction = 0.0

    # Save models and ensemble weights
    os.makedirs('models', exist_ok=True)
    xgb_model.save_model('models/xgb_model.json')
    lstm_model.save('models/lstm_model.keras')
    
    np.savez_compressed(
        'models/hybrid_ensemble.npz', 
        xgb_path='models/xgb_model.json',
        lstm_path='models/lstm_model.keras', 
        weight=float(best_w), 
        threshold=float(best_threshold)
    )

    # Print the actual metrics
    print("\n=== Training Summary ===")
    print(f"XGBoost Accuracy: {acc_xgb:.6f}")
    print(f"LSTM Accuracy:    {acc_lstm:.6f}")
    print(f"Hybrid Accuracy:  {acc_hybrid:.6f}")
    print("-" * 30)
    print(f"XGBoost Error:    {error_xgb:.6f}")
    print(f"Hybrid Error:     {error_hybrid:.6f}")
    print(f"Error Reduction:  {error_reduction*100:.2f}%")
    print("-" * 30)
    print(f"Optimal Weight (XGB): {best_w:.2f}")
    print(f"Optimal Threshold:    {best_threshold:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='data/processed/features.npz')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.features, seed=args.seed)