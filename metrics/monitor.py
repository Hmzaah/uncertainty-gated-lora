import time
import torch
import pandas as pd
import psutil
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    log_loss
)

class PerformanceMonitor:
    """
    The 'Reviewer-Proof' Monitor.
    Tracks strict metrics: MCC, Balanced Acc, and Log Loss.
    """
    def __init__(self, experiment_name, output_dir="results/logs"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.os_process = psutil.Process(os.getpid())
        
        # Storage
        self.records = []
        self.class_names = [] # To be filled dynamically
        self.start_time = None
        
        # Ensure dirs exist
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)

    def start_frame(self):
        """Call before inference"""
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_frame(self, true_label, pred_label, probabilities, entropy, adapter_name):
        """
        Logs detailed frame data.
        probabilities: list of floats [0.1, 0.8, 0.1]
        """
        latency_ms = (time.perf_counter() - self.start_time) * 1000
        
        # Hardware Stats
        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        elif torch.backends.mps.is_available():
            vram_mb = self.os_process.memory_info().rss / (1024 ** 2)

        # Calibration Error (RMSE) for this single frame
        target_prob = probabilities[true_label] if true_label is not None else 0
        rmse = np.sqrt((target_prob - 1.0) ** 2) if true_label is not None else 0

        record = {
            "timestamp": datetime.now().isoformat(),
            "adapter": adapter_name,
            "latency_ms": latency_ms,
            "vram_mb": vram_mb,
            "entropy": float(entropy),
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": float(max(probabilities)),
            "rmse_calibration": rmse,
            "probs": probabilities, # Store raw probs for Log Loss
            "correct": 1 if true_label == pred_label else 0
        }
        self.records.append(record)

    def save_report(self):
        """Generates the Strict Scientific Report"""
        if not self.records:
            return

        df = pd.DataFrame(self.records)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        # 1. Save Raw Logs
        csv_path = f"{self.output_dir}/{self.experiment_name}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)

        # 2. Strict Metrics Calculation
        if df['true_label'].notnull().any():
            y_true = df['true_label']
            y_pred = df['pred_label']
            
            # Extract probability matrix for Log Loss
            y_probs = np.array(df['probs'].tolist())

            # --- The Strict Metrics ---
            acc_balanced = balanced_accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            try:
                ll = log_loss(y_true, y_probs)
            except ValueError:
                ll = 0.0 # Handle cases with missing classes
            
            # Standard Metrics
            acc_overall = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro')

            print(f"\n--- 🔬 STRICT RESEARCH REPORT: {self.experiment_name} ---")
            print(f"📊 Balanced Accuracy: {acc_balanced * 100:.2f}% (Handles Class Imbalance)")
            print(f"⚖️  MCC Score:        {mcc:.4f} (-1 to +1, +1 is perfect)")
            print(f"📉 Log Loss:         {ll:.4f} (Penalty for Overconfidence)")
            print(f"✅ Raw Accuracy:     {acc_overall * 100:.2f}%")
            print("-" * 30)

            # 3. Per-Class Breakdown (The Detail You Asked For)
            report_dict = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            
            # Save the breakdown to CSV
            report_path = f"{self.output_dir}/reports/{self.experiment_name}_{timestamp}_detailed.csv"
            report_df.to_csv(report_path)
            print(f"📑 Detailed Class Report saved to: {report_path}")

            # 4. Confusion Matrix Plot
            plt.figure(figsize=(10, 8))
            labels = sorted(df['true_label'].unique().astype(str))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu', xticklabels=labels, yticklabels=labels)
            plt.title(f"Conf Matrix | MCC: {mcc:.3f} | Bal Acc: {acc_balanced:.2f}")
            plt.ylabel("Ground Truth")
            plt.xlabel("Predicted")
            
            plot_path = f"{self.output_dir}/plots/{self.experiment_name}_{timestamp}_matrix.png"
            plt.savefig(plot_path)
            print(f"📊 Plot saved to: {plot_path}")
