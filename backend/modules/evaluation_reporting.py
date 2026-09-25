"""
=====================================
MODULE 6: Evaluation and Reporting Module
=====================================
Calculates performance metrics and generates reports/graphs.
Uses: Matplotlib, Pandas, Scikit-learn
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from datetime import datetime

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# SECTION A: Model Evaluation on Dataset
# ============================================================

def evaluate_model(model, le, scaler, X_test, y_test) -> dict:
    """
    Evaluate trained model on test set.
    
    Returns:
        dict: accuracy, precision, recall, f1, report
    """
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    classes = le.classes_
    report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)

    print("\n[Evaluation Module] Model Performance:")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "report": report,
        "y_pred": y_pred,
        "y_test": y_test,
        "classes": list(classes)
    }


# ============================================================
# SECTION B: Confusion Matrix Plot
# ============================================================

def plot_confusion_matrix(y_test, y_pred, classes: list, save: bool = True) -> str:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.5
    )
    ax.set_title('Confusion Matrix - Stress/Emotion Detection', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('Actual Label', fontsize=12)
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Confusion matrix saved: {path}")
    plt.close()
    return path


# ============================================================
# SECTION C: Performance Bar Chart
# ============================================================

def plot_performance_metrics(metrics: dict, save: bool = True) -> str:
    """Bar chart for Accuracy, Precision, Recall, F1."""
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.2%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_ylim(0, 1.15)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax.legend()
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(REPORTS_DIR, "performance_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Performance chart saved: {path}")
    plt.close()
    return path


# ============================================================
# SECTION D: Emotion Distribution Plot
# ============================================================

def plot_emotion_distribution(y_labels, classes: list, save: bool = True) -> str:
    """Show emotion distribution in dataset."""
    if hasattr(y_labels[0], 'item'):
        encoded = y_labels
        counts = np.bincount(encoded, minlength=len(classes))
        label_counts = {cls: int(counts[i]) for i, cls in enumerate(classes)}
    else:
        unique, counts = np.unique(y_labels, return_counts=True)
        label_counts = dict(zip(unique, counts.tolist()))

    colors = ['#e74c3c', '#e67e22', '#27ae60', '#3498db', '#9b59b6',
              '#1abc9c', '#f39c12', '#2c3e50']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = ax1.bar(list(label_counts.keys()), list(label_counts.values()),
                   color=colors[:len(label_counts)], edgecolor='white')
    ax1.set_title('Emotion Class Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Emotion')
    ax1.set_ylabel('Count')
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(int(bar.get_height())), ha='center', fontsize=9)

    # Pie chart
    ax2.pie(list(label_counts.values()), labels=list(label_counts.keys()),
            colors=colors[:len(label_counts)], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Emotion Distribution (%)', fontsize=12, fontweight='bold')

    plt.suptitle('Dataset Emotion Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = ""
    if save:
        path = os.path.join(REPORTS_DIR, "emotion_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Distribution chart saved: {path}")
    plt.close()
    return path


# ============================================================
# SECTION E: Single Prediction Report
# ============================================================

def generate_prediction_report(prediction_result: dict, save: bool = True) -> str:
    """
    Generate visual report for single audio prediction.
    
    Args:
        prediction_result: dict from app.py prediction
        save: Save as PNG
    """
    emotion      = prediction_result.get("emotion", "Unknown")
    stress       = prediction_result.get("stress_level", "Unknown")
    lie_prob     = prediction_result.get("lie_probability", "Unknown")
    lie_pct      = prediction_result.get("lie_percentage", "N/A")
    transcription = prediction_result.get("transcription", "")
    confidence   = prediction_result.get("confidence", 0)
    indicators   = prediction_result.get("indicators", [])

    color_map = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Urdu Speech Analysis Report', fontsize=16, fontweight='bold', y=1.02)

    # --- Emotion Panel ---
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    ax1.text(0.5, 0.65, emotion.upper(), ha='center', va='center',
             fontsize=22, fontweight='bold', color='#2c3e50', transform=ax1.transAxes)
    ax1.text(0.5, 0.35, f'Confidence: {confidence:.1%}', ha='center', va='center',
             fontsize=11, color='#7f8c8d', transform=ax1.transAxes)
    ax1.set_title('Emotion Detected', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # --- Stress Panel ---
    ax2 = axes[1]
    stress_color = color_map.get(stress, "#95a5a6")
    ax2.set_facecolor('#f8f9fa')
    circle = plt.Circle((0.5, 0.55), 0.3, color=stress_color, transform=ax2.transAxes)
    ax2.add_patch(circle)
    ax2.text(0.5, 0.55, stress.upper(), ha='center', va='center',
             fontsize=16, fontweight='bold', color='white', transform=ax2.transAxes)
    ax2.text(0.5, 0.18, 'Stress Level', ha='center', va='center',
             fontsize=10, color='#7f8c8d', transform=ax2.transAxes)
    ax2.set_title('Stress Detection', fontsize=13, fontweight='bold')
    ax2.axis('off')

    # --- Lie Panel ---
    ax3 = axes[2]
    lie_color = color_map.get(lie_prob, "#95a5a6")
    ax3.set_facecolor('#f8f9fa')
    ax3.text(0.5, 0.65, lie_prob.upper(), ha='center', va='center',
             fontsize=20, fontweight='bold', color=lie_color, transform=ax3.transAxes)
    ax3.text(0.5, 0.42, f'{lie_pct} probability', ha='center', va='center',
             fontsize=11, color='#7f8c8d', transform=ax3.transAxes)
    ax3.set_title('Lie Detection', fontsize=13, fontweight='bold')
    ax3.axis('off')

    plt.tight_layout()

    path = ""
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"prediction_report_{timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] Prediction report saved: {path}")
    plt.close()
    return path


# ============================================================
# SECTION F: Save CSV Summary Report
# ============================================================

def save_metrics_csv(metrics: dict) -> str:
    """Save metrics to CSV file."""
    data = {
        "Metric":    ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score":     [metrics['accuracy'], metrics['precision'],
                      metrics['recall'], metrics['f1_score']],
        "Percentage": [f"{v*100:.2f}%" for v in [
            metrics['accuracy'], metrics['precision'],
            metrics['recall'], metrics['f1_score']
        ]]
    }
    df = pd.DataFrame(data)
    path = os.path.join(REPORTS_DIR, "model_metrics.csv")
    df.to_csv(path, index=False)
    print(f"[Evaluation] Metrics CSV saved: {path}")
    return path
