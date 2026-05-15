import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.config import FIGURES_DIR

def set_style():
    """Set professional style for thesis graphs."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("Set2")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (10, 8)

def plot_confusion_matrix(cm, class_names, title, save_name):
    """Plot and save confusion matrix heatmap."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                square=True, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{FIGURES_DIR}/{save_name}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved confusion matrix: {save_name}")

def plot_training_history(history, save_name):
    """Plot and save training and validation curves."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#2ecc71')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#e74c3c')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#2ecc71')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#e74c3c')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{FIGURES_DIR}/{save_name}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved training history: {save_name}")

def plot_roc_curves(models_data, save_name):
    """Plot and save ROC curves comparison."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    for idx, (model_name, fpr, tpr, auc_score) in enumerate(models_data):
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})', color=color)
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{FIGURES_DIR}/{save_name}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved ROC curves: {save_name}")

def plot_feature_importance(importances, top_k, save_name):
    """Plot and save feature importance bar chart."""
    set_style()
    
    top_features = importances.head(top_k)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
    
    bars = ax.barh(range(top_k), top_features.values, color=colors[::-1], edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_features.index, fontsize=9)
    ax.set_xlabel('Feature Importance Score', fontsize=12)
    ax.set_title(f'Top {top_k} Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, top_features.values)):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{FIGURES_DIR}/{save_name}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved feature importance: {save_name}")

def plot_correlation_heatmap(corr_matrix, top_features, save_name):
    """Plot and save feature correlation heatmap."""
    set_style()
    
    filtered_corr = corr_matrix.loc[top_features, top_features]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(filtered_corr, dtype=bool))
    
    heatmap = sns.heatmap(
        filtered_corr, 
        mask=mask, 
        annot=True, 
        fmt='.2f', 
        cmap='RdBu_r', 
        center=0, 
        square=True, 
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}, 
        ax=ax
    )
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{FIGURES_DIR}/{save_name}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved correlation heatmap: {save_name}")

def plot_attack_distribution(attack_counts, save_name):
    """Plot and save attack type distribution bar chart."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors_list = []
    for name in attack_counts.index:
        name_lower = str(name).lower()
        if 'http' in name_lower or 'web' in name_lower or 'layer7' in name_lower:
            colors_list.append('#d62728')
        else:
            colors_list.append('#ff7f0e')
    
    bars = ax.bar(range(len(attack_counts)), attack_counts.values, color=colors_list, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(attack_counts)))
    ax.set_xticklabels(attack_counts.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_title('Distribution of DDoS Attack Types in CIC-DDoS2019 Dataset', fontsize=14, fontweight='bold', pad=20)
    
    max_val = max(attack_counts.values)
    for bar, count in zip(bars, attack_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='Layer 7 (HTTP/Web) Attacks'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Layer 3/4 Attacks')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{FIGURES_DIR}/{save_name}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved attack distribution: {save_name}")

def plot_bar_comparison(models_names, metrics_values, metric_name, save_name):
    """Plot and save bar chart comparing models on a specific metric."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(models_names, metrics_values, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'Model Comparison - {metric_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    for bar, val in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{FIGURES_DIR}/{save_name}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved bar comparison: {save_name}")

def plot_latency_comparison(models_names, latencies_ms, save_name):
    """Plot and save latency comparison bar chart."""
    set_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(models_names, latencies_ms, color=colors, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Inference Latency (ms per batch)', fontsize=12)
    ax.set_title('Real-Time Inference Latency Comparison', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, latencies_ms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{FIGURES_DIR}/{save_name}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved latency comparison: {save_name}")

def plot_resource_usage(cpu_data, ram_data, timestamps, save_name):
    """Plot and save CPU and RAM usage over time."""
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(timestamps, cpu_data, linewidth=2, color='#3498db')
    ax1.set_ylabel('CPU Usage (%)', fontsize=12)
    ax1.set_title('CPU Usage During Real-Time Detection', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    ax2.plot(timestamps, ram_data, linewidth=2, color='#2ecc71')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('RAM Usage (%)', fontsize=12)
    ax2.set_title('RAM Usage During Real-Time Detection', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{save_name}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{FIGURES_DIR}/{save_name}.pdf", format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved resource usage plot: {save_name}")