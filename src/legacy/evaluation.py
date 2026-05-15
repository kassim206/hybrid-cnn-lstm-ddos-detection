import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    return metrics

def format_metrics_table(all_metrics):
    rows = []
    for model_name, metrics in all_metrics.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}"
        }
        if 'roc_auc' in metrics:
            row['AUC'] = f"{metrics['roc_auc']:.4f}"
        rows.append(row)
    
    return pd.DataFrame(rows)

def print_comparison_table(all_metrics):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 68)
    
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
    
    print("=" * 70)

def calculate_latency(model, X_test, batch_size=64):
    import time
    
    n_samples = X_test.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    latencies = []
    
    for i in range(0, n_samples, batch_size):
        batch = X_test[i:i+batch_size]
        
        start_time = time.perf_counter()
        
        if hasattr(model, 'predict'):
            model.predict(batch, verbose=0)
        else:
            model.predict(batch)
        
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f"\nInference Latency Analysis:")
    print(f"  Average latency per batch: {avg_latency:.2f} ms")
    print(f"  Standard deviation: {std_latency:.2f} ms")
    print(f"  Minimum: {np.min(latencies):.2f} ms")
    print(f"  Maximum: {np.max(latencies):.2f} ms")
    
    return {'avg_ms': avg_latency, 'std_ms': std_latency, 'min_ms': np.min(latencies), 'max_ms': np.max(latencies)}