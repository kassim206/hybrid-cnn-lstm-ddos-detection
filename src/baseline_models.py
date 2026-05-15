import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from src.config import (RF_N_ESTIMATORS, RF_MAX_DEPTH, SVM_C, SVM_KERNEL, SVM_GAMMA, 
                        MODELS_DIR, RANDOM_STATE)

def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH):
    print("\n" + "=" * 50)
    print("Training Random Forest Classifier")
    print("=" * 50)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    }
    
    print(f"\nRandom Forest Validation Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    model_path = f"{MODELS_DIR}/random_forest.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, metrics

def train_svm(X_train, y_train, X_val, y_val, C=SVM_C, kernel=SVM_KERNEL, gamma=SVM_GAMMA):
    print("\n" + "=" * 50)
    print("Training Support Vector Machine")
    print("=" * 50)
    
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=RANDOM_STATE,
        probability=True,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred)
    }
    
    print(f"\nSVM Validation Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    model_path = f"{MODELS_DIR}/svm.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, metrics

def evaluate_baseline_on_test(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"\n{model_name} Test Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    return metrics