import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from src.config import N_TOP_FEATURES, CORRELATION_THRESHOLD

def select_features_mutual_information(X_train, y_train, k=N_TOP_FEATURES):
    scores = mutual_info_classif(X_train, y_train, random_state=42)
    feature_scores = pd.Series(scores, index=X_train.columns).sort_values(ascending=False)
    
    selected_features = feature_scores.head(k).index.tolist()
    scores_dict = feature_scores.head(k).to_dict()
    
    print(f"\nMutual Information – Top {k} features:")
    for feat, score in list(scores_dict.items())[:10]:
        print(f"  {feat}: {score:.4f}")
    
    return selected_features, scores_dict

def select_features_anova(X_train, y_train, k=N_TOP_FEATURES):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    
    scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
    selected_features = scores.head(k).index.tolist()
    
    print(f"\nANOVA F-test – Top {k} features:")
    for feat, score in list(scores.head(10).items()):
        print(f"  {feat}: {score:.2f}")
    
    return selected_features, scores.to_dict()

def remove_highly_correlated_features(X_train, threshold=CORRELATION_THRESHOLD):
    corr_matrix = X_train.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    dropped_features = []
    for column in upper_triangle.columns:
        if column in dropped_features:
            continue
        correlated = [col for col in upper_triangle.index 
                     if upper_triangle.loc[col, column] > threshold and col != column]
        dropped_features.extend(correlated)
    
    dropped_features = list(set(dropped_features))
    kept_features = [col for col in X_train.columns if col not in dropped_features]
    
    print(f"\nCorrelation analysis – {threshold} threshold:")
    print(f"  Kept: {len(kept_features)} features")
    print(f"  Removed: {len(dropped_features)} features due to high correlation")
    
    return kept_features, dropped_features

def compute_feature_importance_rf(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    
    print("\nRandom Forest Feature Importance (Top 10):")
    for feat, imp in importances.head(10).items():
        print(f"  {feat}: {imp:.4f}")
    
    return importances