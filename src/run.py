import numpy as np
import os
import pandas as pd  # ◄ Added this crucial line!
from utils import load_and_preprocess_data, standardize_features, generate_plots
from logistic_regression import LogisticRegression
from Decision_tree import DecisionTree
from knn import KNN
from metrics import accuracy, confusion_matrix, precision, recall, f1_score


def main():

    X_raw, y_train, feature_headers, raw_dataframe = load_and_preprocess_data('data.csv')
    
    X_train = standardize_features(X_raw)
    print(f"Shape of data : {X_train.shape}")
    
    generate_plots(raw_dataframe, X_train, y_train, feature_headers)
    
    model1 = LogisticRegression(learning_rate=0.01, iterations=1000)
    
    model1.fit(X_train, y_train)

    predictions = model1.predict(X_train)
    res1 = model1.evaluate(X_train, y_train)
    #output for lr
    print(f"\n================ LOGISTIC REGRESSION ================")
    print(f"Accuracy:  {res1['accuracy']:.4f}  |  Precision: {res1['precision']:.4f}")
    print(f"Recall:    {res1['recall']:.4f}  |  F1 Score:  {res1['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{res1['confusion_matrix']}")


    model2 = DecisionTree(max_depth=5)
    model2.fit(X_train, y_train)
    dt_predictions = model2.predict(X_train)
    res2 = model2.evaluate(X_train, y_train)
    #output for DecisionTree
    print(f"\n=================== DECISION TREE ===================")
    print(f"Accuracy:  {res2['accuracy']:.4f}  |  Precision: {res2['precision']:.4f}")
    print(f"Recall:    {res2['recall']:.4f}  |  F1 Score:  {res2['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{res2['confusion_matrix']}")
   

    knn_clf = KNN(k=3)
    knn_clf.fit(X_train, y_train)
    
    knn_predictions = knn_clf.predict(X_train)

    res3 = knn_clf.evaluate(X_train, y_train)
    #output for KNN
    print(f"\n================ K-NEAREST NEIGHBORS ================")
    print(f"Accuracy:  {res3['accuracy']:.4f}  |  Precision: {res3['precision']:.4f}")
    print(f"Recall:    {res3['recall']:.4f}  |  F1 Score:  {res3['f1_score']:.4f}")
    print(f"Confusion Matrix:\n{res3['confusion_matrix']}")
    
    # --------------------------------------------------
    # SAVE RESULTS TO FOLDER
    # --------------------------------------------------
    # Create the 'results' directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # 1. Save 'metrics_table.csv' using Pandas
    summary_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors'],
        'Accuracy': [res1['accuracy'], res2['accuracy'], res3['accuracy']],
        'Precision': [res1['precision'], res2['precision'], res3['precision']],
        'Recall': [res1['recall'], res2['recall'], res3['recall']],
        'F1 Score': [res1['f1_score'], res2['f1_score'], res3['f1_score']]
    }
    
    df_metrics = pd.DataFrame(summary_data)
    df_metrics.to_csv('results/metrics_table.csv', index=False)
    print("\n[SUCCESS] Created results/metrics_table.csv")

    # 2. Save 'experiment_log.txt' with detailed notes
    with open('results/experiment_log.txt', 'w') as log_file:
        log_file.write("=== ML Experiment Log ===\n")
        log_file.write(f"Dataset Shape: {X_train.shape}\n\n")
        
        log_file.write("--- Logistic Regression ---\n")
        log_file.write(f"Confusion Matrix:\n{res1['confusion_matrix']}\n\n")
        
        log_file.write("--- Decision Tree ---\n")
        log_file.write(f"Confusion Matrix:\n{res2['confusion_matrix']}\n\n")
        
        log_file.write("--- K-Nearest Neighbors ---\n")
        log_file.write(f"Confusion Matrix:\n{res3['confusion_matrix']}\n")
        
    print("[SUCCESS] Created results/experiment_log.txt")


if __name__ == "__main__":
    main()