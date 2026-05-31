# Breast Cancer Diagnosis Classification From Scratch

A machine learning project implementing classical ML algorithms from scratch without using scikit-learn to classify breast cancer tumors.

## 📌 Project Overview

This project aims to classify breast cancer tumors as benign or malignant using handcrafted machine learning algorithms and a fully modular machine learning pipeline.

**Implemented algorithms:**
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree

**The project includes:**
* Data preprocessing and feature correlation analysis
* Evaluation metrics built entirely from scratch
* Structured experiment logging and performance tracking
* Visualizations for decision boundaries and metric comparison

## 📂 Project Structure

```text
breast_cancer/
│
├── data/
│
├── figures/
│   ├── 1_feature_distributions.png
│   ├── 2_class_scatter.png
│   ├── 3_decision_boundary.png
│   ├── data_distribution.png
│   ├── feature_analysis.png
│   ├── loss_curve.png
│   └── model_comparison.png
│
├── notebooks/
│   └── exploration.ipynb
│
├── results/
│   ├── experiment_log.txt
│   └── metrics_table.csv
│
├── src/
│   ├── __init__.py
│   ├── Decision_tree.py
│   ├── knn.py
│   ├── logistic_regression.py
│   ├── metrics.py
│   ├── run.py
│   └── utils.py
├── README.md
├── report.ipynb
└── requirements.txt

 Implemented Models
Logistic Regression
  Implemented using gradient descent optimization.
  Utilizes sigmoid activation mapping for probability estimations.
  Optimized via custom binary cross-entropy loss tracking.
K-Nearest Neighbors (KNN)
  Implemented using vectorized Euclidean distance computations.
  Utilizes dynamic majority voting over local parameter spaces.
Decision Tree
  Implemented using Shannon entropy calculations.
  Splits tree branches based on maximizing localized Information Gain.
  Supports recursive node building down to customizable depth parameters.
📊 Evaluation Metrics
Implemented completely from scratch without external library imports:
-Accuracy
-Precision
-Recall
-F1_score
Confusion Matrix matrices
📈 VisualizationsGenerated figures stored inside the workspace:
  Feature distributions and variance evaluations
  30-feature correlation heatmaps analyzing multi-collinearity
  Training optimization loss curves for gradient descent convergence
  Logistic Regression trained decision boundary contours in geometric feature spaces
  Final model testing performance comparison charts
🚀 ResultsThe exact metrics logged during testing evaluations:
Model,Accuracy,Precision,Recall,F1 Score
Logistic Regression,0.9719,0.9841,0.9466,0.9649
K-Nearest Neighbors,0.9807,0.0000,0.0000,0.0000
Decision Tree,0.9415,0.9443,0.9234,0.9332
▶️ How to Run1.
Install dependencies:
  pip install -r requirements.txt
2. Execute pipeline script
Run the central execution pipeline script to process the training workflow and log your outputs:
  python src/run.py
