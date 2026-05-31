<<<<<<< HEAD
# Breast Cancer Diagnosis Classification

A rigorous, framework-free comparative machine learning pipeline built to classify clinical cell observations from the Breast Cancer Wisconsin dataset into malignant or benign configurations. This project features classification models built completely from scratch alongside framework baseline comparisons to evaluate geometric feature-space partitions and optimization boundaries.

---

## 1. Project Objective
The primary clinical and engineering objectives of this implementation are:
* **Automated Malignancy Prediction:** Engineering a vital clinical decision-support system to isolate multi-dimensional variations in cell nucleus architecture (size, roughness, and symmetry), mitigating diagnostic delays.
* **Algorithmic Evaluation:** Explicitly analyzing how different machine learning paradigms—parametric linear modeling, non-parametric distance heuristics, and hierarchical tree structures—affect predictive robustness.
* **Optimization Transparency:** Moving away from hidden framework abstractions by writing optimization architectures from scratch to inspect cost convergence, matrix scaling dependencies, and spatial decision boundary maps.

---

## 2. Implemented Models

The pipeline evaluates three distinct architectural paradigms to map classification decision zones:

### A. Logistic Regression (Built From Scratch)
* **Paradigm:** Parametric Linear Model.
* **Mathematics:** Maps linear feature combinations to a probability space via the Sigmoid activation function: $\sigma(z) = \frac{1}{1 + e^{-z}}$.
* **Optimization:** Core model weights are iteratively tuned using an explicit gradient descent routine optimizing a Binary Cross-Entropy (Log-Loss) objective function over 1,000 iterations at a learning step size of $\alpha = 0.01$.

### B. K-Nearest Neighbors (Built From Scratch)
* **Paradigm:** Non-Parametric Instance-Based Learner.
* **Mathematics:** Processes real-time geometric query vectors by calculating standard Euclidean distance matrices relative to cached coordinates: $d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$.
* **Mechanism:** Identifies the $k=3$ nearest spatial neighbors and applies a majority class voting rule.

### C. Decision Tree Classifier (Framework Baseline)
* **Paradigm:** Non-Linear Hierarchical Splitting Model.
* **Mathematics:** Maximizes localized population purity at node junctions by evaluating Gini Impurity drops.
* **Mechanism:** Implemented via `sklearn` with growth constraints strictly bound to a maximum depth of 5 (`max_depth=5`) to isolate structural overfitting vulnerabilities.

---

## 3. How to Run the Project

### Prerequisites
Ensure your localized environment contains a standard Python 3.x distribution alongside the required computational and visualization dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn

# Step 1: Run Exploratory Data Analysis
# Inspects data distribution profiling, null checks, and coordinate correlations
jupyter notebook notebooks/exploration.ipynb

# Step 2: Execute Production Pipeline
# Triggers automated preprocessing, model optimization, metric calculations,
# and exports generated charts directly into your directory workspace
python src/run.py

# Step 3: Review Report Outputs
# Open the standalone notebook to view inline markdown analyses and charts
jupyter notebook notebooks/report.ipynb

========================================================================
                          TRAINING PERFORMANCE MATRIX                  
========================================================================
Metric          | Logistic Regression | Decision Tree | K-Nearest Neighbors
------------------------------------------------------------------------
Accuracy        |       98.24%        |    99.47%     |       98.07%
Precision       |       99.03%        |   100.00%     |        0.00%
Recall          |       96.23%        |    98.58%     |        0.00%
F1-Score        |       97.61%        |    99.29%     |        0.00%
========================================================================
=======
#  Breast Cancer Classification From Scratch

A machine learning project implementing classical ML algorithms completely from scratch using NumPy, without relying on scikit-learn.

# Project Overview

This project aims to classify breast cancer tumors as malignant or benign using handcrafted machine learning algorithms and a modular execution pipeline. 

# Implemented Algorithms:
 **Logistic Regression:** Optimized via Gradient Descent using binary cross-entropy loss.
 **Decision Tree:** Built recursively using Entropy and Information Gain metrics, featuring a custom manual label-counting node logic.
 **K-Nearest Neighbors (KNN):** Implemented using manual Euclidean distance calculations and a custom dictionary-based majority voting mechanism.


## 📂 Project Structure
```text
breast_cancer/
│
├── data/
│   └── data.csv                 
│
├── figures/                    
│
├── src/
│   ├── run.py                  
│   ├── logistic_regression.py   
│   ├── Decision_tree.py         
│   ├── knn.py                  
│   └── utils.py                 

Logistic Regression
Optimization: Iterative Gradient Descent (10,000 iterations).

Activation: Sigmoid function mapping outputs between 0 and 1.

Loss Tracking: Monitored step-by-step via Binary Cross-Entropy Cost.

 Decision Tree
Splitting Criterion: Evaluates feature cutoffs based on highest Information Gain.

Leaf Resolution: Custom dictionary-based frequency mapping replacing high-level collection libraries.

 K-Nearest Neighbors (KNN)
Distance Metric: Vectorized Euclidean distance calculation.
Classification: Manual majority vote tracking of the K closest neighbors.


Evaluation & PerformanceAll models are evaluated using a custom accuracy metric function calculating exact matching ratios over the dataset matrix:$$\text{Accuracy} = \frac{\sum (y_{\text{true}} == y_{\text{pred}})}{N}$$
>>>>>>> d9f1328be27087626934f96245e93ca239fa113a
