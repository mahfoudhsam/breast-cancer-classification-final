# Breast Cancer Diagnosis Classification From Scratch

A machine learning project that implements classical ML algorithms **from scratch**, without using scikit-learn, to classify breast cancer tumors as **benign** or **malignant**.

---

## 📌 Project Overview

This project aims to classify breast cancer tumors using handcrafted machine learning algorithms and a fully modular machine learning pipeline.

### Implemented Algorithms

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree

### Project Features

* Data preprocessing and feature correlation analysis
* Evaluation metrics implemented entirely from scratch
* Structured experiment logging and performance tracking
* Decision boundary and model comparison visualizations

---

## 🤖 Implemented Models

### Logistic Regression

* Implemented using Gradient Descent optimization
* Uses the Sigmoid activation function for probability estimation
* Optimized through a custom Binary Cross-Entropy loss function

### K-Nearest Neighbors (KNN)

* Implemented using vectorized Euclidean distance calculations
* Uses dynamic majority voting among nearest neighbors

### Decision Tree

* Implemented using Shannon Entropy
* Splits nodes based on Information Gain maximization
* Supports recursive tree construction with customizable depth

---

## 📊 Evaluation Metrics

All evaluation metrics were implemented from scratch without relying on external machine learning libraries.

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📈 Visualizations

Generated figures stored in the workspace include:

* Feature distribution analysis
* Feature variance evaluation
* Correlation heatmaps (30 features)
* Gradient Descent loss convergence curves
* Logistic Regression decision boundaries
* Model performance comparison charts

---

## 🏆 Results

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   0.9719 |    0.9841 | 0.9466 |   0.9649 |
| K-Nearest Neighbors |   0.9807 |    0.0000 | 0.0000 |   0.0000 |
| Decision Tree       |   0.9415 |    0.9443 | 0.9234 |   0.9332 |

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Execute the Pipeline

Run the central execution script:

```bash
python src/run.py
```

This will:

* Load and preprocess the dataset
* Train all implemented models
* Evaluate performance metrics
* Generate visualizations
* Store experiment logs and results

---

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
│
├── README.md
├── report.ipynb
└── requirements.txt
```

---

## 🔧 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

---

## 🎯 Learning Objectives

This project was developed to better understand:

* Machine Learning fundamentals
* Optimization using Gradient Descent
* Distance-based learning algorithms
* Entropy and Information Gain
* Model evaluation metrics
* Building ML pipelines from scratch

---

## 📜 License

This project is intended for educational and research purposes.
