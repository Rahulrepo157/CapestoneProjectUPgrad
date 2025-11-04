

# Credit Card Fraud Detection Project

## Project Overview

This project aims to build and evaluate machine learning models to detect fraudulent credit card transactions. Due to the nature of the data, this is a **highly imbalanced classification problem**, with a very small fraction of transactions being fraudulent.

The project demonstrates a complete machine learning workflow:

1.  **Exploratory Data Analysis (EDA)**
2.  **Data Preprocessing** (scaling, skewness correction)
3.  **Stratified Data Splitting**
4.  **Handling Class Imbalance** (RandomOversampling, SMOTE, ADASYN)
5.  **Hyperparameter Tuning** (using `GridSearchCV`)
6.  **Model Evaluation** (using appropriate metrics like Average Precision / PR AUC)

The core of this project involves using `imblearn` pipelines to correctly apply oversampling techniques *within* a cross-validation loop, preventing data leakage and leading to a reliable model.

## 📋 Dataset

The project uses the `creditcard.csv` dataset, which has the following characteristics:

  * **Features:** Contains numerical input variables that are the result of a PCA transformation (V1, V2, ..., V28) for confidentiality.
  * **Non-PCA Features:** `Time` (elapsed seconds) and `Amount` (transaction amount).
  * **Target:** The `Class` column, where `0` is a normal transaction and `1` is a fraudulent transaction.
  * **Imbalance:** The dataset is highly imbalanced; the positive class (fraud) accounts for a very small percentage of all transactions.

## 🚀 Methodology

The project follows these key steps:

1.  **Data Preparation**

      * Load the `creditcard.csv` file.
      * Perform EDA to visualize the class imbalance and feature distributions.
      * Drop the `Time` column, as it's an elapsed counter and not a generalizable feature.
      * Check the `Amount` feature for skewness and apply a `PowerTransformer` to make its distribution more Gaussian.
      * Split the data into training and testing sets using a **stratified split** to ensure both sets have the same proportion of fraudulent transactions.

2.  **Model 1: Baseline (Imbalanced Data)**

      * A `RandomForestClassifier` (with `class_weight='balanced'`) is trained on the raw, imbalanced data.
      * This provides a baseline understanding of performance before advanced sampling is applied.

3.  **Model 2: Handling Imbalance (Balanced Data)**

      * A `LogisticRegression` model is chosen for the main experiment.
      * Three different oversampling techniques are tested:
          * `RandomOverSampler` (ROS)
          * `SMOTE` (Synthetic Minority Over-sampling Technique)
          * `ADASYN` (Adaptive Synthetic Sampling)
      * To prevent data leakage, an `imblearn.pipeline.Pipeline` is created. This ensures that the oversampling is applied **only to the training folds** during cross-validation.
      * `GridSearchCV` is used to tune the `C` hyperparameter of the `LogisticRegression` model for each sampling method.
      * The scoring metric chosen is **`average_precision`** (also known as PR AUC), which is more informative than ROC AUC for highly imbalanced datasets.

4.  **Evaluation**

      * The best-performing pipeline (e.g., `LogisticRegression` + `SMOTE`) is selected based on the highest validation PR AUC score.
      * This final, optimized model is evaluated on the **hold-out test set**.
      * A full `classification_report`, `average_precision_score`, and `roc_auc_score` are printed.
      * The ROC curve is plotted, and the optimal decision threshold (based on the best Youden's J statistic) is identified.

## 🛠️ Requirements

This project uses the following Python libraries. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

  * **pandas** & **numpy**: For data manipulation.
  * **matplotlib** & **seaborn**: For data visualization.
  * **scikit-learn (sklearn)**: For preprocessing (`PowerTransformer`), modeling (`LogisticRegression`, `RandomForestClassifier`), and evaluation (`GridSearchCV`, `classification_report`).
  * **imbalanced-learn (imblearn)**: For oversampling techniques (`SMOTE`, `ADASYN`, `RandomOverSampler`) and the `ImbPipeline`.
  * **jupyter**: To run the notebook.

## Usage

1.  Clone this repository or download the files.
2.  Ensure you have the `creditcard.csv` file in the same directory as the notebook.
3.  Run the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
4.  Open the `Credit_card_fraud_detection_Starter_code+.ipynb` file and run the cells sequentially from top to bottom.

## 🏁 Results

The project successfully demonstrates that simply training on imbalanced data, even with `class_weight`, is suboptimal.

By using a proper `imblearn` pipeline with `GridSearchCV` and an appropriate oversampling technique (like SMOTE or ADASYN), we achieve a model with high recall and average precision. The final evaluation on the test set confirms the model's ability to effectively identify fraudulent transactions while maintaining reasonable precision.

The final output of the notebook includes:

  * A comparison of the validation scores for ROS, SMOTE, and ADASYN.
  * The final classification report for the best model on the unseen test data.
  * A plot of the ROC curve with the optimal decision threshold.
