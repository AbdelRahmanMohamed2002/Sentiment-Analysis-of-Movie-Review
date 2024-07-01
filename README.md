# Sentiment Analysis of Movie Reviews

## Overview
This repository contains two Jupyter notebooks that perform sentiment analysis on movie reviews using various machine learning models. The notebooks guide you through loading data, preprocessing text, vectorizing text data, training different classifiers, and evaluating their performance.

## Notebooks

### 1. Main._First_DataSet.ipynb
**Description**: Uses the IMDB movie reviews dataset to perform sentiment analysis.
### 2. Main._First_DataSet_With10KFold.ipynb
**Description**: Jupyter notebook for sentiment analysis using the IMDB movie reviews dataset with 10-fold cross-validation.
### 3. Main._Second_DataSet_With10KFold.ipynb
**Description**: Jupyter notebook for sentiment analysis using a different dataset with 10-fold cross-validation.
### 4. Main._Second_DataSet.ipynb
**Description**: Uses a different dataset for training and evaluating the models.
### 5. Pre_Processing_Function.py
**Description**: Python script containing text preprocessing functions.

**Steps**:
- Import libraries and custom preprocessing functions.
- Load and preprocess the IMDB dataset.
- Vectorize text data using TF-IDF.
- Split the dataset into training and testing sets.
- Train and evaluate four models: Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machine.
- Plot performance metrics (accuracy, precision, recall, F1 score) for each model.

### 2. Main._Second_DataSet.ipynb
**Description**: Uses a different dataset for training and evaluating the models.

**Steps**:
- Import libraries and custom preprocessing functions.
- Load and preprocess the training dataset.
- Vectorize text data using TF-IDF.
- Split the dataset into training and testing sets.
- Train and evaluate four models: Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machine.
- Plot performance metrics (accuracy, precision, recall, F1 score) for each model.

## Requirements
- Python 3.x
- Jupyter Notebook
- Libraries:
  - os
  - pandas
  - matplotlib
  - scikit-learn




## Results
all notebooks output the performance metrics for each model in terms of accuracy, precision, recall, and F1 score. The results are visualized in bar plots for easy comparison.

## Author
Abdelrahman  
abdelrahman2011907@gmail.com

## License

