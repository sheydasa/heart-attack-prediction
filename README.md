# Heart Attack Prediction

## Overview
The Heart Attack Prediction project focuses on developing a predictive model to assess an individual’s likelihood of experiencing a heart attack. Using a dataset containing 303 samples and 14 variables-including sex, age, chest pain, and more-the objective is to build and evaluate multiple machine learning algorithm to predict heart attack risks in patients. 
The models tested include Logistic Regression, Support Vector Machines, Random Forest, and XGBoost, with the goal of determining the most accurate approach.

## Dataset
The dataset used for this project can be downloaded through [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset). The downloaded dataset needs to be saved as 'heart.csv' in the same directory as the script.

## Requirements
- Python 2.7 or higher
- The 'heart.csv' dataset from [Kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
- The dependencies listed in 'requirements.txt'

## Installation Instructions
1. Clone this repository:
```bash
git clone https://github.com/sheydasa/heart-attack-prediction.git
cd heart-attack-prediction
```
2. Set up a virtual environment
```bash
python -m venv myenv
source myenv/bin/activate # On Windows use 'myenv\Scripts\activate'
```
3. Install the required packages
```bash
pip install -r requirements.txt
```

## USAGE
1. Make sure the heart.csv dataset is in the same directory as DataPrediction.py
2. Run the script
```bash
python DataPrediction.py
```
3. The script will perform the following:
- Load and clean the dataset
- Visualize data distributions and relationships
- Train multiple machine learning models including Logistic Regression, Support Vector Machine, Decision Trees, Random Forests, Gradient Boosting, K-Nearest Neighbors, XGBoost, and Bernoulli Naive Bayes.
- Evaluate each model's performance using cross-validation and display accuracy, confusion matrix, classification report, and ROC curve

DATASET LINK

DEPENDANCIES (REFERENCE REQUIREMENTS.TXT)

