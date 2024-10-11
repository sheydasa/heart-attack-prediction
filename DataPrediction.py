import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier

# loading data and initial look at the dataset
df = pd.read_csv("heart.csv")
print('The first five rows are: \n' , df.head())

# separate categorical and continuous data
categorical_columns = ['sex', 'exng', 'caa', 'cp', 'restecg', 'fbs', 'slp', 'thall']
continuous_columns = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
categorical_data = df[categorical_columns]
continuous_data = df[continuous_columns]

# to find missing values, duplications, and initial data findings
def data_cleaning(data):    
    print("number of missing values in each column:")
    missingValue = data.isnull().sum()
    print(missingValue, "\n")
    
    print("number of unique values in each column: ")
    print(data.nunique(), "\n")
    
    # check duplicate rows
    duplicateRow = data[data.duplicated()]
    print(duplicateRow)
    # remove duplicate row
    data.drop_duplicates(keep='first',inplace=True)
    # shape of new df
    print("the new shape is: ", data.shape[0], data.shape[1], "\n")
    
    # summary statistics
    print("Summary statistics:\n" , data.describe(), "\n")

data_cleaning(df)

# visualization tools
def boxplot (data):    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()     
    for idx, feature in enumerate(data):
        ax = axes[idx]
        sns.boxplot(x = feature, data = data, showmeans=True, meanprops={"markerfacecolor":"red"}, ax = ax)
        plt.xlabel(feature)
    plt.tight_layout()
    plt.show()
  
def countplot(data):
    fig, axes = plt.subplots(3, 3, figsize=(12,12))
    axes = axes.flatten()
    for idx, feature in enumerate(data):
        ax = axes[idx]
        sns.countplot(x = feature, data = data, ax = ax)
        plt.xlabel(feature)
        plt.ylabel('Count')
    plt.show()

def heatmap(data):
    corr_mat = data.corr()
    print('Correlation: \n' , corr_mat)
    plt.figure(figsize=(12,12))
    mask = np.triu(np.ones_like(corr_mat))
    sns.heatmap(corr_mat,mask=mask,fmt=".1f",annot=True)
    plt.show()

def histplot(data):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()
    for idx, feature in enumerate(data):
        ax = axes[idx]    
        sns.histplot(data = df, x = feature, hue = "output", kde = True, ax = ax)
    plt.tight_layout()
    plt.show()

def kdeplot(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    for idx, feature in enumerate(data):
        ax = axes[idx]    
        sns.kdeplot(data = df, x = feature, hue = "output", ax = ax)
    plt.tight_layout()
    plt.show()

# features that seem inherently relevant
rel_columns = ['cp', 'caa', 'exng']
rel_features = df[rel_columns]

# DATA VISUALIZATION
# univariate analysis
boxplot(continuous_data)
countplot(categorical_data)

# bivariate analysis
heatmap(df)
histplot(continuous_data)
kdeplot(rel_features)

#----------------------------------------------------------------------------------------
# separates  features from 'target'
x = df.drop(['output'],axis=1)
y = df[['output']]

# split the dataset into train and test set
train_data, test_data, train_target, test_target = train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=42)
print("The training set data shape is:\n", train_data.shape)
print("The training set target shape is:\n", train_target.shape)
print("The testing set data shape is:\n", test_data.shape)
print("The testing set target shape is:\n", test_target.shape)

# scaling the continuous data
scaler = RobustScaler()
train_data[continuous_columns] = scaler.fit_transform(train_data[continuous_columns])
test_data[continuous_columns] = scaler.transform(test_data[continuous_columns])

# list of algorithms being used
models = [
    (LogisticRegression(), "Logistic Regression"),
    (SVC(kernel='linear', C=1, random_state=42), "Support Vector Machines"),
    (DecisionTreeClassifier(random_state=42), "Decision Tree"),
    (RandomForestClassifier(), "Random Forest"),
    (GradientBoostingClassifier(n_estimators=300, max_depth=1, subsample=0.8, max_features=0.2, random_state=42), "Gradient Boosting"),
    (KNeighborsClassifier(), "K-Nearest Neighbors"),
    (XGBClassifier(), "XGBoost"),
    (BernoulliNB(), "Bernoulli Naive Bayes")
]

# K-Fold cross-validation
def cross_validation(model, model_name):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = cross_val_score(model, train_data, train_target.values.ravel(), cv=k_fold, scoring='accuracy')
    
    print(f"{model_name} Cross-Validation Accuracies: {accuracies}")
    print(f"{model_name} Mean Accuracy: {accuracies.mean() * 100:.2f}%")
    print(f"{model_name} Standard Deviation: {accuracies.std() * 100:.2f}%")

# evaluate model with the ROC curve except the SVC model
def roc (model, model_name):
    if model_name != "Support Vector Machines":    
        fpr, tpr, _ = roc_curve(test_target, model.predict_proba(test_data)[:,1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"{model_name}")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}, (AUC = {roc_auc:.2f})')
        plt.legend(loc="lower right")
        plt.show()

# for each model, perform cross validation, train the model, make predictions for the test set, 
# show confusion matrix and classification report, and show the ROC curve plot
for model, model_name in models:
    print("----------------------------------------------")
    cross_validation(model, model_name)
    
    # train the model and make predictions on the test set
    model.fit(train_data, train_target)
    predictions = model.predict(test_data)
    
    # Print additional metrics for the final evaluation
    conf = confusion_matrix(test_target, predictions)
    print(f"{model_name} Confusion Matrix : \n{conf}")
    print(f"The accuracy of {model_name} is : {accuracy_score(test_target, predictions) * 100:.2f}%")

    class_report = classification_report(test_target, predictions)
    print(f"{model_name} Classification Report:\n{class_report}")
    
    roc(model, model_name)