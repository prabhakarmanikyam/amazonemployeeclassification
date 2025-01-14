
# Employee Access Classification: Machine Learning Project

### Author: Prabhaker Manikyam
### File: prabhaker_manikyam.ipynb

### Project Overview

This project aims to classify employee access requests as approved or denied based on historical data using machine learning algorithms. The dataset includes features describing employees, their roles, and the resources requested. By leveraging supervised learning techniques, this model automates the decision-making process, reducing manual intervention and optimizing resource allocation.

### Dataset Description

### Training Data:
- File: amazon_employee_access_train.csv
- Size: 32,769 samples
- Target Variable: ACTION (1 = Approved, 0 = Denied)
### Features:

- RESOURCE: ID for each resource
- MGR_ID: Manager ID
- ROLE_ROLLUP_1: Role grouping category 1
- ROLE_ROLLUP_2: Role grouping category 2
- ROLE_DEPTNAME: Department description
- ROLE_TITLE: Business title
- ROLE_FAMILY_DESC: Family extended description
- ROLE_FAMILY: Family description
- ROLE_CODE: Role-specific code

### Testing Data:
- File: amazon_employee_access_test.csv (evaluation dataset)
- Size: Unknown 
- Change your file name in last block of code in jupyter notebook

## Implemented Models

The following machine learning models were implemented and evaluated for their performance:

&nbsp;1. Logistic Regression \
&nbsp;2. Random Forest (Best Model) \
&nbsp;3. XGBoost Classifier \
&nbsp;4. MLP Neural Network

## Best Model: Random Forest
- ### Hyperparameter Tuning:
- Tool Used: RandomizedSearchCV
- Parameters Tuned:
  -   &nbsp;criterion: gini, entropy
  -   &nbsp;max_features: 0.1 to 0.9
  -   &nbsp;n_estimators: 100 to 500
  -   &nbsp;max_depth: 5 to 30
  -   &nbsp;bootstrap: True, False
  -   &nbsp;class_weight: {0: 2, 1: 1}


### Results
  -   &nbsp;Random Forest Accuracy: 95.1
  -   &nbsp;Evaluation Metric: Accuracy

### How to Run the Notebook
&nbsp; 1. Ensure the following files are in the working directory:
- &nbsp;amazon_employee_access_train.csv
- &nbsp;amazon_employee_access_test.csv
&nbsp; 2. Install required libraries:

``` 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression 
```

&nbsp; 3. Open and run the Jupyter Notebook:
```
jupyter notebook prabhaker_manikyam.ipynb
```
&nbsp; 4. Follow the steps in the notebook to preprocess the data, train the models, and evaluate performance.

### Key Files
- Jupyter Notebook: prabhaker_manikyam.ipynb
- Training Data: amazon_employee_access_train.csv
- Test Data: amazon_employee_access_test.csv

### After evaluating the test dataset: True positive rate, False positive rate and accuracy are displayed at the end of the notebook.