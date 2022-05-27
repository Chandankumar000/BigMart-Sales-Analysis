# BigMart-Sales-Analysis

# Problem Statement

- The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.

- Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.

## The Data
We have train (8523) and test (5681) data set, train data set has both input and output variable(s). You need to predict the sales for test data set.

## Data Exploration:
- Looking at categorical and continuous feature summaries and making inferences about the data.

## Data Cleaning
- Imputing missing values in the data.

## Model Building
- Making predictive models on the data



# List of Common Machine Learning Algorithms
- Linear Regression
- Logistic Regression
- Decision Tree
- SVM
- Naive Bayes
- kNN
- K-Means
- Random Forest
- Dimensionality Reduction Algorithms
- Gradient Boosting algorithms
    - GBM
    - XGBoost
    - LightGBM
    -  CatBoost


### 1. Linear Regression
>- It is used to estimate real values (cost of houses, number of calls, total sales etc.) based on continuous variable(s). Here, we establish relationship between independent and dependent variables by fitting a best line. This best fit line is known as regression line and represented by a linear equation Y= a *X + b.

>- seperate the independent and target variable on training data
- X = train_data.drop(columns=['Item_Outlet_Sales'],axis=1)
- y = train_data['Item_Outlet_Sales']
- from sklearn.model_selection import train_test_split
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
- from sklearn.linear_model import LinearRegression
- model = LinearRegression()
-  model.fit(X,Y)
-  model.coef_
-  model.intercept_
-  model.predict(test_x)
-
### 2. Logistic Regression
>- Don’t get confused by its name! It is a classification not a regression algorithm. It is used to estimate discrete values ( Binary values like 0/1, yes/no, true/false ) based on given set of independent variable(s).

 **seperate the independent and target variable on training data**
- X = train_data.drop(columns=['Item_Outlet_Sales'],axis=1)
- y = train_data['Item_Outlet_Sales']
- from sklearn.model_selection import train_test_split
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
- from sklearn.linear_model import LogisticRegression
- model = LogisticRegression()
-  model.fit(X,Y)
-  model.coef_
-  model.intercept_
-  model.predict(test_x)
