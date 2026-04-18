
## Heart Disease Prediction using Decision Tree & Random Forest


This project implements tree-based machine learning models to predict the presence of heart disease using patient health data. It compares Decision Tree and Random Forest models and analyzes their performance.

__Objective__

Train and visualize a Decision Tree model
Understand and control overfitting
Build a Random Forest model and compare performance
Analyze feature importance
Evaluate models using cross-validation

__Tools & Technologies__
Python
Pandas
NumPy
Scikit-learn
Matplotlib

__Dataset Description__

Dataset: heart.csv
Contains medical attributes such as:
Age, Sex
Chest Pain Type (cp)
Blood Pressure (trestbps)
Cholesterol (chol)
Maximum Heart Rate (thalach)
etc.
Target: target
0 → No Heart Disease
1 → Heart Disease

__Steps Performed__

1. Data Preprocessing
Checked for missing values
Cleaned dataset using dropna()
Separated features and target
2. Decision Tree Model
Trained a Decision Tree classifier
Visualized the tree structure
Observed overfitting in default model
3. Overfitting Control
Limited tree depth using max_depth=4
Improved generalization performance
4. Random Forest Model
Trained Random Forest with multiple trees
Compared accuracy with Decision Tree
Achieved better stability and performance
5. Model Evaluation
Accuracy score
Confusion matrix
Classification report
6. Feature Importance
Identified most influential features using Random Forest
Visualized importance using bar chart
7. Cross Validation
Applied 5-fold cross-validation
Ensured model reliability across different splits

__Results & Insights__
Random Forest outperformed Decision Tree in most cases
Controlling tree depth reduced overfitting
Key features like chest pain type and heart rate significantly impact prediction
Cross-validation confirmed model stability
