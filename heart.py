
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("heart.csv")

print("Dataset Preview:")
print(df.head())

print("\nColumns:", df.columns)



print("\nMissing values:\n", df.isnull().sum())


df = df.dropna()


#  'target' as output (0 = no disease, 1 = disease)
y = df['target']
X = df.drop('target', axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree (Before Tuning)")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))


dt_tuned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_tuned.fit(X_train, y_train)

y_pred_dt_tuned = dt_tuned.predict(X_test)

print("\nDecision Tree (After Tuning)")
print("Accuracy:", accuracy_score(y_test, y_pred_dt_tuned))


plt.figure(figsize=(15, 8))
plot_tree(dt_tuned, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))


print("\nModel Comparison:")
print("Decision Tree (Tuned):", accuracy_score(y_test, y_pred_dt_tuned))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))


importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance)

#  Feature Importance
plt.figure(figsize=(10,5))
plt.bar(importance['Feature'], importance['Importance'])
plt.xticks(rotation=90)
plt.title("Feature Importance (Random Forest)")
plt.show()

cv_scores = cross_val_score(rf, X, y, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))

# Detailed Evaluation

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))