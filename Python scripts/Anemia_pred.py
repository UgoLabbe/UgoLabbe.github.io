import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

# Load the dataset
file_path = 'Data/Anemia.csv'  # From https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset/data
df = pd.read_csv(file_path) 

# Preview the dataset
df.head(10) #0 = Male, 1 = Female
len(df)

# Separate the features (X) from the outcomes (y)
df = np.array(df)
X = df[:, :5]  # Features
y = df[:, 5]   # Outcome

# Machine Learning predictions
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scoring to assess the best hyper-parameters
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score)
}

# K-Nearest Neighbors
param_grid_knn = {
    'n_neighbors': range(1,25),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()

grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, scoring=scoring, refit='accuracy', cv=5, verbose=2)
grid_search_knn.fit(X_train, y_train)

print("Best KNN parameters:", grid_search_knn.best_params_)
print("Best KNN accuracy:", grid_search_knn.best_score_)

best_knn = grid_search_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)

print("KNN Test Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Test F1 Score:", f1_score(y_test, y_pred_knn))

# Support Vector Machine
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}

svm = SVC()

grid_search_svm = GridSearchCV(estimator=svm, param_grid=param_grid_svm, scoring=scoring, refit='accuracy', cv=5, verbose=2)
grid_search_svm.fit(X_train, y_train)

print("Best SVM parameters:", grid_search_svm.best_params_)
print("Best SVM accuracy:", grid_search_svm.best_score_)

best_svm = grid_search_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test)

print("SVM Test Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Test F1 Score:", f1_score(y_test, y_pred_svm))