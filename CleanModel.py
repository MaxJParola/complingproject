# import all of the things we need to run the project
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix

# read in the data file
df = pd.read_csv('MasterDataFile.csv')
df['feature'] = df['feature'].replace(to_replace='<.*>', value='', regex=True)

# Split the data into a test and a training set
X_train, X_test, y_train, y_test = train_test_split(df['feature'],
                                                    df['target'],
                                                    train_size = 0.8,
                                                    test_size = 0.2,
                                                    random_state=14)

# , and fit the text so that it can be read into the model
vect = CountVectorizer(min_df=1,
                       ngram_range=(1, 2),
                       stop_words='english',
                       lowercase =False,
                       binary = True
                       ).fit(X_train)

# Transform the data so it can be read into the model
X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)

# Set up a parameter grid so that the GridSearch knows which parameters to run through part of tuning hyper-parameters
LSVCParam_Grid = ({'penalty': ['l2'], 'loss': ['squared_hinge'], 'dual': [True, False], 'C': [1, 100, 1000, 10000],
                  'multi_class': ['ovr', 'crammer_singer'], 'max_iter': [1000, 2000, 8000]},
                  {'penalty': ['l2'], 'loss': ['hinge', 'squared_hinge'], 'dual': [True], 'C': [1, 100, 1000, 10000],
                  'multi_class': ['ovr', 'crammer_singer'], 'max_iter': [1000, 2000, 8000]})

# Additional set-up for tuning hyper parameters. We now have the type of model selected.
lsvc = LinearSVC()
clf = GridSearchCV(lsvc, LSVCParam_Grid)

# fit the model with all of the parameters selected in the LSVCParam_grid
print("Tuning hyper parameters with exhaustive grid search, this may take a while...")
clf.fit(X_train_vectorized, y_train)

print('The best score from training data was: ')
print(clf.best_score_)

print("Best estimator parameters from Exhaustive Grid Search: ")
print(clf.best_estimator_)

clf.score(X_test_vectorized, y_test)
print("Accuracy of best estimator on test data: ")
print(clf.score(X_test_vectorized, y_test))

print("Best parameters for not under/overfitting from grid search: ")
print(clf.best_params_)

# Linear SVC with the best params from the grid search
ourlsvc = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=None, tol=0.0001, verbose=0).fit(X_train_vectorized, y_train)

#use the model to predict values based on text from test dataset
y_predict = ourlsvc.predict(X_test_vectorized)
y_test_compare = y_test.as_matrix()

print('LinearSVC with best parameters from GridSearchCV')
print('Accuracy: ', accuracy_score(y_test_compare, y_predict))
print('f1: ', f1_score(y_test_compare, y_predict))
print('Precision: ', precision_score(y_test_compare, y_predict))
print('Recall: ', recall_score(y_test_compare, y_predict))
print(classification_report(y_test_compare, y_predict))