import time
t0 = time.time()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer   # “Term Frequency times Inverse Document Frequency”
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB       # Multinomial Naive Bayes, supposedly goes well with the data from the transformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
import string

print("Reading the data into a working variable...")        # READ DATA IN
df = pd.read_csv('MasterDataFile.csv')      # to read the training data into working memory
df['feature'] = df['feature'].replace(to_replace='<.*>', value='', regex=True)
# df['feature'] = df['feature'].astype(np.str).replace(to_replace= (('\.'),('\?')), value='', regex=True)
# df['feature'] = df['feature'].astype(np.str).replace(to_replace= ('\,'), value=' ', regex=True)
# df['feature'] = df['feature'].replace(to_replace=string.punctuation, value='')



# print("Removing punctuation...")      # REMOVE PUNCTUATION
# exclude = set(string.punctuation)
# def remove_punctuation(x):
#     """
#     Helper function to remove punctuation from a string
#     x: any string
#     """
#     try:
#         x = ''.join(ch for ch in x if ch not in exclude)
#     except:
#         pass
#     return x
# # Apply the function to the DataFrame
# df.review = df.review.apply(remove_punctuation)


print("Splitting Data into subsets...")         # DATA SPLITING
X_train, X_test, y_train, y_test = train_test_split(df['feature'],   # Test-Train Split function
                                                    df['target'],
                                                    train_size = 0.8,
                                                    test_size = 0.2,
                                                    random_state=14
                                                    )
print("Training shape is: ", X_train.shape)

print("Extracting features...")         # FEATURE EXTRACTION
vect = CountVectorizer(min_df=1,          # Minimum Document Frequency
                       ngram_range=(1, 2),   # unigrams to trigrams
                       stop_words='english',   # stopwords
                       lowercase =False,
                       binary = True
                       ).fit(X_train)
X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)
print(len(vect.get_feature_names()), "features found.")     # Feature check


print("Transforming for tfidf...")          # TFIDF Transformer
X_train_vectorized_tfidf = TfidfTransformer().fit_transform(X_train_vectorized) # or X_train_dense
X_test_vectorized_tfidf = TfidfTransformer().fit_transform(X_test_vectorized) # or X_test_dense

# print("Converting to Numpy Dense Array...")
# X_train_dense = X_train_vectorized.toarray()   # Convert to Dense Numpy Array
# X_test_dense = X_test_vectorized.toarray()

# Classifier Comparison

# # Gaussian Naive Bayes model
# print("Training/Evaluating Gaussian Naive Bayes model...")
# gnb_model = GaussianNB().fit((X_train_dense), y_train)
# print("GaussianNB accuracy: ", gnb_model.score((X_test_dense), y_test))


# Bernoulli Naive Bayes model
print("Training/Evaluating Bernoulli Naive Bayes model...")
bnb_model = BernoulliNB().fit(X_train_vectorized, y_train)
print("BernoulliNB accuracy: ", bnb_model.score(X_test_vectorized, y_test))


# Mulitnomial Naive Bayes model
print("Training/Evaluating Multinomial Naive Bayes model...")
mnb_model = MultinomialNB().fit(X_train_vectorized, y_train)
print("MultinomialNB accuracy: ", mnb_model.score(X_test_vectorized, y_test))


# Linear SVC Model
print("Training/Evaluating Linear SVC model...")
lsvc_model = LinearSVC().fit(X_train_vectorized_tfidf, y_train)
print("Linear SVC accuracy: ", lsvc_model.score(X_test_vectorized_tfidf, y_test))


# Logistic Regression Model
print("Training/Evaluating Logistic Regression model...")
lr_model = LogisticRegression().fit(X_train_vectorized, y_train)
print("Logistic Regression accuracy: ", lr_model.score(X_test_vectorized, y_test))


# # svm.SVC model
# clfrSVM = svm.SVC(kernel='linear', C=0.1)             # double code for tfidf and without
# clfrSVMtfidf = svm.SVC(kernel='linear', C=0.1)
# clfrSVM.fit(X_train_vectorized, y_train)
# clfrSVMtfidf.fit(X_train_vectorized_tfidf, y_train)
# predicted_labels = clfrSVM.predict(X_test_vectorized)
# predicted_labels_tfidf = clfrSVMtfidf.predict(X_test_vectorized_tfidf)
# print("SVM:", accuracy_score(y_test, predicted_labels))
# print("SVM, with tfidf:", accuracy_score(y_test, predicted_labels_tfidf))

###         FINAL STEPS


weighted_prediction = lsvc_model.predict(X_test_vectorized)

print ('Accuracy:', accuracy_score(y_test, weighted_prediction))
print ('F1 score:', f1_score(y_test, weighted_prediction,average='weighted'))
print ('Recall:', recall_score(y_test, weighted_prediction,
                              average='weighted'))
print ('Precision:', precision_score(y_test, weighted_prediction,
                                    average='weighted'))
print ('\n classification report:\n', classification_report(y_test, weighted_prediction))
print ('\n confussion matrix:\n',confusion_matrix(y_test, weighted_prediction))
#           # This is the actual testing data being read into a new variable
# test_doc = pd.read_csv('test_data.csv')     # reading in the testing data file for final testing
#
# X_train2, X_test2, y_train2, y_test2 = train_test_split(test_doc['review'], # creating a similar variable with test data
#                                                         test_doc['target'],
#                                                         test_size=.999999,
#                                                         random_state=0
#                                                         )
#
# X_test2_vect = vect.transform(X_test2)          # Test Data feature extraction
# # print('\n\nX_train shape: ', X_test2_vect.shape)      # Are we on the right track?
# X_test2_final = TfidfTransformer().fit_transform(X_test2_vect)      # Tfidf transformation of Test Data
# print(cross_val_score(lsvc_model, X_test2_final, y_test2, cv=10))   # 10-fold Cross Validation of Test Data
# #           4 CV parameters = Classifier, X_test, y_test, K=


#           # Two more evaluation metrics
# y_pred = lsvc_model.predict(X_test_vectorized_tfidf)
# print("Second Metric: ", accuracy_score(y_test, y_pred))
# print("Third Metric: ", metrics.f1_score(y_test, y_pred, average='micro'))
t1 = time.time()

print("time to run: ", t1-t0)