import random
from sklearn import metrics

data = []       # empty list to store the reviews themselves
data_labels = []    # empty list to store "neg/pos"
temp_data = []      # for shuffling the neg

with open("./IMDB1noIDs.txt") as f:     # assign the positive review file to f
    for i in f:                         # loop through the lines of f
        data.append(i)                  # append the entire string to data
        data_labels.append('pos')       # assign a corresponding label as an appended list
random.shuffle(data)        # Shuffle the positive data

with open("./IMDB0noIDs.txt") as f:     # same as above
    for i in f:
        data.append(i)
        data_labels.append('neg')

for i in f:
    temp_data.append(i)
    data_labels.append('neg')

random.shuffle(temp_data)       # shuffle the negative list

for s in temp_data:       # this realigns the data and data_labels lists
    data.append(s)          # append the neg list to data, only shuffled now

vectorizer = CountVectorizer(       #   sklearn.feature_extraction.text.CountVectorizer # we want to try a different vectorizer
# vectorizer = HashingVectorizer(
    analyzer = 'word',
    # lowercase = False,
    stop_words = 'english',         # skip stop words, to help unmuddy the water
    # min_df=5,
    # ngram_range=(1,2)             # We want to try Ngrams again with more power!!!!
)
features = vectorizer.fit_transform(    # scipy.sparse.csr.csr_matrix
    data
)

features_nd = features.toarray()        # numpy.ndarray

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd,
        data_labels,
        train_size=0.8,            # uses 80% of data as training data
        test_size=0.2)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

log_model = LogisticRegression()
log_model = log_model.fit(X=X_train, y=y_train)
y_pred = log_model.predict(X_test)
print("Logistic Regression: ", accuracy_score(y_test, y_pred))

from sklearn.svm import SVC
clfrSVM = svm.SVC(kernel='linear', C=0.1)
clfrSVM.fit(X_train, y_train)
predicted_labels = clfrSVM.predict(X_test)
print("SVM:", accuracy_score(y_test, predicted_labels))

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Tfidf Transformer
from sklearn.feature_extraction.text import TfidfTransformer        # “Term Frequency times Inverse Document Frequency”
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)

# Multinomial Naive Bayes with Term frequency times inverse document frequency scaling
from sklearn.naive_bayes import MultinomialNB       # Multinomial Naive Bayes, supposedly goes well with the data from the transformer
clf = MultinomialNB().fit(X_train_tfidf, y_train)
MNBy_pred = clf.predict(X_test)
print("Multinomial Naive-Bayes, with tf-idf downscaling: ", accuracy_score(y_test, MNBy_pred))

# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB       # Multinomial Naive Bayes, supposedly goes well with the data from the transformer
clf = MultinomialNB().fit(X_train, y_train)
MNB2y_pred = clf.predict(X_test)
print("Multinomial Naive-Bayes: ", accuracy_score(y_test, MNB2y_pred))

# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB().fit(X_train, y_train)
BNBy_pred = clf.predict(X_test)
print("Bernoulli Naive-Bayes: ", accuracy_score(y_test, BNBy_pred))


# Neural Network - Does not work
from sklearn.preprocessing import StandardScaler            # Neural Networks
scaler = StandardScaler()                                   # Broke shit
X_train = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(2,2,2),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
