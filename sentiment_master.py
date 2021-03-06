import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
# df.head()             # shows the first part of the file
# In our original code, we imported, then Vectorized, then split, then train the model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['review'],
                                                    df['target'],
                                                    random_state=0)
print('X_train first entry:\n\n', X_train[0])
print('\n\nX_train shape: ', X_train.shape)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer().fit(X_train)

# vect.get_feature_names()[::2000]
# len(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)
# X_train_vectorized

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)


from sklearn.metrics import roc_auc_score
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs:\n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))