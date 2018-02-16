from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
import numpy as np
import pandas as pd
import re
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


train = pd.read_csv("train.csv")
train = train [['id','StringToExtract','description']]
X= train [['id','description']]
Y= train [['id','StringToExtract']]

pattern=r'([:\'//,\s"][a-zA-Z]+[\w=-]+[.])|(\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3})'

tokenizer_obj=RegexpTokenizer(pattern)

def tokenizer(doc):
    tokens=tokenizer_obj.tokenize(doc)
#     print(tokens)
#     tokens=[re.sub(r'[:\';//\s"]',"",token) for token in tokens]
#     print(tokens)
    return tokens
# 
# def preprocess(doc):
#     doc=re.sub(r'"',"",doc)
#     return doc
#     
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1)

# print(x_train['description'])
count_vect=CountVectorizer(analyzer='word',stop_words='english',tokenizer=tokenizer)
# print(count_vect.get_params())

X_train_counts = count_vect.fit_transform(x_train['description'])

tfidf_transformer = TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)


print(X_train_counts,X_train_tfidf)
# print(count_vect.get_feature_names())


X_test_counts = count_vect.transform(x_test['description'])
X_test_tfidf=tfidf_transformer.transform(X_test_counts)

features=count_vect.get_feature_names()

# parameter_candidates = {
#     'loss': ('log', 'hinge'),
#     'penalty': ['l1', 'l2', 'elasticnet'],
#     'alpha': [0.001, 0.0001, 0.00001, 0.000001]}

# parameter_candidates = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#                         {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]

#clf=GridSearchCV(estimator=SGDClassifier(), param_grid=parameter_candidates, n_jobs=-1)
clf = SGDClassifier(alpha=1e-05,loss='log',penalty='l2')


print("loaded classifier")
# print(y_test['StringToExtract'])
clf.fit(X_train_tfidf, y_train['StringToExtract'])
# gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
# # gs_clf = gs_clf.fit(x_train['description'],y_train['StringToExtract'])
# print(clf.best_score_)
# print(clf.best_params_)
print("fitted classifier")
predicted=clf.predict(X_test_tfidf)
print("predicted classifier")

for i in range(len(predicted)):
    if predicted[i] != y_test.iloc[i,1]:
        print(predicted[i],y_test.iloc[i,0],y_test.iloc[i,1],x_test.iloc[i,1])
        
#         index=X_test_counts[i].indices
#         for j in range(len(index)):
#             print(features[index[j]])
#         print(X_test_counts[i].data,X_test_tfidf[i].data)
      

# print(count_vect.get_feature_names())
# # print(count_vect.vocabulary_)    
#  
accuracy = np.mean(predicted == y_test['StringToExtract'])
print(accuracy)


# pred = predicted
# ids = test["id"]
# 
# column_order = ["id", "StringToExtract"]
# df = pd.DataFrame({"id": ids, "StringToExtract": pred})
# df[column_order].to_csv("sample-submission.csv", index=False)