'''
Given a blob of text, predict which newsgroup it belongs to.
The dataset contains roughly 20K newsgroup docs split across
20 newsgroups.
See http://qwone.com/~jason/20Newsgroups/

'''

from re import T
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Shuffle, as samples may not be independently distributed,
# and some models won't like that
# Use subset="all" to get train and test
twenty_train = fetch_20newsgroups(subset="train", shuffle=True)
twenty_test = fetch_20newsgroups(subset="test", shuffle=True)

# "data" is for our training data
# "target_names" gives names of newsgroups
# "target" gives a number for each newsgroup, and we'll
# use this as our "label"
print(twenty_train.keys())

# Show a sample record
print(twenty_train.data[0])

# Show list of newsgroup names
print(twenty_train.target_names)
# Show numbers representing labels
print(twenty_train.target)

# We use CountVectorizer to identify all words
# in all documents by (documentId, wordId) and
# then store the frequency of that word
count_vect = CountVectorizer()
# X_train_counts will be a sparse matrix
X_train_counts = count_vect.fit_transform(twenty_train.data)
print("Count Vectorizer Shape (num documents x num distinct words): ",
      X_train_counts.shape)
print("Data Length (num documents): ", len(twenty_train.data))
print(X_train_counts[0])

# Note that we send in the previously made "Count Vectorizer"
# (count occurences of each word in each document) into the
# "term frequency - inverse document frequency" model
# From https://monkeylearn.com/blog/what-is-tf-idf/
# " It works by increasing proportionally to the number of times a word
# appears in a document, but is offset by the number of documents that contain
# the word. So, words that are common in every document, such as this, what,
# and if, rank low even though they may appear many times, since they don’t
# mean much to that document in particular."
tfidf_transformer = TfidfTransformer()
# Get a _score_ for every (documentId, wordId)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

# Now, let's train a model
# LinearSVC is similar to SVC with parameter "linear",
# but has more flexibility
# see https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
# LinearSVC also supports multiclass (which is what we need)
# "tol" is the tolerance for stopping training on our model
# when losses go below "tol", it's time to stop training
# doc says to use dual=False when n_samples > n_features
clf_svc = LinearSVC(penalty="l2", dual=False, tol=1e-3)
# This is using a "one-vs-the-rest" scheme
clf_svc.fit(X_train_tfidf, twenty_train.target)


def how_good_is_model(pipeline):
    pipeline.fit(twenty_train.data, twenty_train.target)
    # let's get the test data
    predicted = pipeline.predict(twenty_test.data)

    acc_svm = accuracy_score(twenty_test.target, predicted)
    return acc_svm


for penalty in ["l2", "l1"]:
    # We can also create a pipeline
    clf_svc_pipeline = Pipeline([
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", LinearSVC(penalty=penalty, dual=False, tol=0.001))
    ])

    accuracy = how_good_is_model(clf_svc_pipeline)
    print(f"Accuracy with {penalty} penalty: ", accuracy)


# Now, let's try without Tfidf
clf_svc_pipeline = Pipeline([
    ("vect", CountVectorizer()),
    ("clf", LinearSVC(penalty=penalty, dual=False, tol=0.001))
])

accuracy = how_good_is_model(clf_svc_pipeline)

print(f"Accuracy without Tfidf (CountVectorizer only): ", accuracy)
