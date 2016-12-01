import numpy as np
import timeit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics

start = timeit.default_timer()
X_train = np.load("./preprocess/features/3doorsdown_herewithoutyou.npy")
y_train = np.load("./preprocess/labels/3doorsdown_herewithoutyou.npy")

X_test = np.load("./preprocess/features/beatles_blackbird.npy")
y_test = np.load("./preprocess/labels/beatles_blackbird.npy")

clf = OneVsRestClassifier(SVC(kernel='poly'))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)  # predict on a new X
score = metrics.accuracy_score(y_test, y_pred)
print ("Accuracy:", score)

stop = timeit.default_timer()

print ("Time used: %.2f s" % (stop - start))