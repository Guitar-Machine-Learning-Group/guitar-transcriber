import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics

X_train = np.load("3doorsdown_herewithoutyou_features.npy")
y_train = np.load("3doorsdown_herewithoutyou_labels.npy")

X_test = np.load("radiohead_creep_features.npy")
y_test = np.load("radiohead_creep_labels.npy")

clf = OneVsRestClassifier(SVC(kernel='poly'))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)  # predict on a new X
score = metrics.accuracy_score(y_test, y_pred)
print (score)
