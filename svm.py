import glob
import numpy as np
import timeit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
import pickle
from sklearn.externals import joblib


# for filename in glob.glob('./preprocess/features/*', recursive=True):
#     print(filename)
# X_train = np.load("./preprocess/features/3doorsdown_herewithoutyou.npy")
# y_train = np.load("./preprocess/labels/3doorsdown_herewithoutyou.npy")
#
# X_test = np.load("./preprocess/features/beatles_blackbird.npy")
# y_test = np.load("./preprocess/labels/beatles_blackbird.npy")
#
# clf = OneVsRestClassifier(SVC(kernel='poly'))
#
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)  # predict on a new X
# score = metrics.accuracy_score(y_test, y_pred)
# print ("Accuracy:", score)
#
# stop = timeit.default_timer()
#
# print ("Time used: %.2f s" % (stop - start))

clf = OneVsRestClassifier(SVC(kernel='poly'))
try:
    s = joblib.load('svm.pkl')
except:
    s = clf

parameters = {
    "estimator__C": [1,2,4,8],
    "estimator__kernel": ["poly","rbf"],
    "estimator__degree":[1, 2, 3, 4],
}

def train(x, y):
    global s
    c = s
    c.fit(x, y)
    s = c

def test(x, y):
    global s
    c = s
    # y_pred = c.predict(x)
    score = c.score(x, y)
    return score

def save_model(clf):
    joblib.dump(clf, 'svm.pkl')

def load_model():
    return joblib.load('svm.pkl')



if __name__ == "__main__":
    start = timeit.default_timer()
    # pre-load data
    featureFileList = glob.glob('./preprocess/features/*', recursive=True)
    labelFileList = glob.glob('./preprocess/labels/*', recursive=True)
    print (labelFileList)
    if len(featureFileList) != len(labelFileList):
        print("Data is not matched")
        exit(-1)
    total = len(labelFileList)
    trainSize = int(total * 0.8)
    # trainSize = 1
    testX = np.array([])
    testY = np.array([])

    for count in range(total):
        selector = np.random.randint(0, len(featureFileList))
        featureFileName = featureFileList.pop(selector)
        # print(count)
        # print(featureFileName)
        for i in range(len(labelFileList)):
            # print (i)
            # print(labelFileList[i].split('/')[-1].split('.')[0])
            if labelFileList[i].split('/')[-1].split('.')[0] == featureFileName.split('/')[-1].split('.')[0]:
                labelFileName = labelFileList.pop(i)
                break
        x = np.load(featureFileName)
        y = np.load(labelFileName)
        # print(y)
        if count < trainSize:
            train(x, y)
        elif count < total - 1:
            testX = np.append(testX, x)
            testY = np.append(testY, y)
        else:
            testX = np.append(testX, x)
            testY = np.append(testY, y)
            result = test(x, y)
            print("score :", result)
            save_model(s)
    stop = timeit.default_timer()
    print ("Time used: %.2f s" % (stop - start))