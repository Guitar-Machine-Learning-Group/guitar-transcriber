import glob
import numpy as np
import timeit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
import pickle
from sklearn.externals import joblib

clf = OneVsRestClassifier(SVC(kernel='poly'))
try:
    s = joblib.load('svm.pkl')
except:
    print("new clf")
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
    score = c.score(x, y)
    return score

def save_model(clf):
    joblib.dump(clf, 'svm.pkl')

def load_model():
    return joblib.load('svm.pkl')



if __name__ == "__main__":
    start = timeit.default_timer()
    # pre-load data
    featureFileList = glob.glob('./preprocess1/features/*', recursive=True)
    labelFileList = glob.glob('./preprocess1/labels/*', recursive=True)
    if len(featureFileList) != len(labelFileList):
        print("Data is not matched")
        exit(-1)
    total = len(labelFileList)
    # trainSize = int(total * 0.8)
    trainSize = 2
    testX = np.array([])
    testY = np.array([])
    trainX = np.array([])
    trainY = np.array([])

    for count in range(total):
        selector = np.random.randint(0, len(featureFileList))
        featureFileName = featureFileList.pop(selector)

        for i in range(len(labelFileList)):
            if labelFileList[i].split('/')[-1].split('.')[0] == featureFileName.split('/')[-1].split('.')[0]:
                labelFileName = labelFileList.pop(i)
                break
        x = np.load(featureFileName)
        y = np.load(labelFileName)
        if count < trainSize:
            # train(x, y)
            # print(x.shape)
            # print(y.shape)
            trainX = np.append(trainX, x)
            trainY = np.append(trainY, y)
        elif count < total:
            if count == trainSize:
                train(trainX.reshape(-1,2048), trainY.reshape(-1, 51))
            testX = np.append(testX, x)
            testY = np.append(testY, y)
            if count == total - 1:
                result = test(testX.reshape(-1, 2048), testY.reshape(-1, 51))
                print("score :", result)
        # save_model(s)
    stop = timeit.default_timer()
    print ("Time used: %.2f s" % (stop - start))