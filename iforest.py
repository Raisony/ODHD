import argparse
import numpy as np
import scipy.io as sio
from sklearn.externals import joblib
from os.path import dirname, join as pjoin
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def DataPreprocessing(dataset):
    mat_fname = pjoin('./', dataset)
    print('Loading', mat_fname)
    mat_contents = sio.loadmat(mat_fname)
    X, y = mat_contents['X'], mat_contents['y']
    y = y.reshape(-1)
    inliers, outliers = np.where(y == 0)[0], np.where(y == 1)[0]#     1 = outliers, 0 = inliers
    return X, y, inliers, outliers

def standardization(X, MAX, MIN):
    t = MIN + (MAX - MIN)/2
    return (X - t)/np.abs((MAX - MIN)/2)

def main(data, seed):
    
    dataset = './Dataset/' + data + '.mat'
    X_set, y_set, inner, outer = DataPreprocessing(dataset)
    np.random.seed  = seed
    p = outer.tolist() + np.random.choice(inner, 3*len(outer), replace = 0).tolist()
    position = [x for x in inner if x not in p]
    Xtrain, Xtest, ytrain, ytest = X_set[position], X_set[p], y_set[position],  y_set[p]
    Xtrain = standardization(Xtrain, Xtrain.max(), Xtrain.min())
    Xtest  = standardization(Xtest,Xtest.max(), Xtest.min())
    
    clf = IsolationForest(random_state = seed, max_samples = 100).fit(Xtrain)
    testLabels = ytest
    y_pred = clf.predict(Xtest)
    guessList = 1*(y_pred < 0)
    acc = np.mean(guessList == ytest)
    f1score = f1_score(testLabels, guessList)
    prauc = average_precision_score(testLabels, guessList)
    rocauc = roc_auc_score(testLabels, guessList)
    print("ISOLATION FOREST")
    print('ACC = ', acc)
    print('AP = ', prauc)
    print('F1-score = ', f1score)
    print('ROC AUC = ', rocauc)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='N', type=str)
    parser.add_argument('--seed', metavar='N', type=int)


    args = parser.parse_args()
    main(args.data, args.seed)
