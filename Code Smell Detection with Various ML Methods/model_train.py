from scipy.io import arff
import pandas as pd
from os import path
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

import warnings


def create_tree(X_train, X_test, Y_train, Y_test):
    # print(X_train.shape,Y_train.shape)
    tree_clf = DecisionTreeClassifier(criterion="gini",min_samples_leaf=20,max_depth=3)
    # out=cross_val_score(tree_clf, X_train, Y_train, cv=10, scoring="accuracy")
    # print(out)
    tree_clf.fit(X_train, Y_train)
    print("Decision tree accuracy",cross_val_score(tree_clf,X_test,Y_test,cv=4,scoring="accuracy"))
    # print("Decision tree accuracy: " + str(accuracy_score(Y_test, tree_clf.predict(X_test))))
    # print("Decision tree f1-score: " + str(f1_score(Y_test, tree_clf.predict(X_test))))
    return tree_clf
def random_forest(X_train, X_test, Y_train, Y_test):
    rand_clf=RandomForestClassifier(n_estimators=150,criterion="gini",min_samples_leaf=25,max_depth=3)
    rand_clf.fit(X_train,Y_train)
    print("Random Forest accuracy",cross_val_score(rand_clf,X_test,Y_test,cv=4,scoring="accuracy"))
    # print("Random Forest accuracy: " + str(accuracy_score(Y_test, rand_clf.predict(X_test))))
    # print("Random Forest f1-score: " + str(f1_score(Y_test, rand_clf.predict(X_test))))
    return rand_clf
def naiveBayes(X_train, X_test, Y_train, Y_test):
    bayes_clf=GaussianNB()
    bayes_clf.fit(X_train,Y_train)
    print("Naive Bayes accuracy",cross_val_score(bayes_clf,X_test,Y_test,cv=4,scoring="accuracy"))
    # print("Naive Bayes accuracy: " + str(accuracy_score(Y_test, bayes_clf.predict(X_test))))
    # print("Naive Bayes f1-score: " + str(f1_score(Y_test, bayes_clf.predict(X_test))))
    return bayes_clf
def svm(X_train, X_test, Y_train, Y_test):
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    svm_lin_clf=SVC(kernel='linear',max_iter=5000)
    svm_lin_clf.fit(X_train,Y_train)
    print("SVM Linear Kernel accuracy",cross_val_score(svm_lin_clf,X_test,Y_test,cv=4,scoring="accuracy"))

    svm_poly_clf = SVC(kernel='poly',degree=8,max_iter=20000)
    svm_poly_clf.fit(X_train, Y_train)
    print("SVM Polynomial Kernel accuracy",cross_val_score(svm_poly_clf,X_test,Y_test,cv=4,scoring="accuracy"))

    svm_rbf_clf = SVC(kernel='rbf', random_state=0,gamma=.01,C=10)
    svm_rbf_clf.fit(X_train, Y_train)
    print("SVM RBF Kernel accuracy",cross_val_score(svm_rbf_clf,X_test,Y_test,cv=4,scoring="accuracy"))

    svm_sig_clf = SVC(kernel='sigmoid')
    svm_sig_clf.fit(X_train, Y_train)
    print("SVM Sigmoid Kernel accuracy",cross_val_score(svm_sig_clf,X_test ,Y_test,cv=4,scoring="accuracy"))

    return svm_lin_clf,svm_poly_clf,svm_rbf_clf,svm_sig_clf
def make_splits(filename):
    # 'feature-envy.arff'
    if(not path.exists(filename)):
        print("error",filename,"not found")
        sys.exit()
    data = arff.loadarff(filename)
    print(filename)
    df = pd.DataFrame(data[0])
    X_copy = df.iloc[:, :-1].copy()
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_copy)
    new_X = imputer.transform(X_copy)
    Y_envy = df.iloc[:, -1].values
    le = LabelEncoder()
    Y_envy=le.fit_transform(Y_envy)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
    for train_index, test_index in sss.split(new_X, Y_envy):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = new_X[train_index], new_X[test_index]
        Y_train, Y_test = Y_envy[train_index], Y_envy[test_index]


    return X_train, X_test, Y_train, Y_test
def make_model(filename):
    X_train, X_test, Y_train, Y_test = make_splits(filename)
    tree = create_tree(X_train, X_test, Y_train, Y_test)
    rand_forest = random_forest(X_train, X_test, Y_train, Y_test)
    bayes = naiveBayes(X_train, X_test, Y_train, Y_test)
    svmLinear, svmPoly, svmRBF, svmSig = svm(X_train, X_test, Y_train, Y_test)
    return [X_train,X_test,Y_train,Y_test],[tree,rand_forest,bayes,svmLinear,svmPoly,svmRBF,svmSig]
