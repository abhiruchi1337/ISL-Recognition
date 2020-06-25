import numpy as np
import cv2
import os
import csv
import sklearn.metrics as sm
from brisk_processing import descriptor
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import random
import warnings
import pickle
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as lr
import numpy as np
import sklearn.metrics as sm

path="train"
label=0
img_descs=[]
y=[]

#utility functions
def cluster_features(X_clust, img_descs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Parameters:
    -----------
    X : list of lists of SIFT descriptors (img_descs)
    training_idxs : array/list of integers
        Indicies for the training rows in img_descs
    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features
    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters
    training_descs = X_clust
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)

    print ('%i descriptors before clustering' % all_train_descriptors.shape[0])

    # Cluster descriptors to get codebook
    print ('Clustering with parameters: %s...' % repr(cluster_model))
    print ('Clustering on training set to get codebook of %i words' % n_clusters)

    # train kmeans or other cluster model on those descriptors selected above
    cluster_model.fit(all_train_descriptors)
    pickle.dump(cluster_model, open('cluster_model_new.sav','wb'))
    print ('done clustering. Using clustering model to generate BoW histograms for each image.')

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print ('Training examples generated.')

    return X, cluster_model

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def calc_accuracy(method,label_test,pred):
    print("accuracy score for ",method,sm.accuracy_score(label_test,pred))
    print("all precision_scores for ",method,sm.precision_score(label_test,pred,average='None'))
    print("weighted precision_score for ",method,sm.precision_score(label_test,pred,average='weighted'))
    print("all f1 score for ",method,sm.f1_score(label_test,pred,average='None'))
    print("f1 score for ",method,sm.f1_score(label_test,pred,average='weighted'))
    print("all recall score for ",method,sm.recall_score(label_test,pred,average='None'))
    print("recall score for ",method,sm.recall_score(label_test,pred,average='weighted'))

def predict_svm(X, y, X_train, X_test, y_train, y_test):
    svc=SVC(kernel='linear')
    #svc=SVC(kernel='rbf', gamma=0.001, C=10)
    print("========  SVM  ========")
    svc.fit(X_train,y_train)
    pickle.dump(svc, open('svm_trained_brisk_.sav','wb'))
    y_pred=svc.predict(X_test)
    calc_accuracy("SVM",y_test,y_pred)
    np.savetxt('submission_brisk_svm.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
##    cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=0)
##    estimator = SVC(kernel='linear')
##    plot_learning_curve(estimator, "SVM learning curve", X, y, cv=cv, n_jobs=4)
##    plt.show()
##---------------------------grid search for svm-------------------------------##
##    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
##                     'C': [0.1, 1, 10, 100]},
##                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
##    grid_search = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv=5)
##    grid_search.fit(X_train, y_train)
##    print("Best parameters set found on development set:")
##    print()
##    print(grid_search.best_params_)
##-----------------------------------------------------------------------------##
    
    

def predict_lr(X, y, X_train, X_test, y_train, y_test):
    clf = lr(solver='lbfgs', multi_class='ovr')
    print("======Logistic Regression======")
    clf.fit(X_train,y_train)
    pickle.dump(clf, open('logreg_trained_brisk_.sav','wb'))
    y_pred=clf.predict(X_test)
    calc_accuracy("Logistic regression",y_test,y_pred)
    np.savetxt('submission_brisk_lr.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
##    cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=0)
##    estimator = lr(solver='lbfgs')
##    plot_learning_curve(estimator, "Logistic Regression learning curve", X, y, cv=cv, n_jobs=4)
##
##    plt.show()


def predict_nb(X, y, X_train, X_test, y_train, y_test):
    clf = nb()
    print("======== Naive Bayes ========")
    clf.fit(X_train,y_train)
    pickle.dump(clf, open('naivebayes_trained_brisk_.sav','wb'))
    y_pred=clf.predict(X_test)
    calc_accuracy("Naive Bayes",y_test,y_pred)
    np.savetxt('submission_brisk_nb.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
    #cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    #estimator = nb()
    #plot_learning_curve(estimator, "Naive Bayes learning curve", X, y, cv=cv, n_jobs=4)

    #plt.show()
    


def predict_knn(X, y, X_train, X_test, y_train, y_test):
    clf=knn(n_neighbors=3)
    print("======= KNN =======")
    clf.fit(X_train,y_train)
    pickle.dump(clf, open('knn_trained_brisk_.sav','wb'))
    y_pred=clf.predict(X_test)
    calc_accuracy("K nearest neighbours",y_test,y_pred)
    np.savetxt('submission_brisk_knn.csv', np.c_[range(1,len(y_test)+1),y_pred,y_test], delimiter=',', header = 'ImageId,Label,TrueLabel', comments = '', fmt='%d')
##    cv = ShuffleSplit(n_splits=10, test_size=0.4, random_state=0)
##    estimator = knn(n_neighbors=3)
##    plot_learning_curve(estimator, "KNN learning curve", X, y, cv=cv, n_jobs=4)
##
##    plt.show()


#loading images from dataset
for (dirpath,dirnames,filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for(direcpath,direcnames,files) in os.walk(path+"\\"+dirname):
            for file in files:
                img_path=path+"\\\\"+dirname+"\\\\"+file
                print(img_path)
                des=descriptor(img_path)
                img_descs.append(des)
                y.append(label)
        label=label+1

##----------Convenience debugging commands---------##
        
##with open('imgdescs', 'wb') as fp:
##    pickle.dump(img_descs, fp)
##with open('y', 'wb') as fp:
##    pickle.dump(y,fp)

##with open("imgdescs", "rb") as fp:   # Unpickling
##    img_descs = pickle.load(fp)
##with open("y", "rb") as fp:   # Unpickling
##    y = pickle.load(fp)
#--------------------------------------------------##

print('done loading images')
y=np.array(y)

#splitting image descriptors into training set for clustering
X_clust_train, X_clust_test, y_train, y_test=train_test_split(img_descs, y, test_size=0.4)

##val_size=0.0
##X_clust_train, X_val, y_train, y_val 
##    = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

#clustering and creating histograms using kmeans minibatch cluster model
X, cluster_model = cluster_features(X_clust_train, img_descs, MiniBatchKMeans(n_clusters=150))

#splitting dataset into test, train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

print('Train size:',len(X_train),"Test size:",len(X_test))

#using classification methods
predict_knn(X, y, X_train, X_test, y_train, y_test)
predict_svm(X, y, X_train, X_test,y_train, y_test)
predict_lr(X, y, X_train, X_test,y_train, y_test)
predict_nb(X, y, X_train, X_test,y_train, y_test)
