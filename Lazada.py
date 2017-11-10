import sys,random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import csv
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn import grid_search
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import math
from sklearn.metrics.scorer import make_scorer
from sklearn.calibration import CalibratedClassifierCV


def single_rmse(y_pred,y_ref):
    nr = float(math.pow(y_pred-y_ref,2))
    rmse = math.sqrt (nr)
    return rmse

def get_rmse(y_preds,y_refs):
    N = len(y_preds)
    nr = []
    for i,y_pred in enumerate(y_preds):
        y_ref = y_refs[i]
        nr.append(math.pow(y_pred-y_ref,2))

    nr = sum(nr)
    dr = float(N)
    rmse = math.sqrt (nr/dr)
    return rmse

def get_best_model_from_trainingset (X,Y, base_classifier,Params):
    my_scorer = make_scorer(get_rmse, greater_is_better=False)
    Classifier = grid_search.GridSearchCV(base_classifier, Params, n_jobs=-1, cv=3,
    scoring=my_scorer)
    Classifier.fit(X, Y)
    try:
        print 'best estimator after 5 fold CV: ', Classifier.best_estimator_
    except:
        pass

    Classifier = CalibratedClassifierCV (Classifier.best_estimator_)
    Classifier.fit(X, Y)
    return Classifier

def run_n_times_with_cv (X,Y, base_classifier,Params,n=5):
    rmses = []
    my_scorer = make_scorer(get_rmse, greater_is_better=False)
    # my_scorer = make_scorer(single_rmse, greater_is_better=False)
    for i in xrange(n):
        print 'run ', i + 1
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,
                                                            random_state = random.randint(0,100))
        print 'computed training and testing matrices', X_train.shape, X_test.shape
        # Params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        # Classifier = grid_search.GridSearchCV(LinearSVC(class_weight='balanced'),
        # Params, n_jobs=-1, cv=3,
        # scoring=my_scorer)
        # Params = {'fit_prior':[True,False],'alpha':[0,0.25,5,0.75,1]}
        # Classifier = grid_search.GridSearchCV(MultinomialNB(),
        # Params, n_jobs=-1, cv=3,
        # scoring = my_scorer)
        Classifier = grid_search.GridSearchCV(base_classifier, Params, n_jobs=-1, cv=3, scoring=my_scorer)
        Classifier.fit(X_train, y_train)

    try:
        print 'best estimator after 5 fold CV: ', Classifier.best_estimator_
    except:
        pass

    Classifier = CalibratedClassifierCV (Classifier.best_estimator_)
    Classifier.fit(X_train, y_train)
    # PerformFeatAnalysis (Classifier, X_train, Y, Vocab)

    Preds = Classifier.predict_proba(X_test)
    y_preds = Preds[:,1]
    y_refs = y_test

    rmse = get_rmse(y_preds,y_refs)
    print 'For run: {}, rmse: {}'.format(i,rmse)
    rmses.append(rmse)
    rmses = np.array(rmses)
    print 'Average rmse: ', rmses.mean()
    print 'Std rmse: ', rmses.std()


def read_data(filename):
    with open(filename, 'r') as data_file:
        data_reader = csv.reader(data_file)
        return [row for row in data_reader]

def read_labels(filename):
    with open(filename, 'r') as label_file:
        return [int(line.strip()) for line in label_file]

def write_predictions(filename, predictions):
    with open(filename, 'w') as pred_file:
        pred_file.writelines(map(lambda x: str(x) + '\n', predictions))

def get_titles(data):
    return [row[2] for row in data]

def run(training_fn, label_fn, testing_fn, pred_fn):
    training_data = read_data(training_fn)
    training_labels = read_labels(label_fn)
    print 'loaded training data and labels'
    print 'training data labels distribution: ',Counter(training_labels)

    vectorizer = CountVectorizer()
    training_vectors = vectorizer.fit_transform(get_titles(training_data))
    print 'training vectors shape: ',training_vectors.shape

    #run_n_times_with_cv (training_vectors,training_labels,base_classifier,Params)
    classifier = MultinomialNB()
    classifier.fit(training_vectors, training_labels)
    testing_data = read_data(testing_fn)
    testing_vectors = vectorizer.transform(get_titles(testing_data))
    print 'testing vectors shape: ', testing_vectors.shape
    y = classifier.predict_proba(testing_vectors)
    write_predictions(pred_fn, y[:,1]) # output the probability value of predicting '1'

def main(args):
    # training_fn = args[0]
    # label_fn = args[1]
    # testing_fn = args[2]
    # pred_fn = args[3]
    training_fn = '../data/training/data_train.csv'
    label_fn = '../data/training/clarity_train.labels'
    testing_fn = '../data/validation/data_valid.csv'
    pred_fn = 'clarity_valid.predict'
    run(training_fn, label_fn, testing_fn, pred_fn)

if __name__ == '__main__':
    main(sys.argv[1:])