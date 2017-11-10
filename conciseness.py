import sys
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import csv
from collections import Counter
from Lazada import *
from numpy import hstack
from sklearn.svm import SVC,LinearSVC
from pprint import pprint
from gensim.models import Doc2Vec
from collections import namedtuple

def label_sentences(data):
    labeled_sentences = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(data):
        # for text in sentences:
        text = ''.join(text)
        words = text.lower().split()
        tags = [i]
        labeled_sentences.append(analyzedDocument(words, tags))

    return labeled_sentences


def train_doc2vec_model(labeled_sentences):
    model = Doc2Vec(dm=0, dm_mean =1, size=200, window=10, min_count=1, workers=20, alpha=0.025, min_alpha=0.025)
    model.build_vocab(labeled_sentences)
#    model.train(labeled_sentences)
#    model.train(labeled_sentences,total_examples=model.corpus_count, epochs=model.iter)
    for epoch in range(10):
        model.train(labeled_sentences,total_examples=model.corpus_count)
        model.alpha -= 0.002
        model.min_alpha = model.alpha

    return model

def clean_description(desc):
    meaningful_words = ''
    tag_flag = 0
    for ch in desc:
        if ch == '<':
            tag_flag = 1
            continue
        elif ch == '>':
            tag_flag = 0
            meaningful_words += ' '
            continue
        elif tag_flag == 1:
            continue
        else:
            meaningful_words += ch
            meaningful_words = ' '.join(meaningful_words.split())
    return meaningful_words


def get_titles(data):
    return [row[2] for row in data]

def get_cat1(data):
    return [row[3] for row in data]

def get_cat2(data):
    return [row[4] for row in data]

def get_cat3(data):
    return [row[5] for row in data]

def get_desc(data):
    descs = [row[6] for row in data]
    return [clean_description(d) for d in descs]

def get_length_based_feats (data):
    titles = get_titles(data)
    descs = get_desc(data)
    length_feats = []
    for t,d in zip(titles,descs):
        title_char_len = len(t)
        title_word_len = len(t.split())
        desc_char_len = len(d)
        desc_word_len = len(d.split())
        char_len_diff = desc_char_len - title_char_len
        word_len_diff = desc_word_len - title_word_len
        length_feats.append([title_char_len,title_word_len,
                        desc_char_len,desc_word_len,
                        char_len_diff,word_len_diff])

    length_feats = np.array (length_feats)
    return length_feats

def run(training_fn, label_fn, testing_fn, pred_fn):
    training_data = read_data(training_fn)
    training_labels = read_labels(label_fn)
    test_data = read_data(testing_fn)
    print 'loaded training data and labels'
    print 'training data labels distribution: ',Counter(training_labels)

    train_length_feats = get_length_based_feats(training_data)
    test_length_feats = get_length_based_feats(test_data)
    print 'train_length_feats shape: ', train_length_feats.shape
    print 'test_length_feats shape: ', test_length_feats.shape

    train_title_sen = label_sentences(get_titles(training_data))
    train_model = train_doc2vec_model(train_title_sen)
    train_title_vects = np.array([train_model.docvecs[i] for i in range(0,len(get_titles(training_data)))])
    print 'Doc2Vec representation of training data'
    print 'train_title_vects shape: ', train_title_vects.shape

    train_desc_sen = label_sentences(get_desc(training_data))
    train_model = train_doc2vec_model(train_desc_sen)
    train_desc_vects = np.array([train_model.docvecs[i] for i in range(0,len(get_desc(training_data)))])
    print 'train_decs_vects.shape shape: ', train_desc_vects.shape

    test_title_sen = label_sentences(get_titles(test_data))
    test_model = train_doc2vec_model(test_title_sen)
    test_title_vects = np.array([test_model.docvecs[i] for i in range(0,len(get_titles(test_data)))])
    print 'Doc2Vec representation of test data'
    print 'test_title_vects shape: ',test_title_vects.shape

    test_desc_sen = label_sentences(get_desc(test_data))
    test_model = train_doc2vec_model(test_desc_sen)
    test_desc_vects = np.array([test_model.docvecs[i] for i in range(0,len(get_desc(test_data)))])
    print 'Doc2Vec representation of test data'
    print 'test_desc_vects shape: ',test_desc_vects.shape

    # title_vectorizer = TfidfVectorizer(lowercase=True,max_df=1.0, min_df=1,use_idf=False,ngram_range=(1,1))
    # title_vectorizer = CountVectorizer(lowercase=True,max_df=0.95, min_df=1,ngram_range=(1,1))
    # cat1_vectorizer = TfidfVectorizer(lowercase=True,max_df=1.0, min_df=1,use_idf=False)
    # cat2_vectorizer = TfidfVectorizer(lowercase=True,max_df=1.0, min_df=1,use_idf=False)
    # cat3_vectorizer = TfidfVectorizer(lowercase=True,max_df=1.0, min_df=1,use_idf=False)
    # desc_vectorizer = TfidfVectorizer(lowercase=True,max_df=1.0, min_df=1,use_idf=False,ngram_range=(1,3))

    # train_title_vects = title_vectorizer.fit_transform(get_titles(training_data))
    # test_title_vects = title_vectorizer.transform(get_titles(test_data))
    # train_cat1_vects = cat1_vectorizer.fit_transform(get_cat1(training_data))
    # train_cat2_vects = cat2_vectorizer.fit_transform(get_cat2(training_data))
    # train_cat3_vects = cat3_vectorizer.fit_transform(get_cat3(training_data))
    # train_desc_vects = desc_vectorizer.fit_transform(get_desc(training_data))
    #
    # print 'obatined vectors for individual modalities: ' \
    # ' title: {}, cat1: {}, cat2: {}, cat3: {}'.format(
    # train_title_vects.shape,
    # train_cat1_vects.shape,
    # train_cat2_vects.shape,
    # train_cat3_vects.shape,
    # train_desc_vects.shape)
    #
    train_full_vects = hstack((train_title_vects, train_desc_vects))
    test_full_vects = hstack((test_title_vects, test_desc_vects))
    #
    # print 'concatenated training vectors shape: ',train_full_vects.shape
    # run_n_times_with_cv(train_country_vects, training_labels, n=5)
    # run_n_times_with_cv(train_title_vects, training_labels, n=5)
    # run_n_times_with_cv(train_cat1_vects, training_labels, n=5)
    # run_n_times_with_cv(train_cat2_vects, training_labels, n=5)
    # run_n_times_with_cv(train_cat3_vects, training_labels, n=5)
    # run_n_times_with_cv(train_type_vects, training_labels, n=5)
    # Params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # 'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
    # 'kernel':['rbf']}
    # Classifier = SVC()
    # Params = {'fit_prior': [True, False], 'alpha': [0, 0.25, 5, 0.75, 1]}
    # Classifier = MultinomialNB()
    #
    Params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    Classifier = LinearSVC()

    train_full_len_vects = hstack((train_full_vects, train_length_feats))
    test_full_len_vects = hstack((test_full_vects, test_length_feats))
    run_n_times_with_cv(train_full_len_vects, training_labels, Classifier, Params)


#    train_title_len_vects = hstack((train_title_vects, train_length_feats))
#    test_title_len_vects = hstack((test_title_vects, test_length_feats))
#    run_n_times_with_cv(train_title_len_vects, training_labels, Classifier, Params)


    best_model = get_best_model_from_trainingset(train_full_len_vects, training_labels,
                                                    Classifier, Params)
    # Preds = best_model.predict_proba(test_title_len_vects)
    Preds = best_model.predict_proba(test_full_len_vects)
    y_preds = Preds[:, 1]

    with open(pred_fn,'w') as fh:
        for p in y_preds:
            print>>fh,p

def main(args):
    training_fn = '../data/training/data_train.csv'
    # label_fn = '../data/training/clarity_train.labels'
    label_fn = '../data/training/conciseness_train.labels'
    testing_fn = '../data/validation/data_valid.csv'
    pred_fn = 'conciseness_valid.predict'
    run(training_fn, label_fn, testing_fn, pred_fn)

if __name__ == '__main__':
    main(sys.argv[1:])