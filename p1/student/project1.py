"""EECS 445 - Fall 2022.

Project 1
"""

import pandas as pd
import numpy as np
import nltk
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from matplotlib import pyplot as plt
from random import uniform
from nltk.corpus import stopwords


from helper import *

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)


#################### 6 modified extract_word() ##################
def extract_word_modified(input_string):
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    # TODO: Implement this function
    input_string = input_string.lower()
    for character in string.punctuation:
        input_string = input_string.replace(character, ' ')
    result = input_string.split()
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    # print("stop_words: ", stop_words)
    # print("list before: ", result)
    result = [w for w in result if not w in stop_words]
    # print("list: ", result)
    return result


def extract_dictionary_modified(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    # TODO: Implement this function
    i = 0
    for sentence in df['text']:
        extract_array = extract_word_modified(sentence)
        for word in extract_array:
            if (word not in word_dict):
                word_dict[word] = i
                i += 1

    # print(len(word_dict))
    return word_dict


def generate_feature_matrix_modified(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    ##########tfidf#############
    # number_of_reviews = df.shape[0]
    # number_of_words = len(word_dict)
    # feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    # tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(word_dict)
    # return tfidf_vectorizer_vectors

    ################################

    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # print("number_of_reviews ", number_of_reviews)
    # TODO: Implement this function
    i = 0
    for sentence in df['text']:
        extract_array = extract_word_modified(sentence)
        for word in extract_array:
            if word in word_dict:
                feature_matrix[i][word_dict[word]] = 1
        i += 1

    return feature_matrix


###############################################################################
def extract_word(input_string):
    """Preprocess review into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It's my favorite course!")
    > ["i", "love", "eecs", "445", "it", "s", "my", "favorite", "course"]

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    input_string = input_string.lower()
    for character in string.punctuation:
        input_string = input_string.replace(character, ' ')
    result = input_string.split()

    return result


def extract_dictionary(df):
    """Map words to index.

    Reads a pandas dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was
    found).

    E.g., with input:
        | text                          | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index should be autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    # TODO: Implement this function
    i = 0
    for sentence in df['text']:
        extract_array = extract_word(sentence)
        for word in extract_array:
            if (word not in word_dict):
                word_dict[word] = i
                i += 1

    # print(len(word_dict))
    return word_dict


def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.

    Reads a dataframe and the dictionary of unique words to generate a matrix
    of {1, 0} feature vectors for each review.  Use the word_dict to find the
    correct index to set to 1 for each place in the feature vector. The
    resulting feature matrix should be of dimension (# of reviews, # of words
    in dictionary).

    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    # print("number_of_reviews ", number_of_reviews)
    # TODO: Implement this function
    i = 0
    for sentence in df['text']:
        extract_array = extract_word(sentence)
        for word in extract_array:
            if word in word_dict:
                feature_matrix[i][word_dict[word]] = 1
        i += 1

    return feature_matrix
# my################################3


def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
    Returns:
        the performance as an np.float64
    """
    ############### challenge #####################

    n = len(y_true)
    conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0, -1])
    # top = 0
    # for i in range(len(conf_matrix)):
    #     for j in range(len(conf_matrix)):
    #         if i == j:
    #             top += conf_matrix[i][j]
    top = conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[2][2]
    return top / n
    ###############################################

    # # TODO: Implement this function
    # if metric == "auroc":
    #     return metrics.roc_auc_score(y_true, y_pred)

    # # print(metrics.confusion_matrix(y_true, y_pred, labels=[1,-1]))
    # tp = metrics.confusion_matrix(y_true, y_pred, labels=[1, -1])[0][0]
    # fp = metrics.confusion_matrix(y_true, y_pred, labels=[1, -1])[1][0]
    # tn = metrics.confusion_matrix(y_true, y_pred, labels=[1, -1])[1][1]
    # fn = metrics.confusion_matrix(y_true, y_pred, labels=[1, -1])[0][1]

    # if metric == "accuracy":
    #     if (tp+fp+tn+fn == 0):
    #         return 0
    #     else:
    #         return (tp+tn)/(tp+fp+tn+fn)

    # elif metric == "f1-score":
    #     # temp = tp/(tp+fp) + tn/(tn+fp)

    #     if (tp+fp == 0):
    #         precision = 0
    #     else:
    #         precision = tp/(tp+fp)

    #     if (tp+fn == 0):
    #         sensitivity = 0
    #     else:
    #         sensitivity = tp/(tp+fn)

    #     if (precision+sensitivity == 0):
    #         return 0
    #     else:
    #         return (2*precision*sensitivity)/(precision+sensitivity)

    # elif metric == "precision":
    #     if (tp+fp == 0):
    #         return 0
    #     else:
    #         return (tp/(tp+fp))

    # elif metric == "sensitivity":
    #     if (tp+fn == 0):
    #         return 0
    #     else:
    #         return (tp/(tp+fn))

    # elif metric == "specificity":
    #     if (tn+fp == 0):
    #         return 0
    #     else:
    #         return (tn/(tn+fp))

    # This is an optional but very useful function to implement.
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates and returns the k-fold cross-validation performance metric for
    classifier clf by averaging the performance across folds.
    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
    Returns:
        average 'test' performance across the k folds as np.float64
    """
    # TODO: Implement this function
    # HINT: You may find the StratifiedKFold from sklearn.model_selection
    # to be useful

    # Put the performance of the model on each fold in the scores array
    scores = []
    skf = StratifiedKFold(n_splits=k)
    # Stratified K-Folds cross-validator.
    # Provides train/test indices to split data in train/test sets.

    for train_index, test_index in skf.split(X, y):
        # print(train_index, " ", test_index) # debug
        # print("TRAIN:", train_index, "TEST:", test_index)# debug

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)  # train the model using the training data

        if (metric == "auroc"):
            # predict using the test data
            y_pred = clf.decision_function(X_test)
        else:
            y_pred = clf.predict(X_test)  # predict using the test data

        scores.append(performance(y_test, y_pred, metric))
        # measure the perfomance by comparin the true y_test and the predict y_pred

    return np.array(scores).mean()

# def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
#     """
#     Return a linear svm classifier based on the given
#     penalty function and regularization parameter c.
#     """
#     if degree == 1:
#         if penalty == 'l2':
# ## will not use LinearSVC function here to match with classes→implementation
#             clf = SVC(kernel='linear', C=c, degree=degree, coef0=r, class_weight=class_weight)
#         else :
#             clf = LinearSVC(penalty='l1', dual=False, C=c, class_weight='balanced', max_iter=10000)
#     elif degree == 2 :
#         clf = SVC(kernel='poly', C=c, degree=degree, coef0=r, class_weight=class_weight, gamma='auto')
#     return clf


def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy',
             other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1"ß)
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    # TODO: Implement this function
    best_c = 0.0
    best_performance_so_far = 0.0
    for c in C_range:
        clf = LinearSVC(C=c,
                        random_state=445, loss="hinge", penalty="l2", dual=True)
        # clf = select_classifier(penalty=penalty, c=c)

        # clf = select_classifier(penalty, c)
        # clf = SVC(C=c, kernel="linear", random_state=445)

        # print(c)
        perf = cv_performance(clf, X, y, k, metric)
        print(" - C:", '{0:5}'.format(c), " perf: ", '{0:5}'.format(perf))
        # print()
        if perf > best_performance_so_far:
            best_performance_so_far = perf
            best_c = c
    # print("best_c:", best_c)
    # print("best_performance_so_far:", best_performance_so_far)

    print(metric, "C: ", best_c, " Perf: ", best_performance_so_far)
    return best_c

    # HINT: You should be using your cv_performance function here
    # to evaluate the performance of each SVM


def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    # TODO: Implement this part of the function
    # Here, for each value of c in C_range, you should
    # append to norm0 the L0-norm of the theta vector that is learned
    # when fitting an L2- or L1-penalty, degree=1 SVM to the data (X, y)
    # i = 0
    for c in C_range:
        clf = LinearSVC(C=c, random_state=445, loss=loss,
                        penalty=penalty, dual=dual)
        clf.fit(X, y)  # train the model using the training data
        # norm0[i] = np.linalg.norm(clf.coef_[0], ord=0)
        # i += 1
        norm0.append(np.count_nonzero(clf.coef_))

    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
    calculating the k-fold CV performance for each setting on X, y.
    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default='accuracy'
                 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
                 and 'specificity')
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    # TODO: Implement this function
    # Hint: This will be very similar to select_param_linear, except
    # the type of SVM model you are using will be different...
    best_C_val, best_r_val = 0.0, 0.0
    best_performance_so_far = 0.0

    ############## Grid Search ################
    for c_ind in range(len(param_range)):
        for r_ind in range(len(param_range)):
            c = param_range[c_ind][0]
            r = param_range[r_ind][1]
            clf = SVC(kernel='poly', degree=2, C=c, coef0=r, gamma='auto')
            perf = cv_performance(clf, X, y, k, metric)
            # print(" - C:", '{0:5}'.format(c), " - r:",
            #      '{0:5}'.format(r), " perf: ", '{0:5}'.format(perf))
            if perf > best_performance_so_far:
                best_performance_so_far = perf
                best_C_val = c
                best_r_val = r

    ############## Random Search ################
    # for i in range(25):
    #     c = 10 ** uniform(-2, 3)
    #     r = 10 ** uniform(-2, 3)
    #     clf = SVC(kernel='poly', degree=2, C=c, coef0=r, gamma='auto')
    #     perf = cv_performance(clf, X, y, k, metric)
    #     if perf > best_performance_so_far:
    #         best_performance_so_far = perf
    #         best_C_val = c
    #         best_r_val = r
    ######################################################

    print(metric, "best_C_val: ", best_C_val, "best_r_val: ",
          best_r_val, " best_performance: ", best_performance_so_far)
    return best_C_val, best_r_val


def test_SVM(X_train, Y_train, X_test, Y_test, C, penalty='l2', metric="accuracy", class_weight='balanced', loss="hinge"):
    # clf = select_classifier(penalty=penalty, c=C, class_weight=class_weight, loss="hinge")
    # clf = LinearSVC(c=C, class_weight=class_weight,
    #                 loss="hinge", dual=False, max_iter=10000)
    clf = LinearSVC(C=C, random_state=445, loss="hinge",
                    class_weight=class_weight, penalty=penalty, dual=False)
    clf.fit(X_train, Y_train)

    if metric == "auroc":
        y_pred = clf.decision_function(X_test)
    else:
        y_pred = clf.predict(X_test)

    # y_pred = clf.predict(X_test) if (metric!="auroc") else clf.decision_function(X_test)
    return performance(Y_test, y_pred, metric)


# def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced', loss="hinge"):
#     if penalty == 'l2':
#         clf = SVC(kernel='linear', C=c, degree=degree,
#                   coef0=r, class_weight=class_weight, loss="hinge")
#     else:
#         clf = LinearSVC(penalty='l1', dual=False, C=c,
#                         class_weight='balanced', max_iter=10000)
#     return clf


def main():
    # Read binary data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_matrix AND extract_dictionary
    # X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
    #     fname="data/dataset.csv"
    # )
    # IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
    #     dictionary_binary, fname="data/dataset.csv"
    # )
    #     print(len(X_train))# 2662
    #     print(len(X_train[0]))  # 4920
    #     print(len(X_train[2661])) # 4920
    #     print(X_train[0][2663]) # 0
    #     print(X_train.shape)# (2662, 4920)
    # print(Y_train)

    #     print((X_train.sum())/(X_train.shape[0]))  # 11.530303030303031
    #     print(len(dictionary_binary)) #4920
    # cv_performance(clf, X, y, k=5, metric="accuracy")
    # print(X_train)
    # print(dictionary_binary)

    #######################3(c)ii find the most word#######################
    # sum_array = np.sum(X_train, axis=0)
    # max_index = np.argmax(sum_array)
    # # print(sum_array)
    # # print(max_index)
    # most_word = {
    #     i for i in dictionary_binary if dictionary_binary[i] == max_index}
    # print(most_word)
    ############################################################################
    # input_string = "It's a test sentence! Does it look CORRECT?"
    # extract_word(input_string)
    # extract_dictionary(load_data())
    #     C_range = np.logspace(-3, 3, 7)
    #     penalty = 'l2'
    mets = ["accuracy", "f1-score", "auroc",
            "precision", "sensitivity", "specificity"]

    #########################################################
    ####      4 Hyperparameter and Model Selection       ####
    #########################################################
    c_range = [10 ** -3, 10 ** -2, 10 ** -1,
               10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    m_s = ['accuracy', 'f1-score', 'auroc',
           'precision', 'sensitivity', 'specificity']

    ############## 4.1(b) #################

    # for m in m_s:
    #     select_param_linear(X_train, Y_train, k=5, metric=m, C_range=c_range)

    ############## 4.1(c) ##################
    # clf = LinearSVC(C=0.1, random_state=445,
    #                 loss="hinge", penalty="l2", dual=True)
    # clf.fit(X_train, Y_train)  # train the model using the training data
    # for m in m_s:
    #     if(m == "auroc"):
    #         # predict using the test data
    #         y_pred = clf.decision_function(X_test)
    #     else:
    #         y_pred = clf.predict(X_test)  # predict using the test data

    #     print(m, " ", performance(Y_test, y_pred, m))

    ################# 4.1(d) ################
    # plot_weight(X_train, Y_train, 'l2', c_range, 'hinge', dual=True)

    ################# 4.1(e) #################
    # clf = LinearSVC(C=0.1, random_state=445,
    #                 loss="hinge", penalty="l2", dual=True)
    # clf.fit(X_train, Y_train)  # train the model using the training data
    # for m in m_s:
    #     if(m == "auroc"):
    #         # predict using the test data
    #         y_pred = clf.decision_function(X_test)
    #     else:
    #         y_pred = clf.predict(X_test)  # predict using the test data

    #     print(m, " ", performance(Y_test, y_pred, m))

    ################# 4.2(a) ###################

    #############find c##############
    c_range2 = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0]
    # select_param_linear(X_train, Y_train, k=5,
    #                     metric='auroc', C_range=c_range2)
    # - C: 0.001  perf:  0.9235342569907916
    # - C:  0.01  perf:  0.9584071905708343
    # - C:   0.1  perf:  0.9730361110175962
    # - C:     1  perf:  0.9695321030463292
    # result: best C:  0.1 best Perf:  0.9730361110175962

    #############train a SVM with squared hinge loss on the entirety of the training data############
    # clf = LinearSVC(penalty='l1', loss='squared_hinge', C=0.1, dual=False)
    # clf.fit(X_train, Y_train)  # train the model using the training data
    # y_pred = clf.decision_function(X_test)  # predict using the test data

    # print("AUROC performance on the entire test: ",
    #      http://localhost:8888/edit/student/project1.py# performance(Y_test, y_pred, "auroc"))

    ################# 4.2(b) ######################
    # plot_weight(X_train, Y_train, 'l1', c_range2, 'squared_hinge', dual=False)

    ################# 4.3(a) ######################
    # pr = np.array([[10 ** -3, 10 ** -3],
    #                [10 ** -2, 10 ** -2],
    #                [10 ** -1, 10 ** -1],
    #                [10 ** 0, 10 ** 0],
    #                [10 ** 1, 10 ** 1],
    #                [10 ** 2, 10 ** 2],
    #                [10 ** 3, 10 ** 3]])

    # best_C_val, best_r_val = select_param_quadratic(
    #     X_train, Y_train, 5, metric="auroc", param_range=pr)

    # clf = SVC(kernel='poly', degree=2, C=best_C_val,
    #           coef0=best_r_val, gamma='auto')
    # clf.fit(X_train, Y_train)
    # y_pred = clf.decision_function(X_test)
    # print("AUROC performance on the entire test: ",
    #       performance(Y_test, y_pred, "auroc"))


######################################################

    #########################################################
    #### 5 Asymmetric Cost Functions and Class Imbalance ####
    #########################################################

    ######## 5.1(c) Arbitrary class weights ########

    # class_weight = {-1: 1, 1: 10}

    # for m in mets:
    #     #     perf_score = test_SVM(X_train, Y_train, X_test, Y_test,
    #     #                           0.01, penalty='l2', metric=m, class_weight=class_weight)
    #     clf = LinearSVC(C=0.01, random_state=445, loss="hinge",
    #                     class_weight=class_weight, penalty='l2')
    #     clf.fit(X_train, Y_train)

    #     if m == "auroc":
    #         y_pred = clf.decision_function(X_test)
    #     else:
    #         y_pred = clf.predict(X_test)
    #     perf_score = performance(Y_test, y_pred, m)
    #     print(m, " : ", perf_score)

    #############################################

    ######## 5.2(a) Imbalanced data ########
    # class_weight2 = {-1: 1, 1: 1}

    # for m in mets:
    #     # perf_score = test_SVM(IMB_features, IMB_labels, IMB_test_features,
    #     #                       IMB_test_labels, 0.01, penalty='l2', metric=m, class_weight=class_weight2, loss="hinge")
    #     clf = LinearSVC(C=0.01, random_state=445, loss="hinge",
    #                     class_weight=class_weight2, penalty='l2')
    #     clf.fit(IMB_features, IMB_labels)

    #     if m == "auroc":
    #         y_pred = clf.decision_function(IMB_test_features)
    #     else:
    #         y_pred = clf.predict(IMB_test_features)
    #     perf_score = performance(IMB_test_labels, y_pred, m)
    #     print(m, " : ", perf_score)

    ####### 5.3 choose appropriate class weights ########
    # best_scores = 0
    # best_wn = 0
    # best_wp = 0
    # for wn in range(1, 10):
    #     for wp in range(1, 10):
    #         print(" Wp:", '{0:3}'.format(wp/10),
    #               " Wn: ", '{0:3}'.format(wn/10))
    #         c_w3 = {-1: wn/10, 1: wp/10}
    #         # clf = select_classifier(penalty='l2', c=.1, class_weight=c_w3)
    #         clf = SVC(kernel='linear', C=0.01,
    #                   class_weight=c_w3)
    #         score = cv_performance(
    #             clf, IMB_features, IMB_labels, k=5, metric="AUROC")
    #         print(" - score:", '{0:7}'.format(score))
    #         if score > best_scores:
    #             best_scores = score
    #             best_wn = wn
    #             best_wp = wp

    # print(best_scores)
    # print(best_wn, " ", best_wp)

    ############## 5.3(b) ######################
    # class_weight3 = {-1: 9, 1: 6}

    # for m in mets:
    #     # perf_score = test_SVM(IMB_features, IMB_labels, IMB_test_features,
    #     #                       IMB_test_labels, 0.01, penalty='l2', metric=m, class_weight=class_weight2, loss="hinge")
    #     clf = LinearSVC(C=0.01, random_state=445, loss="hinge",
    #                     class_weight=class_weight3, penalty='l2')
    #     clf.fit(IMB_features, IMB_labels)

    #     if m == "auroc":
    #         y_pred = clf.decision_function(IMB_test_features)
    #     else:
    #         y_pred = clf.predict(IMB_test_features)
    #     perf_score = performance(IMB_test_labels, y_pred, m)
    #     print(m, " : ", perf_score)

    ########## 5.4 ROC Curve ###########
    # c_w = {-1: 1, 1: 1}
    # # clf1 = SVC(kernel='linear', C=0.01, class_weight=c_w)
    # clf1 = LinearSVC(C=0.01, random_state=445, loss="hinge",
    #                  class_weight=c_w, penalty='l2')

    # score1 = cv_performance(
    #     clf1, IMB_features, IMB_labels, k=5, metric='auroc')

    # c_w_custom = {-1: 9, 1: 6}
    # # clf2 = SVC(kernel='linear', C=0.01, class_weight=c_w_custom)
    # clf2 = LinearSVC(C=0.01, random_state=445, loss="hinge",
    #                  class_weight=c_w_custom, penalty='l2')

    # score2 = cv_performance(
    #     clf2, IMB_features, IMB_labels, k=5, metric='auroc')

    # print("score1: ", score1, " score2: ", score2)

    # clf1_disp = metrics.plot_roc_curve(
    #     clf1, X=IMB_test_features, y=IMB_test_labels, name='Wn=1 Wp=1')
    # clf2_cisp = metrics.plot_roc_curve(
    #     clf2, X=IMB_test_features, y=IMB_test_labels, name='Wn=9 Wp=6', ax=clf1_disp.ax_)
    # clf1_disp.figure_.suptitle("AUROC curve comparison")
    # plt.savefig('AUROC_Curve.png')
    # plt.close()
    ##################################
    #
    # Read multiclass data
    # TODO: Question 6: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels

    #########################################################
    ##################### 6 Challenge #######################
    #########################################################
    (multiclass_features,
     multiclass_labels,
     multiclass_dictionary) = get_multiclass_training_data()

    heldout_features = get_heldout_reviews(multiclass_dictionary)

    ############ First use the 5-fold CV to find the best hyperparameters######
    ################# linear #############
    # c_range3 = [10 ** -2, 10 ** -1,
    #             10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3]
    # best_c = select_param_linear(multiclass_features,
    #                              multiclass_labels, k=5, C_range=c_range3)

    # ################# quadratic ############
    # pr = np.array([[10 ** -3, 10 ** -3],
    #                [10 ** -2, 10 ** -2],
    #                [10 ** -1, 10 ** -1],
    #                [10 ** 0, 10 ** 0],
    #                [10 ** 1, 10 ** 1],
    #                [10 ** 2, 10 ** 2],
    #                [10 ** 3, 10 ** 3]])
    # best_C_val, best_r_val = select_param_quadratic(
    #     multiclass_features, multiclass_labels, 5, param_range=pr)

    # print("********** got train heldout features, making model... *********")
    # SVC(kernel='linear', C=0.1, degree=2, coef0=2, gamma = 'scale')

    ############# Second  #################
    # Use the c we found before
    # create a SVM.
    # Train this SVM on multiclass_features.
    # Test on heldout_features
    # clf = LinearSVC(C=0.1, random_state=445, loss="hinge", penalty="l2", dual=True)
    # print("********** making model... *********")

    # model = LinearSVC(C=1, random_state=445, loss="square hinge",
    #                  penalty = 'l1', dual = True)
    # model = LinearSVC(penalty='l1', loss='squared_hinge', C=0.1, dual=False)

    model = SVC(kernel='poly', degree=2, C=10, coef0=100, gamma='auto')

    # print("********** made the model, fitting... *********")
    model.fit(multiclass_features, multiclass_labels)
    # train the model using the training data
    # for m in m_s:
    #     if (m == "auroc"):
    #         # predict using the test data
    #         y_pred = clf.decision_function(X_test)
    #     else:
    #         y_pred = clf.predict(X_test)  # predict using the test data
    # print("********** fit the model, predicting... *********")
    predictions = model.predict(heldout_features)

    # print("accuracy ", performance(Y_test, y_pred, m))

    perf_score = cv_performance(
        model, multiclass_features, multiclass_labels, k=5)
    print(perf_score)
    # print("********** predicted, saving... *********")
    generate_challenge_labels(predictions, 'xinyup')

    # print("**************** DONE ****************")


if __name__ == "__main__":
    main()
