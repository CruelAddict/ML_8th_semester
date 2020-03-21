from os import listdir
import re
import numpy as np
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

n_gram_range = (1, 1)
alpha_smoothing = 1e-10
lambdas_best = [1e190, 1]


def parse_doc_line(line):
    parsed = re.search(r'\d[\d\s]+\d', line)
    return "empty" if parsed is None else parsed[0]


def get_roc_point(clf, x_set, y_set, threshold):
    loo = LeaveOneOut()
    vectorizer = CountVectorizer(ngram_range=n_gram_range)
    roc_predictions = np.empty(0)
    answers = np.empty(0)

    i = 1
    for train_index, test_index in loo.split(x_set):
        x_train = [obj for partition in x_set[train_index] for obj in partition]
        x_test = [obj for partition in x_set[test_index] for obj in partition]
        x_vectorized = vectorizer.fit_transform(x_train + x_test).toarray()
        x_train, x_test = x_vectorized[:len(x_train)], x_vectorized[-len(x_test):]
        y_train, y_test = y_set[train_index], y_set[test_index]
        clf.fit(x_train, y_train.flatten())
        answers = np.append(answers, y_test)
        roc_predictions = np.append(roc_predictions,
                                    ['spmsg' if prediction[0] <= threshold else 'legit' for prediction in
                                     clf.predict_proba(x_test)])
        print(f'Finished iteration {i} / 10')
        i += 1

    true_negatives_, true_positives_, false_negatives_, false_positives_ = 0, 0, 0, 0
    for prediction, answer in zip(roc_predictions, answers):
        if prediction == 'spmsg':
            if answer == 'spmsg':
                true_positives_ += 1
            else:
                false_positives_ += 1
        else:
            if answer == 'legit':
                true_negatives_ += 1
            else:
                false_negatives_ += 1
    roc_point_ = (
        1 - (true_negatives_ / (true_negatives_ + false_positives_)),
        true_positives_ / (true_positives_ + false_negatives_))
    return roc_point_


def get_cv_score(clf, x_set, y_set):
    loo = LeaveOneOut()
    vectorizer = CountVectorizer(ngram_range=n_gram_range)
    predictions = np.empty(0)
    answers = np.empty(0)

    i = 1
    for train_index, test_index in loo.split(x_set):
        x_train = [obj for partition in x_set[train_index] for obj in partition]
        x_test = [obj for partition in x_set[test_index] for obj in partition]
        x_vectorized = vectorizer.fit_transform(x_train + x_test).toarray()
        x_train, x_test = x_vectorized[:len(x_train)], x_vectorized[-len(x_test):]
        y_train, y_test = y_set[train_index], y_set[test_index]
        clf.fit(x_train, y_train.flatten())
        predictions = np.append(predictions, clf.predict(x_test))
        answers = np.append(answers, y_test)
        print(f'Finished iteration {i} / 10')
        i += 1

    true_negatives_, true_positives_, false_negatives_, false_positives_ = 0, 0, 0, 0
    for prediction, answer in zip(predictions, answers):
        if prediction == 'spmsg':
            if answer == 'spmsg':
                true_positives_ += 1
            else:
                false_positives_ += 1
        else:
            if answer == 'legit':
                true_negatives_ += 1
            else:
                false_negatives_ += 1
    f1_result = f1_score(answers, predictions, average='macro')
    return f1_result, true_negatives_, true_positives_, false_negatives_, false_positives_


parts_X = []
parts_Y = []

for part in range(1, 11):
    parts_X.append([])
    parts_Y.append([])
    for file in listdir(f'messages/part{part}'):
        f = open(f'messages/part{part}/{file}', "r")
        one = parse_doc_line(f.readline())
        f.readline()
        two = parse_doc_line(f.readline())
        curr_obj = one + " " + two
        parts_Y[-1].append(re.findall(r'\D+', file)[0])
        parts_X[-1].append(curr_obj)
        f.close()

roc_points = []
for thresh in range(0, 11):
    roc_points.append(get_roc_point(
        MultinomialNB(alpha=alpha_smoothing), np.array(parts_X), np.array(parts_Y), thresh / 10))

f1_points = []
true_positives_list = []
false_positives_list = []
true_negatives_list = []
false_negatives_list = []
lambda_ratios = [1, 1e5, 1e10, 1e20, 1e40, 1e80, 1e160, 1e190]
for lambda_ratio in lambda_ratios:
    f1, true_negatives, true_positives, false_negatives, false_positives = get_cv_score(
        MultinomialNB(class_prior=(lambda_ratio, 1), alpha=alpha_smoothing), np.array(parts_X), np.array(parts_Y))
    print(f'F1 score: {f1}\n True negatives: {true_negatives}\n True positives: {true_positives}\n False negatives: '
          f'{false_negatives}\n False positives: {false_positives}')
    f1_points.append(f1)
    true_positives_list.append(true_positives)
    false_positives_list.append(false_positives)
    true_negatives_list.append(true_negatives)
    false_negatives_list.append(false_negatives)

fig, plts = plt.subplots(3)
plts[0].margins(0.0)
plts[0].set_ylim(ymin=0)
plts[0].plot([point[0] for point in roc_points], [point[1] for point in roc_points])
plts[0].set_ylabel('Roc Curve')

plts[1].set_xscale('log')
plts[1].plot(lambda_ratios, f1_points, '-b')
plts[1].set_ylabel('F1 score')
plts[1].set_xlim(xmin=1)

plts[2].set_xscale('log')
plts[2].set_yscale('log')
plts[2].plot(lambda_ratios, true_positives_list, '-r', label='True positives')
plts[2].plot(lambda_ratios, false_positives_list, '-g', label='False positives')
plts[2].plot(lambda_ratios, true_negatives_list, '-b', label='True negatives')
plts[2].plot(lambda_ratios, false_negatives_list, '-y', label='False negatives')
plts[2].legend(loc="upper right")
plts[2].set_xlabel('Lambda_legit / Lambda_spam')
plts[2].set_xlim(xmin=1)
plt.show()
