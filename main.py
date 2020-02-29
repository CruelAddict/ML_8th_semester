import re

import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def get_weight_func(func_name, param):
    def epanechnikov(values):
        for i, distances_set in enumerate(values):
            for j, distance in enumerate(distances_set):
                values[i][j] = 1 - (distance ** 2 / param ** 2)
            values[i] = normalize_list(values[i])
        return values

    def gaussian(values):
        for i, distances_set in enumerate(values):
            for j, distance in enumerate(distances_set):
                values[i][j] = np.exp(-(distance ** 2 / 2 * param ** 2))
            values[i] = normalize_list(values[i])
        return values

    def tophat(values):
        for i, distances_set in enumerate(values):
            for j, distance in enumerate(distances_set):
                values[i][j] = 1 if distance < param else 0
            values[i] = normalize_list(values[i])
        return values

    def exponential(values):
        for i, distances_set in enumerate(values):
            for j, distance in enumerate(distances_set):
                values[i][j] = np.exp(-distance / param)
            values[i] = normalize_list(values[i])
        return values

    def linear(values):
        for i, distances_set in enumerate(values):
            for j, distance in enumerate(distances_set):
                values[i][j] = 1 - distance / param if distance < param else 0
            values[i] = normalize_list(values[i])
        return values

    def cosine(values):
        for i, distances_set in enumerate(values):
            for j, distance in enumerate(distances_set):
                values[i][j] = np.cos(np.pi * distance / (2 * param)) if distance < param else 0
            values[i] = normalize_list(values[i])
        return values

    def epanechnikov_variable(values):
        for i, distances_set in enumerate(values):
            threshold_variable = values[i][0] * param
            threshold_variable = threshold_variable if threshold_variable > 0.5 else 0.5
            for j, distance in enumerate(distances_set):
                values[i][j] = 1 - (distance ** 2 / threshold_variable ** 2)
            values[i] = normalize_list(values[i])
        return values

    def gaussian_variable(values):
        for i, distances_set in enumerate(values):
            threshold_variable = values[i][0] * param
            threshold_variable = threshold_variable if threshold_variable > 0.5 else 0.5
            for j, distance in enumerate(distances_set):
                values[i][j] = np.exp(-(distance ** 2 / 2 * threshold_variable ** 2))
            values[i] = normalize_list(values[i])
        return values

    def tophat_variable(values):
        for i, distances_set in enumerate(values):
            threshold_variable = values[i][0] * param
            threshold_variable = threshold_variable if threshold_variable > 0.5 else 0.5
            for j, distance in enumerate(distances_set):
                values[i][j] = 1 if distance < threshold_variable else 0
            values[i] = normalize_list(values[i])
        return values

    def exponential_variable(values):
        for i, distances_set in enumerate(values):
            threshold_variable = values[i][0] * param
            threshold_variable = threshold_variable if threshold_variable > 0.5 else 0.5
            for j, distance in enumerate(distances_set):
                values[i][j] = np.exp(-distance / threshold_variable)
            values[i] = normalize_list(values[i])
        return values

    def linear_variable(values):
        for i, distances_set in enumerate(values):
            threshold_variable = values[i][0] * param
            threshold_variable = threshold_variable if threshold_variable > 0.5 else 0.5
            for j, distance in enumerate(distances_set):
                values[i][j] = 1 - distance / threshold_variable if distance < threshold_variable else 0
            values[i] = normalize_list(values[i])
        return values

    def cosine_variable(values):
        for i, distances_set in enumerate(values):
            threshold_variable = values[i][0] * param
            threshold_variable = threshold_variable if threshold_variable > 0.5 else 0.5
            for j, distance in enumerate(distances_set):
                values[i][j] = np.cos(
                    np.pi * distance / (2 * threshold_variable)) if distance < threshold_variable else 0
            values[i] = normalize_list(values[i])
        return values

    funcs = {
        'epanechnikov': epanechnikov,
        'gaussian': gaussian,
        'tophat': tophat,
        'exponential': exponential,
        'linear': linear,
        'cosine': cosine,
        'epanechnikov_variable': epanechnikov_variable,
        'gaussian_variable': gaussian_variable,
        'tophat_variable': tophat_variable,
        'exponential_variable': exponential_variable,
        'linear_variable': linear_variable,
        'cosine_variable': cosine_variable
    }
    return funcs[func_name]


def normalize_list(raw):
    norm = [float(i) / sum(raw) if sum(raw) != 0 else 1 / len(raw) for i in raw]
    return norm


def get_f1_loo_score_naive(regressor, x_set, y_set):
    loo = sk.model_selection.LeaveOneOut()
    y_predicted = []
    for train_index, test_index in loo.split(x_set):
        x_train, x_test = x_set[train_index], x_set[test_index]
        y_train, y_test = y_set[train_index], y_set[test_index]
        regressor.fit(x_train, y_train)
        y_predicted.append(int(regressor.predict(x_test) + 0.5))

    f1_result = f1_score(y_set.tolist(), y_predicted, average='macro')
    return f1_result


def get_f1_loo_score_one_hot(regressor, x_set, y_set):
    loo = sk.model_selection.LeaveOneOut()
    y_actual = []
    y_predicted = []
    for train_index, test_index in loo.split(x_set):
        x_train, x_test = x_set[train_index], x_set[test_index]
        y_train, y_test = y_set[train_index], y_set[test_index]
        regressor.fit(x_train, y_train)
        prediction = regressor.predict(x_test)
        predicted_class = max(range(len(prediction[0])), key=prediction[0].__getitem__)
        actual_class = max(range(len(y_test[0])), key=y_test[0].__getitem__)
        y_predicted.append(predicted_class)
        y_actual.append(actual_class)

    f1_result = f1_score(y_actual, y_predicted, average='macro')
    return f1_result


# SETUP:

weight_func_options = []
for func in ['epanechnikov', 'gaussian', 'tophat', 'exponential', 'linear', 'cosine']:
    for threshold in np.linspace(0.1, 1.5, 4):  # TODO: change second value
        weight_func_options.append(get_weight_func(func, threshold))

# for func in ['epanechnikov_variable', 'gaussian_variable', 'tophat_variable', 'exponential_variable', 'linear_variable',
#              'cosine_variable']:
#     for threshold in np.linspace(0.1, 1.5, 4):
#         weight_func_options.append(get_weight_func(func, threshold))

param_grid = {
    # 'n_neighbors': [3, 4, 5, 5, 6, 7, 8, 9, 10],
    'n_neighbors': [16],
    'weights': weight_func_options,
    'metric': ['minkowski', 'euclidean', 'chebyshev'],
    'p': [1, 2, 3]
}

data = pd.read_csv("dataset_42_soybean.csv")
data_original = data

# NAIVE:

# Vectorization

vectorization_map = {}

for attribute in data.columns:
    vectorization_map[attribute] = {}
    for idx, unique_value in enumerate(data[attribute].unique()):
        vectorization_map[attribute][unique_value] = idx

for attribute in data.columns:
    for idx, attr_value in enumerate(data[attribute]):
        data[attribute][idx] = vectorization_map[attribute][attr_value]

data_params = list(data.columns[:-1])

X = data[data_params].to_numpy()
Y = data['class'].to_numpy()

# Optimization

knn_regressor = KNeighborsRegressor()
grid_search_naive = GridSearchCV(estimator=knn_regressor, param_grid=param_grid)
grid_search_naive.fit(X, Y)
print(grid_search_naive.best_params_)

knn_regressor = KNeighborsRegressor(metric=grid_search_naive.best_params_['metric'],
                                    n_neighbors=grid_search_naive.best_params_['n_neighbors'],
                                    p=grid_search_naive.best_params_['p'],
                                    weights=grid_search_naive.best_params_['weights'])

kernel_family_naive = re.search(r">\..{5,22} at", f'{grid_search_naive.best_params_["weights"]}').group(0)[2:][:-3]

f1_score_naive = get_f1_loo_score_naive(knn_regressor, X, Y)

# ONE HOT:

# Vectorization

data_params = list(data.columns[:-1])
X = data[data_params]
Y = data['class']
x_one_hot = pd.get_dummies(X)
y_one_hot = pd.get_dummies(Y)
x_one_hot_scaled = StandardScaler().fit_transform(x_one_hot)
y_one_hot_scaled = StandardScaler().fit_transform(y_one_hot)

# Optimization

knn_regressor = KNeighborsRegressor()
grid_search_one_hot = GridSearchCV(estimator=knn_regressor, param_grid=param_grid)
grid_search_one_hot.fit(x_one_hot_scaled, y_one_hot_scaled)
print(grid_search_one_hot.best_params_)

knn_regressor = KNeighborsRegressor(metric=grid_search_one_hot.best_params_['metric'],
                                    n_neighbors=grid_search_one_hot.best_params_['n_neighbors'],
                                    p=grid_search_one_hot.best_params_['p'],
                                    weights=grid_search_one_hot.best_params_['weights'])

kernel_family_one_hot = re.search(r">\..{5,22} at", f'{grid_search_one_hot.best_params_["weights"]}').group(0)[2:][:-3]

f1_score_one_hot = get_f1_loo_score_one_hot(knn_regressor, x_one_hot_scaled, y_one_hot_scaled)

print(f'Scores\n    Naive: {f1_score_naive}\n    One Hot: {f1_score_one_hot}')

# Graph building

window_sizes = []
f1_scores_windows = []
neighbors_amounts = []
f1_scores_neighbors = []
if f1_score_naive > f1_score_one_hot:
    grid_search = grid_search_naive
    x_final = X
    y_final = Y
    best_algo = 'Naive'
    get_f1_loo_score = get_f1_loo_score_naive
    kernel_family = kernel_family_naive
else:
    grid_search = grid_search_one_hot
    x_final = x_one_hot_scaled
    y_final = y_one_hot_scaled
    best_algo = 'One Hot'
    get_f1_loo_score = get_f1_loo_score_one_hot
    kernel_family = kernel_family_one_hot


for window_size in np.linspace(0.1, 10, 9):
    window_sizes.append(window_size)
    knn_regressor = KNeighborsRegressor(metric=grid_search.best_params_['metric'],
                                        n_neighbors=grid_search.best_params_['n_neighbors'],
                                        p=grid_search.best_params_['p'],
                                        weights=get_weight_func(kernel_family, window_size))
    f1_scores_windows.append(get_f1_loo_score(knn_regressor, x_final, y_final))

for n_neighbors in range(6, 20):
    neighbors_amounts.append(n_neighbors)
    knn_regressor = KNeighborsRegressor(metric=grid_search.best_params_['metric'],
                                        n_neighbors=n_neighbors,
                                        p=grid_search.best_params_['p'],
                                        weights=grid_search.best_params_['weights'])
    f1_scores_neighbors.append(get_f1_loo_score(knn_regressor, x_final, y_final))

fig, (plt_1, plt_2) = plt.subplots(1, 2)
fig.canvas.set_window_title(f'F1-scores for {best_algo} vectorization')

plt_1.plot(window_sizes, f1_scores_windows)
plt_1.set_ylabel('F1 score')
plt_1.set_xlabel('Window size')

plt_2.plot(neighbors_amounts, f1_scores_neighbors)
plt_2.set_ylabel('F1 score')
plt_2.set_xlabel('Neighbors amount')

box = plt_2.get_position()
box.x0 = box.x0 + 0.1
box.x1 = box.x1 + 0.1
plt_2.set_position(box)

plt.show()
#
# knn_regressor = KNeighborsRegressor()
#     loo = sk.model_selection.LeaveOneOut()
#     y_predicted = []
#     y_actual = []
#     for train_index, test_index in loo.split(X):
#         x_train, x_test = x_one_hot_scaled[train_index], x_one_hot_scaled[test_index]
#         y_train, y_test = y_one_hot_scaled[train_index], y_one_hot_scaled[test_index]
#         knn_regressor.fit(x_train, y_train)
#         prediction = knn_regressor.predict(x_test)
#         predicted_class = np.where(prediction == np.amin(prediction))[0][0]
#         actual_class = np.where(y_test == np.amin(y_test))[0][0]
#         y_predicted.append(predicted_class)
#         y_actual.append(actual_class)
#
# print(f1_score(y_actual, y_predicted, average='macro'))
