import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, make_scorer

h = 0.02  # graph resolution

default_param_grid = {
    'C': [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    'gamma': ['auto', 'scale', 0.01, 0.1, 0.5]
}

poly_param_grid = {
    'C': [0.1, 1.0, 10.0],
    'degree': [1, 2, 3, 4, 5, 6]
}

for dataset_num, dataset in enumerate(['chips', 'geyser']):
    # Data processing:
    data = pd.read_csv(f"{dataset}.csv")
    print(f'Items of class N in the dataset: {len([result for result in data[data.columns[-1]] if result == "N"])}')
    print(f'Items of class P in the dataset: {len([result for result in data[data.columns[-1]] if result == "P"])}')
    classes = data[data.columns[-1]].unique()
    data[data.columns[-1]] = pd.factorize(data[data.columns[-1]])[0]
    X = data[data.columns[:-1]].to_numpy()
    Y = data[data.columns[-1]]

    for i, kernel in enumerate(['rbf', 'linear', 'poly', 'sigmoid']):
        # Calculating the best kernel parameters:
        if kernel == 'poly':
            current_param_grid = poly_param_grid.copy()
        else:
            current_param_grid = default_param_grid.copy()
        current_param_grid['kernel'] = [kernel]
        if kernel == 'sigmoid':
            current_param_grid['coef0'] = [0.0, 0.1, 0.5, 1, 5, 10]
            current_param_grid['C'] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        grid_search = GridSearchCV(estimator=SVC(), param_grid=current_param_grid,
                                   scoring=make_scorer(f1_score, average='micro')).fit(X, Y)
        svc_best = SVC(**grid_search.best_params_).fit(X, Y)

        # Create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # Drawing:
        plt.subplot(4, 2, i + 1 + 4 * dataset_num)
        Z = svc_best.predict(np.c_[xx.ravel(), yy.ravel()])
        print(
            f'{kernel} kernel predictions: {round((len([result for result in Z if result == 1]) / len(Z)) * 100, 0)}% '
            f'of class {classes[1]}, '
            f'{round((len([result for result in Z if result == 0]) / len(Z)) * 100, 0)}% '
            f'of class {classes[0]}')
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(f'{i + 1 + 4 * dataset_num}. {kernel.upper()} kernel on {dataset} dataset')
        print(f'{kernel.upper()} kernel for {dataset} dataset calculated; best params: {grid_search.best_params_}')

plt.gcf().canvas.set_window_title('Lab 3 - SVM')
plt.show()
