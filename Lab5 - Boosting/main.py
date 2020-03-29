import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from matplotlib import animation

# plt.style.use('seaborn-pastel')
h = 0.02  # graph resolution
max_iterations = 100


# animation function
def animate(variable_args, x, y, f1_max, f1_min):
    z, f1 = variable_args
    ax[0].contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax[0].scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
    ax[1].axis(xmin=1, xmax=max_iterations - 1)
    ax[1].axis(ymin=f1_min, ymax=f1_max)
    ax[1].plot(range(1, len(f1) + 1), f1, color='black')
    return ax


for dataset_num, dataset in enumerate(['chips', 'geyser']):
    fig, ax = plt.subplots(2, 1)
    # Data processing:
    data = pd.read_csv(f"{dataset}.csv")
    classes = data[data.columns[-1]].unique()
    data[data.columns[-1]] = pd.factorize(data[data.columns[-1]])[0]
    X = data[data.columns[:-1]].to_numpy()
    Y = data[data.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    predictions = []
    # Create a mesh to plot in
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    plt.gcf().canvas.set_window_title('Lab 5 - Boosting')
    # plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm)
    ax[0].axis(xmin=xx.min(), xmax=xx.max())
    ax[0].axis(ymin=yy.min(), ymax=yy.max())
    # ax[0].xticks(())
    # ax[0].yticks(())
    # .title(f'Boosting on {dataset} dataset - up to {max_iterations} iterations')
    f1_scores = []
    f1_scores_current = []

    for iterations_amount in range(1, max_iterations):
        # Drawing:
        # plt.subplot()

        clf = AdaBoostClassifier(n_estimators=iterations_amount)
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        score_Predictions = clf.predict(X_test)
        f1_scores_current.append(f1_score(y_test, score_Predictions))
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        predictions.append((Z, f1_scores_current.copy()))
    anim = animation.FuncAnimation(fig, animate, frames=predictions,
                                   fargs=(X_test, y_test, max(f1_scores_current), min(f1_scores_current)))
    anim.save(f'{dataset}.gif')

