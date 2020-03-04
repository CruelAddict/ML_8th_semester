import numpy as np
from sklearn.model_selection import GridSearchCV
from skimage.metrics import normalized_root_mse
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def update_weights(m, b, x, y, learning_rate):
    m_deriv_sum = np.zeros(len(m))
    b_deriv_sum = 0
    squares = 0
    y_range = np.max(y) - np.min(y)
    n = len(x)
    for t in range(n):
        hypothesis = (np.dot(m, x[t]) + b)
        for j in range(len(m)):
            m_deriv_sum[j] += -2 * x[t][j] * (y[t] - hypothesis)
        b_deriv_sum += -2 * (y[t] - hypothesis)
        squares += (y[t] - hypothesis) ** 2
    for j in range(len(m)):
        m[j] -= (m_deriv_sum[j] / (2 * np.sqrt(squares * float(n)) * y_range)) * learning_rate

    b -= (b_deriv_sum / (2 * np.sqrt(squares * float(n)) * y_range)) * learning_rate

    return m, b


def get_predictions(m, b, x):
    predictions = []
    for x_obj in x:
        predictions.append(np.dot(m, x_obj) + b)
    return predictions


def mate(parent_1, parent_2, cross_points):
    if parent_1.shape == parent_2.shape:
        parents = [parent_1, parent_2]
        n = len(parent_1)
        cross_div, cross_mod = np.divmod(n, cross_points)
        child_genes = []
        for i in range(cross_points):
            parent_idx = int(np.divmod(i, 2)[1])
            transmitted_genes_start = i * cross_div + min(i, cross_mod)
            transmitted_genes_end = (i + 1) * cross_div + min(i + 1, cross_mod)
            child_genes.append(parents[parent_idx][transmitted_genes_start: transmitted_genes_end])
        child = []
        for gene_set in child_genes:
            gs = np.reshape(gene_set, -1).tolist()
            child += gs
        return child
    else:
        raise ValueError("Shapes don't match")


data_set = 2

f = open(f"Linear/{data_set}.txt", "r")
# f.readline()
features_amount = int(f.readline())
train_objects_amount = int(f.readline())
train_X = []
train_Y = []
for i in range(0, train_objects_amount):
    obj_line = f.readline().split(" ")
    for idx, obj in enumerate(obj_line):
        obj_line[idx] = int(obj)
    train_Y.append(obj_line.pop())
    train_X.append(np.array(obj_line))
test_objects_amount = int(f.readline())
test_X = []
test_Y = []
for i in range(0, test_objects_amount):
    obj_line = f.readline().split(" ")
    for idx, obj in enumerate(obj_line):
        obj_line[idx] = int(obj)
    test_Y.append(obj_line.pop())
    test_X.append(np.array(obj_line))

'''
    Least squares:
                     '''

print('\n*** Least Squares ***\n')

ridge_param_grid = {
    'alpha': [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0],
    'solver': ['svd']
}
ridge_grid_search = GridSearchCV(Ridge(), ridge_param_grid)

print('Looking for the best Ridge regularization strength...')
ridge_grid_search.fit(train_X, train_Y)
print(f'The best reg. strength is {ridge_grid_search.best_params_["alpha"]}\n')

ridge_regressor = Ridge(**ridge_grid_search.best_params_).fit(train_X, train_Y)

nrmse_ridge_train = normalized_root_mse(np.array(train_Y), np.array(ridge_regressor.predict(train_X)))
nrmse_ridge_test = normalized_root_mse(np.array(test_Y), np.array(ridge_regressor.predict(test_X)))

print(f'Ridge regressor:\n'
      f'    NRMSE on train: {nrmse_ridge_train}\n'
      f'    NRMSE on test: {nrmse_ridge_test}')

'''
    Gradient descent 
                     '''

print('\n*** Gradient descent ***\n')

# Best result (q_min ~0,05) with 300 iterations & learning rate 3000000.
iterations_amount = 40
learning_rate = 4000000

m = np.ones(features_amount)
b = 0

print(
    f'Calculating the best parameters using gradient descent with {iterations_amount} iterations and '
    f'learning rate = {learning_rate}. \nNRMSE values for each iteration:')

m, b = update_weights(m, b, train_X, train_Y, learning_rate)
m_min = m.copy()
b_min = b

q_min = normalized_root_mse(np.array(train_Y), np.array(get_predictions(m, b, train_X)))
print(f'    {1}. {q_min}')

descent_results_train = []
descent_results_test = []
for i in range(1, iterations_amount):
    m, b = update_weights(m, b, train_X, train_Y, learning_rate)
    q_current = normalized_root_mse(np.array(train_Y), np.array(get_predictions(m, b, train_X)))
    print(f'    {i + 1}. {q_current}')
    descent_results_train.append(q_current)
    q_test = normalized_root_mse(np.array(test_Y), np.array(get_predictions(m, b, test_X)))
    descent_results_test.append(q_test)
    if q_current < q_min:
        q_min = q_current
        m_min = m.copy()
        b_min = b

print(
    f'\nThe best NRMSE result is {round(q_min, 4)} for train set, '
    f'{round(normalized_root_mse(np.array(test_Y), np.array(get_predictions(m, b, test_X))), 4)} for test set.')

'''
    Genetic algorithm
                        '''

print('\n*** Genetic Algorithm ***\n')

inds_per_pop = 300  # individuals per population
num_generations = 100
num_parents_mating = 100
crossover_points = 3
number_of_mutations = 3500
population = np.random.uniform(low=-1, high=1, size=(inds_per_pop, features_amount + 1))  # last item is b

gen_alg_results_train = []
gen_alg_results_test = []

for generation in range(num_generations-1):
    inds_fitnesses = [
        normalized_root_mse(
            np.array(train_Y),
            np.array(get_predictions(individual[:-1], individual[-1:][0], train_X))
        )
        for individual in population]
    mating_pool = [individual for fitness, individual in sorted(zip(inds_fitnesses, population), key=lambda x: x[0])][
                  :num_parents_mating]
    best_on_train = np.min(inds_fitnesses)
    best_on_test = normalized_root_mse(
        np.array(test_Y),
        np.array(get_predictions(mating_pool[0][:-1], mating_pool[0][-1:][0], test_X))
    )
    gen_alg_results_train.append(best_on_train)
    gen_alg_results_test.append(best_on_test)
    print(f'Best NRMSE for generation #{generation + 1}: {best_on_train} on train, {best_on_test} on test')
    # Crossover:
    for index in range(inds_per_pop - num_parents_mating):
        children = mate(mating_pool[np.remainder(index, num_parents_mating)],
                        mating_pool[np.random.randint(num_parents_mating)],
                        crossover_points)
        population[num_parents_mating + index] = children
    # Mutation
    for i in range(number_of_mutations):
        population[np.random.randint(inds_per_pop)][np.random.randint(features_amount)] *= ((-1) ** np.random.randint(
            2)) * np.random.random_sample() * 2
        population[np.random.randint(inds_per_pop)][np.random.randint(features_amount)] += ((-1) ** np.random.randint(
            2)) * np.random.random_sample() * 1

fig, plts = plt.subplots(2, 2, figsize=(7, 7))
plt.tight_layout()
fig.canvas.set_window_title(f'NRMSE scores for different amount of iterations')

plts[0, 0].plot(range(1, iterations_amount), descent_results_train)
plts[0, 0].set_ylabel('NRMSE for descent on train')
plts[0, 0].set_xlabel('Iterations')

plts[0, 1].plot(range(1, iterations_amount), descent_results_test)
plts[0, 1].set_ylabel('NRMSE for descent on test')
plts[0, 1].set_xlabel('Iterations')

plts[1, 0].plot(range(1, num_generations), gen_alg_results_train)
plts[1, 0].set_ylabel('NRMSE for genetic algorithm on train')
plts[1, 0].set_xlabel('Iterations')

plts[1, 1].plot(range(1, num_generations), gen_alg_results_test)
plts[1, 1].set_ylabel('NRMSE for genetic algorithm on test')
plts[1, 1].set_xlabel('Iterations')

# box = plts[0, 1].get_position()
# box.x0 = box.x0 + 0.2
# box.x1 = box.x1 + 0.2
# plts[0, 1].set_position(box)
#
# box = plts[1, 1].get_position()
# box.x0 = box.x0 + 0.2
# box.x1 = box.x1 + 0.2
# plts[1, 1].set_position(box)
#
# box = plts[1, 0].get_position()
# box.y0 = box.y0 - 0.1
# box.y1 = box.y1 - 0.1
# plts[1, 0].set_position(box)
#
# box = plts[1, 1].get_position()
# box.y0 = box.y0 - 0.1
# box.y1 = box.y1 - 0.1
# plts[1, 1].set_position(box)

plt.show()
