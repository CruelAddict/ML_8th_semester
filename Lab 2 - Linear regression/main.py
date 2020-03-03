import numpy as np
from sklearn.model_selection import GridSearchCV
from skimage.metrics import normalized_root_mse
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


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

ridge_param_grid = {
    'alpha': [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0],
    'max_iter': [10, 50, 100, 1000],
    'solver': ['sparse_cg']
}
lasso_param_grid = {
    'alpha': [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0, 100000000.0, 1000000000.0],
    'max_iter': [10, 50, 100, 1000, 10000]
}

ridge_grid_search = GridSearchCV(Ridge(), ridge_param_grid)
lasso_grid_search = GridSearchCV(Lasso(), lasso_param_grid)

ridge_grid_search.fit(train_X, train_Y)
lasso_grid_search.fit(train_X, train_Y)

ridge_regressor = Ridge(**ridge_grid_search.best_params_).fit(train_X, train_Y)
lasso_regressor = Lasso(**lasso_grid_search.best_params_).fit(train_X, train_Y)

nrmse_ridge_train = normalized_root_mse(np.array(train_Y), np.array(ridge_regressor.predict(train_X)))
nrmse_ridge_test = normalized_root_mse(np.array(test_Y), np.array(ridge_regressor.predict(test_X)))

print(f'Ridge regressor:\n'
      f'    NRMSE on train: {nrmse_ridge_train}\n'
      f'    NRMSE on test: {nrmse_ridge_test}')

nrmse_lasso_train = normalized_root_mse(np.array(train_Y), np.array(lasso_regressor.predict(train_X)))
nrmse_lasso_test = normalized_root_mse(np.array(test_Y), np.array(lasso_regressor.predict(test_X)))

print(f'Lasso regressor:\n'
      f'    NRMSE on train: {nrmse_lasso_train}\n'
      f'    NRMSE on test: {nrmse_lasso_test}')

if (nrmse_ridge_train + nrmse_ridge_test) / 2 < (nrmse_lasso_train + nrmse_lasso_test) / 2:
    least_squares_regressor = Ridge
    least_squares_params = ridge_grid_search.best_params_
else:
    least_squares_regressor = Lasso
    least_squares_params = lasso_grid_search.best_params_

'''
    Gradient descent 
                     '''

# Best result (q_min ~0,05) with 300 iterations & learning rate 3000000.
iterations_amount = 10
learning_rate = 7000000

m = np.ones(features_amount)
b = 0

print(
    f'\nCalculating the best parameters using gradient descent with {iterations_amount} iterations and '
    f'learning rate = {learning_rate}. \nNRMSE values for each iteration:')

m, b = update_weights(m, b, train_X, train_Y, learning_rate)
m_min = m.copy()
b_min = b

q_min = normalized_root_mse(np.array(train_Y), np.array(get_predictions(m, b, train_X)))
print(f'    {1}. {q_min}')

for i in range(1, iterations_amount):
    m, b = update_weights(m, b, train_X, train_Y, learning_rate)
    q_current = normalized_root_mse(np.array(train_Y), np.array(get_predictions(m, b, train_X)))
    print(f'    {i + 1}. {q_current}')
    if q_current < q_min:
        q_min = q_current
        m_min = m.copy()
        b_min = b

print(
    f'The best NRMSE result is {round(q_min, 4)} for train set, '
    f'{round(normalized_root_mse(np.array(test_Y), np.array(get_predictions(m, b, test_X))), 4) } for test set.')

'''
    Genetic algorithm
                        '''

inds_per_pop = 300  # individuals per population
num_generations = 10
num_parents_mating = 100
crossover_points = 3
number_of_mutations = 500
population = np.random.uniform(low=-1, high=1, size=(inds_per_pop, features_amount + 1))  # last item is b

for generation in range(num_generations):
    inds_fitnesses = [
        normalized_root_mse(
            np.array(train_Y),
            np.array(get_predictions(individual[:-1], individual[-1:][0], train_X))
        )
        for individual in population]
    mating_pool = [individual for fitness, individual in sorted(zip(inds_fitnesses, population),  key=lambda x: x[0])][:num_parents_mating]
    print(f'Best NRMSE for generation {generation}: {np.min(inds_fitnesses)}')
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
