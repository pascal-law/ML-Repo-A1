import os
from autograd import grad
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import scipy.optimize
import scipy.special

def minimizeL2(X, y):
	return np.linalg.solve(X.T @ X, X.T @ y)

def minimizeLinf(X, y):

	d = np.size(X, 1)
	n = np.size(X, 0)

	c = np.concatenate((np.zeros((d,1)), np.ones((1,1))), axis=0)

	g11 = np.zeros((1, d))
	g12 = -np.ones((1, 1))
	g21 = X
	g22 = -np.ones((n, 1))
	g31 = -X
	g32 = -np.ones((n, 1))

	G = np.concatenate(
		(
		np.concatenate((g11, g12),axis=1),
		np.concatenate((g21, g22),axis=1),
		np.concatenate((g31, g32),axis=1)
		)
		, axis=0
	)

	h1 = np.zeros((1,1))
	h2 = y
	h3 = -y

	h = np.concatenate((h1, h2, h3),axis=0)

	c = matrix(c)
	G = matrix(G)
	h = matrix(h)

	solvers.options['show_progress'] = False
	sol = solvers.lp(c,G,h)
	u = np.array(sol['x'])
	w = u[:-1]
	# delta = u[-1]

	return w

def synRegExperiments():

	def genData(n_points, is_training=False):
		'''
		This function generate synthetic data
		'''
		X = np.random.randn(n_points, d) # input matrix
		X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
		y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label
		if is_training:
			y[0] *= -0.1
		return X, y

		# # following is debug code for sample data
		# if is_training:
		# 	path = "./toy_data/regression_train.csv"
		# else:
		# 	path = "./toy_data/regression_test.csv"

		# df = pd.read_csv(path)
		# X = df["x"].values.reshape(-1, 1)
		# X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
		# y = df["y"].values.reshape(-1, 1)

		# return X, y
	
	# if using actual data, change the n_train, n_test and d values
	n_runs = 100
	n_train = 30
	n_test = 1000
	d = 5
	noise = 0.2
	train_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
	test_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
	
	# TODO: Change the following random seed to one of your student IDs
	np.random.seed(101307254)

	for r in range(n_runs):

		w_true = np.random.randn(d + 1, 1)
		Xtrain, ytrain = genData(n_train, is_training=True)
		Xtest, ytest = genData(n_test, is_training=False)

		w_L2 = minimizeL2(Xtrain, ytrain)
		w_Linf = minimizeLinf(Xtrain, ytrain)

		# # testing w_L2 and w_Linf
		# print(w_L2)
		# print(w_Linf)

		# TODO: Evaluate the two models' performance (for each model,
		# calculate the L2 and L infinity losses on the training
		# data). Save them to `train_loss`
		L2Model_L2loss_train = 1 / (2 * n_train) * np.linalg.norm(Xtrain @ w_L2 - ytrain, 2)**2
		L2Model_Linfloss_train = np.linalg.norm(Xtrain @ w_L2 - ytrain, np.inf)
		LinfModel_L2loss_train = 1 / (2 * n_train) * np.linalg.norm(Xtrain @ w_Linf - ytrain, 2)**2
		LinfModel_Linfloss_train = np.linalg.norm(Xtrain @ w_Linf - ytrain, np.inf)
		
		train_loss[r] += np.array([
			[L2Model_L2loss_train, L2Model_Linfloss_train],
            [LinfModel_L2loss_train, LinfModel_Linfloss_train]])

		# TODO: Evaluate the two models' performance (for each model,
		# calculate the L2 and L infinity losses on the test
		# data). Save them to `test_loss`
		L2Model_L2loss_test = 1 / (2 * n_test) * np.linalg.norm(Xtest @ w_L2 - ytest, 2)**2
		L2Model_Linfloss_test = np.linalg.norm(Xtest @ w_L2 - ytest, np.inf)
		LinfModel_L2loss_test = 1 / (2 * n_test) * np.linalg.norm(Xtest @ w_Linf - ytest, 2)**2
		LinfModel_Linfloss_test = np.linalg.norm(Xtest @ w_Linf - ytest, np.inf)

		test_loss[r] += np.array([
			[L2Model_L2loss_test, L2Model_Linfloss_test],
            [LinfModel_L2loss_test, LinfModel_Linfloss_test]])

	# TODO: compute the average losses over runs
	accumulator_train_loss = np.zeros((2 , 2))
	accumulator_test_loss = np.zeros((2 , 2))
	for r in range(n_runs):
		accumulator_train_loss += train_loss[r]
		accumulator_test_loss += test_loss[r]

	# print(train_loss)
	# print("---")
	# print(test_loss)

	# TODO: return a 2-by-2 training loss variable and a 2-by-2 test loss variable

	# # testing 
	# print(accumulator_train_loss / n_runs)
	# print(accumulator_test_loss / n_runs)

	return accumulator_train_loss / n_runs, accumulator_test_loss / n_runs

# print(synRegExperiments())

def preprocessCCS(dataset_folder):
	path = os.path.join(dataset_folder, "Concrete_Data.xls")

	print(path)

	df = pd.read_excel(path)
	

	# cutting off the last column in df
	n = len(df)
	X = df.iloc[:, 0: -1].values.reshape(n, -1)
	y = df.iloc[:, -1].values.reshape(-1, 1)

	return X, y

# preprocessCCS(os.path.abspath("./toy_data"))

def runCCS(dataset_folder):
	# # test data
	X, y = preprocessCCS(dataset_folder)
	n, d = X.shape
	X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
	n_runs = 100
	n_train = n_test = int(n / 2)

	train_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
	test_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
	# TODO: Change the following random seed to one of your student IDs
	np.random.seed(101318299)
	gen = np.random.default_rng(101318299)

	# concatenate X, y together first
	concatenated_X_y = np.concatenate((X, y), axis=1)

	for r in range(n_runs):
		# TODO: Randomly partition the dataset into two parts (50%
		# training and 50% test)
		
		# # the following comments don't work because the X, y rows don't line up together
		# X_randomized = X
		# y_randomized = y
		# gen.shuffle(X_randomized)
		# gen.shuffle(y_randomized)
		# print(X_randomized)
		# print(y_randomized)

		X_y_randomized = concatenated_X_y
		gen.shuffle(X_y_randomized)
		X_y_train = X_y_randomized[:int(n / 2)]
		Xtrain = X_y_train[:,:-1]
		ytrain = X_y_train[:,-1].reshape(-1, 1) # after slicing with one single column left we have to reshape

		X_y_test = X_y_randomized[int(n / 2):]
		Xtest = X_y_test[:, :-1]
		ytest = X_y_test[:,-1].reshape(-1, 1) # after slicing with one single column left we have to reshape


		# TODO: Learn two different models from the training data
		# using L2 and L infinity losses
		w_L2 = minimizeL2(Xtrain, ytrain)
		w_Linf = minimizeLinf(Xtrain, ytrain)

		# TODO: Evaluate the two models' performance (for each model,
		# calculate the L2 and L infinity losses on the training
		# data). Save them to `train_loss`
		L2Model_L2loss_train = 1 / (2 * n_train) * np.linalg.norm(Xtrain @ w_L2 - ytrain, 2)**2
		L2Model_Linfloss_train = np.linalg.norm(Xtrain @ w_L2 - ytrain, np.inf)
		LinfModel_L2loss_train = 1 / (2 * n_train) * np.linalg.norm(Xtrain @ w_Linf - ytrain, 2)**2
		LinfModel_Linfloss_train = np.linalg.norm(Xtrain @ w_Linf - ytrain, np.inf)
		
		train_loss[r] += np.array([
			[L2Model_L2loss_train, L2Model_Linfloss_train],
            [LinfModel_L2loss_train, LinfModel_Linfloss_train]])

		# TODO: Evaluate the two models' performance (for each model,
		# calculate the L2 and L infinity losses on the test
		# data). Save them to `test_loss`
		L2Model_L2loss_test = 1 / (2 * n_test) * np.linalg.norm(Xtest @ w_L2 - ytest, 2)**2
		L2Model_Linfloss_test = np.linalg.norm(Xtest @ w_L2 - ytest, np.inf)
		LinfModel_L2loss_test = 1 / (2 * n_test) * np.linalg.norm(Xtest @ w_Linf - ytest, 2)**2
		LinfModel_Linfloss_test = np.linalg.norm(Xtest @ w_Linf - ytest, np.inf)

		test_loss[r] += np.array([
			[L2Model_L2loss_test, L2Model_Linfloss_test],
            [LinfModel_L2loss_test, LinfModel_Linfloss_test]])

	# TODO: compute the average losses over runs
	accumulator_train_loss = np.zeros((2 , 2))
	accumulator_test_loss = np.zeros((2 , 2))
	for r in range(n_runs):
		accumulator_train_loss += train_loss[r]
		accumulator_test_loss += test_loss[r]

	# TODO: return a 2-by-2 training loss variable and a 2-by-2 test loss variable
	return accumulator_train_loss / n_runs, accumulator_test_loss / n_runs

# print(runCCS(os.path.abspath("./toy_data")))

# Q2

def linearRegL2Obj(w, X, y):
	n, d = np.shape(X)
	obj_val = 1 / (2 * n) * np.linalg.norm(X @ w - y, 2)**2
	return obj_val

def linearRegL2Grad(w, X, y):
	n, d = np.shape(X)
	gradient = 1 / n * X.T @ (X @ w - y)
	return gradient

def find_opt(obj_func, grad_func, X, y):
	d = X.shape[1]
	# TODO: Initialize a random 1-D array of parameters of size d
	w_0 = np.random.randn(d)

	# TODO: Define an objective function `func` that takes a single argument (w)
	def func(w):
		return obj_func(w[:, None], X, y)
		
	# TODO: Define a gradient function `gd` that takes a single argument (w)
	def gd(w):
		return grad_func(w[:, None], X, y)
	
	print(np.shape(w_0))
	return scipy.optimize.minimize(func, w_0, jac=gd)['x'][:, None]

"""
n_points = 5
d = 100
noise = 0.2
w_true = np.random.randn(d + 1, 1)

X = np.random.randn(n_points, d) # input matrix
X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label


import autograd.numpy as autogradnp  # Thinly-wrapped numpy

def tanh(x):                 # Define a function
	return (1.0 - autogradnp.exp((-2 * x))) / (1.0 + autogradnp.exp(-(2 * x)))

grad_obj = grad(linearRegL2Obj)       # Obtain its gradient function
              # Evaluate the gradient at x = 1.0
print(grad_obj(w_true, X, y))
print((linearRegL2Obj(w_true + 0.00001, X, y) - linearRegL2Obj(w_true, X, y)) / 0.00001)  # Compare to finite differences
"""

# w = find_opt(linearRegL2Obj, linearRegL2Grad, X, y)

# print(w)

def logisticRegObj(w, X, y):
	n, d = np.shape(X)
	obj_val = 1 / n * (-y.T @ np.log(scipy.special.expit(X @ w)) - (np.ones((n, 1)) - y).T @ np.log(np.ones((n, 1)) - scipy.special.expit(X @ w)))
	print(obj_val)
	return obj_val

def logisticRegGrad(w, X, y):
	n, d = np.shape(X)
	gradient = 1 / n * X.T @ (scipy.special.expit(X @ w) - y)
	return gradient


