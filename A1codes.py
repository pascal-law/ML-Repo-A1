import os
from autograd import grad
import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import scipy.optimize
import scipy.special

# This function takes in an X (n x d) matrix and a y (n x 1) matrix, and returns the L2 loss
def minimizeL2(X, y):
	return np.linalg.solve(X.T @ X, X.T @ y)

# This function takes in an X (n x d) matrix and a y (n x 1) matrix, and returns the Linf loss
def minimizeLinf(X, y):

	d = np.size(X, 1)
	n = np.size(X, 0)

	c = np.concatenate((np.zeros((d,1)), np.ones((1,1))), axis=0)

	# each of the following variables corresponds to each g matrix that was used in the
	# appendix of the Linear Regression lecture (slides 23-25)
	g11 = np.zeros((1, d))
	g12 = -np.ones((1, 1))
	g21 = X
	g22 = -np.ones((n, 1))
	g31 = -X
	g32 = -np.ones((n, 1))

	G = np.concatenate((
		np.concatenate((g11, g12),axis=1),
		np.concatenate((g21, g22),axis=1),
		np.concatenate((g31, g32),axis=1)
		), axis=0)

	# each of these h also corresponds with the appropriate h from the same lecture
	# as above (slides 23-25)
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

		# Each of these variables corresponds to each model/loss pairing
		# as specified in the assignment document
		L2Model_L2loss_train = 1 / (2 * n_train) * np.linalg.norm(Xtrain @ w_L2 - ytrain, 2)**2
		L2Model_Linfloss_train = np.linalg.norm(Xtrain @ w_L2 - ytrain, np.inf)
		LinfModel_L2loss_train = 1 / (2 * n_train) * np.linalg.norm(Xtrain @ w_Linf - ytrain, 2)**2
		LinfModel_Linfloss_train = np.linalg.norm(Xtrain @ w_Linf - ytrain, np.inf)
		
		# The pairings are tallied in train_loss
		train_loss[r] += np.array([
			[L2Model_L2loss_train, L2Model_Linfloss_train],
            [LinfModel_L2loss_train, LinfModel_Linfloss_train]])

		# TODO: Evaluate the two models' performance (for each model,
		# calculate the L2 and L infinity losses on the test
		# data). Save them to `test_loss`

		# Each of these variables corresponds to each model/loss pairing
		# as specified in the assignment document
		L2Model_L2loss_test = 1 / (2 * n_test) * np.linalg.norm(Xtest @ w_L2 - ytest, 2)**2
		L2Model_Linfloss_test = np.linalg.norm(Xtest @ w_L2 - ytest, np.inf)
		LinfModel_L2loss_test = 1 / (2 * n_test) * np.linalg.norm(Xtest @ w_Linf - ytest, 2)**2
		LinfModel_Linfloss_test = np.linalg.norm(Xtest @ w_Linf - ytest, np.inf)

		# The pairings are tallied in test_loss
		test_loss[r] += np.array([
			[L2Model_L2loss_test, L2Model_Linfloss_test],
            [LinfModel_L2loss_test, LinfModel_Linfloss_test]])

	# TODO: compute the average losses over runs

	# Two matrices of size 2x2 are initialized with the intention to hold
	# the averages of each model/loss pairing
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

	# the averages are not held in a variable but instead returned directly
	return accumulator_train_loss / n_runs, accumulator_test_loss / n_runs

# print(synRegExperiments())

# This function follows the assignment specification, taking in an absolute path
# wherein the Concrete_Data.xls file is located. It appropriately formats
# and returns an X and a y that work with the previously defined functions
def preprocessCCS(dataset_folder):
	path = os.path.join(dataset_folder, "Concrete_Data.xls")

	df = pd.read_excel(path)
	

	# cutting off the last column in df
	n = len(df)
	X = df.iloc[:, 0: -1].values.reshape(n, -1) # We form our X as an (n x d) matrix
	y = df.iloc[:, -1].values.reshape(-1, 1) # We form our y as a (n x 1) matrix

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

	# in order to maintain the X, y pairings, we concatenate the given matrices
	# so that we may randomize them correctly

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

		# We place the concatenated matrix into another variable so that we do not modify
		# the original concatenated matrix
		X_y_randomized = concatenated_X_y
		gen.shuffle(X_y_randomized)

		# We split the entire matrix into two parts, making sure to typecast (n/2)
		# as an integer

		X_y_train = X_y_randomized[:int(n / 2)]
		Xtrain = X_y_train[:,:-1] # this omits the last line (which is y) from the X training data
		ytrain = X_y_train[:,-1].reshape(-1, 1) # after slicing with one single column left we have to reshape

		X_y_test = X_y_randomized[int(n / 2):]
		Xtest = X_y_test[:, :-1] # this omits the last line (which is y) from the X test data
		ytest = X_y_test[:,-1].reshape(-1, 1) # after slicing with one single column left we have to reshape


		# TODO: Learn two different models from the training data
		# using L2 and L infinity losses
		w_L2 = minimizeL2(Xtrain, ytrain)
		w_Linf = minimizeLinf(Xtrain, ytrain)

		# TODO: Evaluate the two models' performance (for each model,
		# calculate the L2 and L infinity losses on the training
		# data). Save them to `train_loss`
		
		# Each of these variables corresponds to each model/loss pairing
		# as specified in the assignment document
		L2Model_L2loss_train = 1 / (2 * n_train) * np.linalg.norm(Xtrain @ w_L2 - ytrain, 2)**2
		L2Model_Linfloss_train = np.linalg.norm(Xtrain @ w_L2 - ytrain, np.inf)
		LinfModel_L2loss_train = 1 / (2 * n_train) * np.linalg.norm(Xtrain @ w_Linf - ytrain, 2)**2
		LinfModel_Linfloss_train = np.linalg.norm(Xtrain @ w_Linf - ytrain, np.inf)
		
		# The pairings are tallied in train_loss
		train_loss[r] += np.array([
			[L2Model_L2loss_train, L2Model_Linfloss_train],
            [LinfModel_L2loss_train, LinfModel_Linfloss_train]])

		# TODO: Evaluate the two models' performance (for each model,
		# calculate the L2 and L infinity losses on the test
		# data). Save them to `test_loss`

		# Each of these variables corresponds to each model/loss pairing
		# as specified in the assignment document
		L2Model_L2loss_test = 1 / (2 * n_test) * np.linalg.norm(Xtest @ w_L2 - ytest, 2)**2
		L2Model_Linfloss_test = np.linalg.norm(Xtest @ w_L2 - ytest, np.inf)
		LinfModel_L2loss_test = 1 / (2 * n_test) * np.linalg.norm(Xtest @ w_Linf - ytest, 2)**2
		LinfModel_Linfloss_test = np.linalg.norm(Xtest @ w_Linf - ytest, np.inf)

		# The pairings are tallied in test_loss
		test_loss[r] += np.array([
			[L2Model_L2loss_test, L2Model_Linfloss_test],
            [LinfModel_L2loss_test, LinfModel_Linfloss_test]])

	# TODO: compute the average losses over runs

	# Two matrices of size 2x2 are initialized with the intention to hold
	# the averages of each model/loss pairing
	accumulator_train_loss = np.zeros((2 , 2))
	accumulator_test_loss = np.zeros((2 , 2))
	for r in range(n_runs):
		accumulator_train_loss += train_loss[r]
		accumulator_test_loss += test_loss[r]

	# TODO: return a 2-by-2 training loss variable and a 2-by-2 test loss variable

	# the averages are not held in a variable but instead returned directly
	return accumulator_train_loss / n_runs, accumulator_test_loss / n_runs

# print(runCCS(os.path.abspath("./toy_data")))

"""
Q2
"""

# This function finds the objective value of the given w, X, and y matrices
def linearRegL2Obj(w, X, y):
	n, d = np.shape(X)
	obj_val = 1 / (2 * n) * np.linalg.norm(X @ w - y, 2)**2
	return obj_val

# This function finds the analytic form gradient (size d x 1) when provided
# with appropriate w, X, and y matrices
def linearRegL2Grad(w, X, y):
	n, d = np.shape(X)
	gradient = 1 / n * X.T @ (X @ w - y)
	return gradient

# find optimal w for a convex optimization problem
def find_opt(obj_func, grad_func, X, y):
	d = X.shape[1]
	# TODO: Initialize a random 1-D array of parameters of size d
	w_0 = np.zeros(d) # changed to 1-D array of zeros for better use in the BCW database

	# TODO: Define an objective function `func` that takes a single argument (w)
	def func(w):
		return obj_func(w[:, None], X, y)
		
	# TODO: Define a gradient function `gd` that takes a single argument (w)
	def gd(w):
		return grad_func(w[:, None], X, y).flatten() # added flatten()
	
	# print(np.shape(w_0))
	return scipy.optimize.minimize(func, w_0, jac=gd)['x'][:, None]

""" TESTING
# n_points = 5
# d = 100
# noise = 0.2
# w_true = np.random.randn(d + 1, 1)

# X = np.random.randn(n_points, d) # input matrix
# X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
# y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label


# import autograd.numpy as autogradnp  # Thinly-wrapped numpy

# def tanh(x):                 # Define a function
# 	return (1.0 - autogradnp.exp((-2 * x))) / (1.0 + autogradnp.exp(-(2 * x)))

# grad_obj = grad(linearRegL2Obj)       # Obtain its gradient function
#               # Evaluate the gradient at x = 1.0
# print(grad_obj(w_true, X, y))
# print((linearRegL2Obj(w_true + 0.00001, X, y) - linearRegL2Obj(w_true, X, y)) / 0.00001)  # Compare to finite differences


w = find_opt(linearRegL2Obj, linearRegL2Grad, X, y)

print(w)
"""
# This function returns the objective value (similar to linearRegL2Obj),
# but instead utilizes logistic regression
def logisticRegObj(w, X, y):
	n, d = np.shape(X)
	obj_val = 1 / n * (-y.T @ np.log(scipy.special.expit(X @ w)) - (np.ones((n, 1)) - y).T @ np.log(np.ones((n, 1)) - scipy.special.expit(X @ w)))
	return obj_val

# This function returns the analytic form gradient (similar to linearRegL2Grad),
# but instead utilizes logistic regression
def logisticRegGrad(w, X, y):
	n, d = np.shape(X)
	gradient = 1 / n * X.T @ (scipy.special.expit(X @ w) - y)
	return gradient

# returns a 4 × 3 matrix train acc of average training accuracies and 
# a 4 × 3 matrix test acc of average test accuracies over 100 runs
def synClsExperiments():
	def genData(n_points, dim1, dim2):
		'''
		This function generate synthetic data
		'''
		c0 = np.ones([1, dim1]) # class 0 center
		c1 = -np.ones([1, dim1]) # class 1 center
		X0 = np.random.randn(n_points, dim1 + dim2) # class 0 input
		X0[:, :dim1] += c0
		X1 = np.random.randn(n_points, dim1 + dim2) # class 1 input
		X1[:, :dim1] += c1
		X = np.concatenate((X0, X1), axis=0)
		X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1) # augmentation
		y = np.concatenate([np.zeros([n_points, 1]), np.ones([n_points, 1])], axis=0)
		return X, y
		
	def runClsExp(m=100, dim1=2, dim2=2):
		'''
		Run classification experiment with the specified arguments
		'''
		n_test = 1000
		Xtrain, ytrain = genData(m, dim1, dim2)
		Xtest, ytest = genData(n_test, dim1, dim2)
		w_logit = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

		# TODO: Compute predicted labels of the training points
		ztrain_hat = Xtrain @ w_logit # let this linear prediction boundary = 0
		ytrain_hat = (ztrain_hat >= 0).astype(int) # LogReg Slides page 5/18
		# TODO: Compute the accuarcy of the training set
		train_acc = 1 - np.average(np.abs(ytrain_hat - ytrain)) # we check element-wise if all given predictions are correct
		# we find accuracy by 1 - misclassification rate

		# TODO: Compute predicted labels of the test points
		ztest_hat = Xtest @ w_logit # let this linear prediction boundary = 0
		ytest_hat = (ztest_hat >= 0).astype(int) # LogReg Slides page 5/18
		# TODO: Compute the accuarcy of the test set
		test_acc = 1 - np.average(np.abs(ytest_hat - ytest)) # we check element-wise if all given predictions are correct
		# we find accuracy by 1 - misclassification rate

		return train_acc, test_acc
	
	n_runs = 100
	train_acc = np.zeros([n_runs, 4, 3])
	test_acc = np.zeros([n_runs, 4, 3])
	# TODO: Change the following random seed to one of your student IDs
	np.random.seed(101318299)
	for r in range(n_runs):
		# print("repeat")
		for i, m in enumerate((10, 50, 100, 200)):
			train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
			# print("1", r, i, m)
		for i, dim1 in enumerate((1, 2, 4, 8)):
			train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(dim1=dim1)
			# print("2", r, i, dim1)
		for i, dim2 in enumerate((1, 2, 4, 8)):
			train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(dim2=dim2)
			# print("3", r, i, dim2)
	# TODO: compute the average accuracies over runs

	
	# Two matrices of size 4x3 are initialized with the intention to hold
	# the averages of each accuracy for each hyper-parameter change
	accumulator_train_acc = np.zeros((4, 3))
	accumulator_test_acc = np.zeros((4, 3))
	for r in range(n_runs):
		accumulator_train_acc += train_acc[r]
		accumulator_test_acc += test_acc[r]

	# TODO: return a 4-by-3 training accuracy variable and a 4-by-3 test accuracy variable

	# the averages are not held in a variable but instead returned directly
	return accumulator_train_acc / n_runs, accumulator_test_acc / n_runs

# print(synClsExperiments())

# This function follows the assignment specification, taking in an absolute path
# wherein the Concrete_Data.xls file is located. It appropriately formats
# and returns an X and a y that work with the previously defined functions
def preprocessBCW(dataset_folder):
	path = os.path.join(dataset_folder, "wdbc.data")
	df = pd.read_csv(path, header=None, sep=',') # it's a semi-csv file so we could just read it using pandas
	n = len(df)
	
	y = df.iloc[:, 1].values.reshape(n, -1) # get index 1 column as y
	y = (y == "M").astype(int) # converting M, B -> 1, 0
	X = df.iloc[:, [2,26,27]].values.reshape(n, -1) # gets index [2, 26, 27] columns as X
	return X, y

# preprocessBCW(os.path.abspath("./toy_data"))

def runBCW(dataset_folder):

	X, y = preprocessBCW(dataset_folder)
	n, d = X.shape
	X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment

	n_runs = 100
	train_acc = np.zeros([n_runs])
	test_acc = np.zeros([n_runs])

	# TODO: Change the following random seed to one of your student IDs
	np.random.seed(101307254)
	gen = np.random.default_rng(101307254)

	# in order to maintain the X, y pairings, we concatenate the given matrices
	# so that we may randomize them correctly

	# concatenate X, y together first
	concatenated_X_y = np.concatenate((X, y), axis=1)

	for r in range(n_runs):
		# TODO: Randomly partition the dataset into two parts (50%
		# training and 50% test)

		X_y_randomized = concatenated_X_y # each run, randomize based on the original data
		gen.shuffle(X_y_randomized)

		X_y_train = X_y_randomized[:int(n / 2)]
		Xtrain = X_y_train[:,:-1]
		ytrain = X_y_train[:,-1].reshape(-1, 1) # after slicing with one single column left we have to reshape

		X_y_test = X_y_randomized[int(n / 2):]
		Xtest = X_y_test[:,:-1]
		ytest = X_y_test[:,-1].reshape(-1, 1) # after slicing with one single column left we have to reshape

		w = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

		# TODO: Evaluate the model's accuracy on the training
		# data. Save it to `train_acc`

		ztrain_hat = Xtrain @ w # let this linear prediction boundary = 0
		ytrain_hat = (ztrain_hat >= 0).astype(int) # LogReg Slides page 5/18
		train_acc[r] = 1 - np.average(np.abs(ytrain_hat - ytrain))  # we check element-wise if all given predictions are correct
		# we find accuracy by 1 - misclassification rate

		# TODO: Evaluate the model's accuracy on the test
		# data. Save it to `test_acc`
		ztest_hat = Xtest @ w # let this linear prediction boundary = 0
		ytest_hat = (ztest_hat >= 0).astype(int) # LogReg Slides page 5/18
		test_acc[r] = 1 - np.average(np.abs(ytest_hat - ytest))  # we check element-wise if all given predictions are correct
		# we find accuracy by 1 - misclassification rate

	# TODO: compute the average accuracies over runs
	# TODO: return two variables: the average training accuracy and average test accuracy

	return np.mean(train_acc), np.mean(test_acc)

# print(runBCW("./toy_data"))
