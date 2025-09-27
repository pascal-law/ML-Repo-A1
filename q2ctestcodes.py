from autograd import grad
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special


# Q2c Logistic Regression toy_data_classification .csv file testings


# find optimal w for a convex optimization problem
def find_opt(obj_func, grad_func, X, y):
	d = X.shape[1]
	# TODO: Initialize a random 1-D array of parameters of size d
	w_0 = np.random.randn(d)

	# TODO: Define an objective function `func` that takes a single argument (w)
	def func(w):
		return obj_func(w[:, None], X, y)
		
	# TODO: Define a gradient function `gd` that takes a single argument (w)
	def gd(w):
		return grad_func(w[:, None], X, y).flatten() # added flatten()
	
	# print(np.shape(w_0))
	return scipy.optimize.minimize(func, w_0, jac=gd)['x'][:, None]



def logisticRegObj(w, X, y):
	n, d = np.shape(X)
	obj_val = 1 / n * (-y.T @ np.log(scipy.special.expit(X @ w)) - (np.ones((n, 1)) - y).T @ np.log(np.ones((n, 1)) - scipy.special.expit(X @ w)))
	return obj_val

def logisticRegGrad(w, X, y):
	n, d = np.shape(X)
	gradient = 1 / n * X.T @ (scipy.special.expit(X @ w) - y)
	return gradient

# returns a 4 × 3 matrix train acc of average training accuracies and 
# a 4 × 3 matrix test acc of average test accuracies over 100 runs
def synClsExperiments():
	def genData(n_points, dim, is_training):
		# following is debug code for sample data
		if is_training:
			path = "./toy_data/classification_train.csv"
		else:
			path = "./toy_data/classification_test.csv"
		df = pd.read_csv(path)
		X1 = df["x1"].values.reshape(n_points, 1)
		X2 = df["x2"].values.reshape(n_points, 1)
		X = np.concatenate((X1, X2), axis=1)
		X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augmentation

		y = df["y"].values.reshape(n_points, 1)

		return X, y

	def runClsExp(m, dim):
		'''
		Run classification experiment with the specified arguments
		'''
		Xtrain, ytrain = genData(m, dim, is_training=True)
		Xtest, ytest = genData(m, dim, is_training=False)
		w_logit = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)
		print(w_logit)

		# TODO: Compute predicted labels of the training points
		ztrain_hat = Xtrain @ w_logit # let this linear prediction boundary = 0
		ytrain_hat = (ztrain_hat >= 0).astype(int) # LogReg Slides page 5/18
		# TODO: Compute the accuarcy of the training set
		train_acc = 1 - np.average(np.abs(ytrain_hat - ytrain))

		# TODO: Compute predicted labels of the test points
		ztest_hat = Xtest @ w_logit # let this linear prediction boundary = 0
		ytest_hat = (ztest_hat >= 0).astype(int) # LogReg Slides page 5/18
		# TODO: Compute the accuarcy of the test set
		test_acc = 1 - np.average(np.abs(ytest_hat - ytest))
		return train_acc, test_acc
	
	train_acc, test_acc = runClsExp(200, 2)
	return train_acc, test_acc

print(synClsExperiments())