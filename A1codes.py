import numpy as np
from cvxopt import matrix, solvers

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
