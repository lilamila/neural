# neural networks

import numpy as np



NN = Neural_Network()



"""Backpropagation"""
def costFunctionPrime(self, X, y):
	#compute derivative with respect to W1 and W2
    self.yHat = self.forward(X)
    delta3 = np.multiply(-y-self.yHat), self.sigmoidPrime(self.z3))
	dJdW2 = np.dot(self.a2.T, delta3)

# take derivative across synapses


cost1 = NN.costFunction(X,y)
dJdW1, dJdW2 = NN.costFunctionPrime(X,y)


"""Training
Use BFGS to make more informed movements downhill
"""
from videoSupport import *

from scipy import optimize
class trainer(object):
		def __init__(self,N):
			#Mkae local reference to Neural Network:
			self.N = N

		def costFunctionWrapper(self,params, X, y):
			self.N.setParams(params)
			cost = self.N.costFunction(X, y)
			grad = self.N.computeGradients(X, y)
			return cost, grad

		def callbackF(self, params):
			self.N.setParams(params)
			self.J.append(self.N.costFunction(self.X, self.y))


		def train(self, X, y):
			#make internal variable for callback function:
			self.x = x
			self.y = y

			#make empty list to store costs (allows us to track the cost function value as we train the network:
			self.J = []


			paramsO = self.N.getParams()
			options = {'maxiter': 200, 'disp' : True}
			#minimize requires ab objective function that accepts a vector of parameters
			#input and output data and returns both the costs and gradients
			_res = optimize.minimize(self.costFunctionWrapper, params(), \
					jac = True, method='BFGS', args = (X, y), callback =)

			self.N.setParams(_res.x)
			self.optimizationResults = _res

