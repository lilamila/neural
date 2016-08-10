# PART 2 - Forward Propagation https://www.youtube.com/watch?v=UJwK6jAStmg
class  Neural_Network(object):
	def __init__(self):
		# define hyperparameters
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		# Weights (Parameters) initialize matrices within init method
		self.W1 = np.random.randn(self.inputLayerSize, \
								  self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, \
								  self.outputLayerSize)

	def forward(self, X):
		# propagate inputs through network
		# use matrices to pass through multiple inputs at once for computational speedups esp when using matlab or numpy
		# use numpy's built in dot method
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(z):
		# apply sigmoid activation function
		return 1/(1+np.exp(-z))


testInput = np.arange(-6, 6, 0.01)
plot(testInput, sigmoid(testInput), linewidth=2)
grid(1)


# PART 3 - Gradient Descent https://www.youtube.com/watch?v=5u0jaA3qAGk
# need to quantify how bad our predictions are with cost function



