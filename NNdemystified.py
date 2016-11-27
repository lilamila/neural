
# coding: utf-8

# ### Part 1

# In[1]:

from IPython.display import YouTubeVideo
YouTubeVideo('bxe2T-V8XRs')


# In[2]:

get_ipython().magic(u'pylab inline')


# In[3]:

import numpy as np
# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([0.75], [0.82], [0.93]), dtype=float)

print X.shape, y.shape


# ### PART 2 - Forward Propagation

# In[47]:

import numpy as np
import matplotlib.pyplot as plt

# https://www.youtube.com/watch?v=UJwK6jAStmg
class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
    
    def sigmoid(self, z):
        # apply sigmoid activation function
        return 1/(1+np.exp(-z))
    
# 
# def sigmoid(x):
#     # apply sigmoid activation function
#         return 1/(1+np.exp(-z))
    


# In[48]:

testInput = np.arange(-6, 6, 0.01)
plt.plot(testInput, sigmoid(testInput), linewidth=2)
grid(1)


# In[49]:

sigmoid(np.random.randn(3,3))


# In[50]:

# PART 3 - Gradient Descent https://www.youtube.com/watch?v=5u0jaA3qAGk
# need to quantify how bad our predictions are with cost function

NN = Neural_Network()
print NN.W1


# In[51]:

yHat = NN.forward(X)
print yHat
print yHat.shape


# In[52]:

y


# ### PART 3 - Gradient Descent

# In[53]:


#Compare estimate, yHat, to actually score
plt.bar([0,1,2], y, width = 0.35, alpha=0.4)
plt.bar([0.35,1.35,2.35],yHat, width = 0.35, color='pink', alpha=0.5)
grid(1)
legend(['y', 'yHat'])


# Next step is to apply a cost function to see how inaccurate our predictions were.
# --> curse of dimensionality

# In[54]:

import time

weightsToTry = np.linspace(-5, 5, 1000)
costs = np.zeros(1000)

startTime = time.clock()
for i in range(1000):
    NN.W1[0,0] = weightsToTry[i]
    yHat = NN.forward(X)
    costs[i] = 0.5*sum((y-yHat)**2)

endTime = time.clock()


# In[55]:

timeElapsed = endTime-startTime
timeElapsed


# In[56]:

plt.plot(weightsToTry, costs)
grid(1)
ylabel('Cost')
xlabel('Weight')


# To find the rate of change of yHat with respect to z3, we need to **differentiate** our sigmoid activation function with respect to z. Our derivative should be largest where our sigmoid function is the steepest.

# In[59]:

def sigmoidPrime(self,z):
    #Derivative of sigmoid function
    return np.exp(-z)/((1+np.exp(-z))**2)


# In[60]:

testValues = np.arange(-5,5,0.01)
plt.plot(testValues, sigmoid(testValues), linewidth=2)
plt.plot(testValues, sigmoidPrime(testValues), linewidth=2)
grid(1)
legend(['sigmoid', 'sigmoidPrime'])


# 

# we can now replace dyHat/z3 with f prime(z3)
# 
# Z3 is the matrix product of our activities (a2) and our weights (W2). The activities in layer 2 are multiplied by their corresponding weights and added together to yield z3. If we focus on a single synapse, we see a simple linear relationship between W and z, where a is the slope. So for each synapse, dJdW2 is just the activation a on that synapse. 
# 
# Another way to think about the calculus here is that it is **backpropagating** an error to each weight.
# The weights that contribute more to the overall error, will have larger activations, yield larger djdW2 values and will be changed **more** when we perform gradient descent.
# 
# Need to review to 4:44
# 
# 
# 

# In[61]:

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
    def forward(self, X):
        #Propagate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
    
    def sigmoid(self, z):
        # apply sigmoid activation function
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self, z):
        # gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        # computer cost for given X, y, use weights already stored in class
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
    
    def costFunctionPrime(self, X, y):
        # compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2
    


# If we want to create a deeper neural network, we can just stack a bunch of these operations together.
# How do we change our Ws to **decrease** our cost? We can now compute dJdW which tells us which way is uphill, and which is downhill.
# 
# Stochastic gradient descent
# Don't need to be afraid of non-convex loss functions

# In[62]:

NN= Neural_Network()
cost1 = NN.costFunctionPrime(X, y)
dJdW1, dJdW2 = NN.costFunctionPrime(X,y)


# In[63]:

dJdW1


# In[64]:

dJdW2


# if we move this way by adding a scalar times our derivative to our weights, our cost will increase, and if we do the opposite, subtract our gradient from our weights, we will move downhill and reduce our cost. This simple step downhill is the core of gradient descent and a key part of how even very sophisticated learning algos are trained

# In[65]:

scalar = 3
NN.W1 = NN.W1 + scalar*dJdW1
NN.W2 = NN.W2 + scalar*dJdW2
cost2 = NN.costFunction(X, y)


# In[66]:

print cost1, cost2


# ### PART 5 - Numerical Gradient Checking
# 
# to ensure coded up gradients correctedly
# - review derivatives
# 

# In[67]:

get_ipython().magic(u'pylab inline')

def f(x):
    return x**2
epsilon = 1e-4
x= 1.5

numericalGradient = (f(x+epsilon)-f(x-epsilon))/(2*epsilon)

numericalGradient, 2*x


# In[68]:

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (parameters)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3) 
        return yHat
        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


# we can use the same approach to numerically evaluate the gradient of our neural network. it's a little more complicated because we have 9 gradient values now, and we're interested in gradient of cost function/.
# solution = test one gradient at a time, perturbing each weight, adding epsilon to the current value and computing the cost function, subtracting epsilon from the current value and computing the cost function, and then computing the slope between these two values

# In[69]:

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 


# In[70]:

NN = Neural_Network()
numgrad = computeNumericalGradient(NN, X, y)
numgrad


# In[71]:

grad = NN.computeGradients(X,y)
grad

norm(grad-numgrad)/norm(grad+numgrad)


# ### PART 6 - Training
# 
# Use BFGS algorithim to find 2nd order derivative, or curvature of cost

# In[72]:

from scipy import optimize

class trainer(object):
    def __init__(self, N):
        # make local reference to Neural Network
        self.N = N
    
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))
    
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)

        return cost, grad
    
    def train(self, X, y):
        # make an internal variable for the callback function:
        self.X = X
        self.y = y
        
        # make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()
        
        # set jacobian parameter to true since we are computing the gradient within our neural network class
        # set callback function that allows us to track the cost function value as we train the network
        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',                                args=(X, y), options=options, callback=self.callbackF)
        
        self.N.setParams(_res.x)
        self.optimizationResults = _res
        


# In[73]:

NN = Neural_Network()


# In[74]:

T = trainer(NN)


# In[75]:

T.train(X, y)


# In[76]:

plt.plot(T.J)
grid(1)
xlabel('Iterations')
ylabel('Cost')


# In[77]:

NN.costFunctionPrime(X, y)


# In[78]:

NN.forward(X)


# In[79]:

y


# In[80]:

# test network for various combinations of sleep/study:
hoursSleep = linspace(0, 10, 100)
hoursStudy = linspace(0, 5, 100)

# normalize data (same way training data way normalized)
hoursSleepNorm = hoursSleep/10.
hoursStudyNorm = hoursStudy/5.

# create 2-d versions of input for plotting
a, b = meshgrid(hoursSleep, hoursStudyNorm)

# join into a single input matrix:
allInputs = np.zeros((a.size, 2))
allInputs[:, 0] = a.ravel()
allInputs[:, 1] = b.ravel()


# In[81]:

allOutputs = NN.forward(allInputs)


# In[82]:

allOutputs


# In[83]:

# contour plot

yy = np.dot(hoursStudy.reshape(100,1), np.ones((1,100)))
xx = np.dot(hoursSleep.reshape(100,1), np.ones((1,100))).T

CS = contour(xx, yy, 100*allOutputs.reshape(100, 100))
clabel(CS, inline=1, fontsize=10)
xlabel('Hours Sleep')
ylabel('Hours Study')


# In[85]:

# 3d plot:
## uncommont to plot out of notebook(you'll be able to rotate)
# %matplotlib qt

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gcq(projection='3d')

surf = ax.plot_surface(xx, yy, 100*allOutputs.reshape(100,100), cmap=cm.jet)

ax.set_xlabel('Hours Sleep')
ax.set_ylabel('Hours Study')
ax.set_zlabel('Test Score')


# ### PART 7 - Overfitting, Testing, and Regularization
# 
# ensure that the model is fitting on the signal, not the noise
# make overfitting obvious
# good rule of thumb is to have at least 10 times as many examples as the degrees of freedom in your model.

# In[91]:

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
y = np.array(([75], [82], [93], [70]), dtype=float)


# In[92]:

# plot projections of our new data:
fig = figure(0,(8,3))

subplot(1,2,1)
scatter(X[:, 0], y)
grid(1)
xlabel('Hours Sleeping')
ylabel('Test Score')

subplot(1,2,2)
scatter(X[:,1], y)
grid(1)
xlabel('Hours Sleeping')
ylabel('Test Score')


# In[93]:

# normalize
X = X/np.amax(X, axis=0)
y = y/100 # max test score is 100

# train network with new data:
T = trainer(NN)
T.train(X, y)


# In[95]:

# plot the cost during training:
plot(T.J)
grid(1)
xlabel('Iterations')
ylabel('Cost')


# another way to avoid overfitting is by regularization, which is a way of penalizing overly complex models by adding a **term**. (lasso regression?)
# or ADD TOGETHER SQUARE OF THE WEIGHTS OF THE COST FUNCTION. this way models with larger magnitudes of weight cost more
# we'll need to normalize the other part of our cost function to ensure that our ratio of the two error terms does not change in respect to the number of examples.
# -introduce a regularization ie to add a term to our cost function that penalizes overly complex models. a simple, but effective way to do this is to add together the square of our weights our cost function. this way, models with larger magnitudes of weights cost more. we'll need to normalize the other part of our cost function to ensure that our ratio of the two error terms does not change with respect to the number of examples.
# we introduce a regularization hyper-parameter, lambda, that will allow us to tune the relative cost - higher values of lambda will impose bigger penalties for high model complexity

# In[ ]:

# regularization parameter:
Lambda = 0.0001

# need to make changes to costFunction and costFunctionPrim:

def costFunction(self, X, y):
    # compute cost for given X, w use weights already stored in class
    self.yHat = self.forward(X)
    

