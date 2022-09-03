# Gradient Descent for Linear Regression
# Source: Nicholas Renotte (https://youtu.be/Souzjv6WfrY)
# yhat = wx + b
# Loss = (y-yhat)**2 / N

import numpy as np

# Initialize some parameters
x = np.random.randn(10,1)
# print(x)
y = 5*x + np.random.randn()

# Parameters
w = 0.0
b = 0.0

# Hyperparameter
learning_rate = 0.01
# print(x.shape[0])

# Create the gradient descent function
def descend(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    # Loss = (y-(wx + b))**2
    for xi, yi in zip(x,y):
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))

    # Make an updates to the w parameter
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    return w, b

# Iteratively make updates
for epoch in range(400):
    w, b = descend(x,y,w,b,learning_rate)
    # Run gradient descent
    yhat = w*x + b
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])
    print(f'Epoch:{epoch} --- Loss is {loss}, parameters w:{w}, b:{b}')