import numpy.random as random
import numpy as np


# Generating the dataset
random.seed(10)
mu, sigma = 0, 1

X = []
Y = []
# ERROR = []
for x1 in range(0, 25, 1):
    for x2 in range(0, 25, 1):
        X.append([x1, x2, 1])
        error = random.normal(mu, sigma, 1)
        # ERROR.append(error)
        # chosen plane: y = 2x1 + 4x2 + 7, w1 = 2, w2 = 4, w3 = 7#
        Y.append(2 * x1 + 4 * x2 + 7 + error)


X_np = np.array(X)
Y_np = np.array(Y)

#Calculating the weights based on the dataset and the formula W = (X^TX)^-1X^TY

X_Transpose = X_np.transpose()
X_Transpose_X = np.matmul(X_Transpose, X_np)
X_Transpose_X_Inv = np.linalg.inv(X_Transpose_X)

X_Transpose_X_Inv_X_Transpose = np.matmul(X_Transpose_X_Inv, X_Transpose)

W = np.matmul(X_Transpose_X_Inv_X_Transpose, Y_np)

print("The function is: y = ", W[0], "x1 + ", W[1], "x2 + ", W[2])

