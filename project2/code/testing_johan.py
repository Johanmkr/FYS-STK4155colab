import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)


# # VERY SIMPLE EXAMPLE
# d = 2
# coeff = np.random.randint(1,10,d+1)


# def f(x):
#     return coeff[0] + coeff[1]*x + coeff[2]*x**2

# def df(x):
#     return coeff[1] + 2 * coeff[2] * x

# x = np.linspace(-10,10,100)

# plt.plot(x, f(x))

# n_iter = 1000
# x_old = x[0]
# plt.plot(x_old, f(x_old), ".")
# gamma = .01

# for _ in range(n_iter):
#     x_new = x_old - gamma*df(x_old)
#     x_old = x_new
#     plt.plot(x_new, f(x_new), ".")

# print(x_new)
# plt.show()


#   LINEAR REGRESSION TEST

#   number of data points
n = 100
# x = np.random.rand(n,1)
x = np.random.uniform(-1,1,n)
x_lin = np.linspace(-1,1,n)
x = x_lin
# print(np.random.normal(0,1,n))
y = 4 + 3*x + 8*x**2 + np.random.normal(0,1,n)

plt.plot(x, y ,'ro')
# plt.show()

X = np.c_[np.ones((n,1)), x, x**2]

print(X)

#   Hessian
H = (2/n) * X.T @ X 

#Eigenvalues
EigVal, EigVec = np.linalg.eig(H)

beta = np.random.randn(3,1)
eta = 1.0/np.max(EigVal)
Niterations = 1000

for _ in range(Niterations):
    gradient = 2/n * X.T @ (X @ beta - y)
    from IPython import embed; embed()
    beta -= eta*gradient
# from IPython import embed; embed()
print(beta)

# plt.plot(x, X@beta)
# plt.show()



# # xnew = np.array([[0],[2]])
# # xbnew = np.c_[np.ones((2,1)), xnew]
# # ypredict = xbnew.dot(beta)
# # plt.plot(xnew, ypredict, "r-")
# x_new = np.linspace(0,1,100)
# X_new = np.c_[np.ones((n, 1)), x_new]
# ypredl = X_new @ beta
# # from IPython import embed; embed()
# plt.plot(x_new, ypredl, "-b")
# plt.plot(x, y ,'ro')
# plt.axis([0,2.0,0, 15.0])
# plt.xlabel(r'$x$')
# plt.ylabel(r'$y$')
# plt.title(r'Gradient descent example')
# plt.show()

