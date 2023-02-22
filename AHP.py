import numpy as np
from functools import reduce

'''1.Input data'''
print("Please enter the judgment matrix dimension:")
n = eval(input())
print("Please enter the judgment matrix:")
A = np.ones((n, n))
for i in range(n):
    A[i] = input().split(" ")
    A[i] = list(map(float, A[i]))
print("The judgment matrix is:\n{}".format(A))

'''2.Consistency Check'''
w, v = np.linalg.eig(A)
wIndex = np.argmax(w)
wMax = np.real(w[wIndex])
print("Maximum eigenvalue value:{}".format(wMax))

CI = (wMax - n) / (n - 1)
print("CI = {}".format(CI))
# RI is from "Judgment scales and consistency measure in AHP".
# Procedia Economics and Finance, 12(2014).
RI = [0, 0, 0.0001, 0.52, 0.89, 1.12, 1.26, 1.36,
      1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59,
      1.5943, 1.6064, 1.6133, 1.6207, 1.6292]
print("RI = {}".format(RI[n]))
CR = CI / RI[n]
print("CR = {}".format(CR))

# make consistency check
if CR > 0.1:
    print("The consistency of the judgment matrix A is not acceptable.")
else:
    print("The consistency of the judgment matrix A is acceptable.")

'''3.Normalization'''
lineSum = [sum(m) for m in zip(*A)]
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        D[i][j] = A[i][j] / lineSum[j]
print("The normalized judgment matrix is:\n{}".format(D))

'''4.Calculate weight'''
# Arithmetic mean method to calculate the weight
ans = np.zeros(n)
for i in range(n):
    ans[i] = np.average(D[i])
print("The result of Arithmetic mean method is:\n{}".format(ans))
# Geometric mean method to calculate the weight
ans = np.zeros(n)
for i in range(n):
    ans[i] = reduce(lambda x, y: x * y, A[i])
    ans[i] = pow(ans[i], 1 / n)
ans = [e / np.sum(ans) for e in ans]
print("The result of Geometric mean method is:\n{}".format(ans))
# Eigenvalue method to calculate weight
ans = np.zeros(n)
vIndex = np.argmax(v)  # Eigenvector index corresponding to largest eigenvalue
vMax = np.real(v[:, vIndex])
ans = [e / np.sum(vMax) for e in vMax]
print("The weight calculation result of the eigenvalue method is:\n{}".format(ans))
