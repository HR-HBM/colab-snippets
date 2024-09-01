# initiate numpy

import numpy as np

# matrix definition

a = np.array([[1, 2, 3], [4, 5, 8], [7, 1, 3]])
b = np.array([[1, 0, 2], [3, 5, 4], [1, 7, 10]])
print (f"A: \n{a}")
print (f"B: \n{b}")
c_emul = a*b
print (f"C(element-wise multiplication): \n{c_emul}")
c = np.dot(a,b) 
# or use np.matmul(a,b)
print (f"C: \n{c}")
d = np.divide(a,b)
print (f"D: \n{d}")
e = a.T
print (f"E: \n{e}")
f = b.T
print (f"F: \n{f}")
inverse_A = np.linalg.inv(a)
print (f"inverse of A: \n{inverse_A}")
inverse_B = np.linalg.inv(b)
print (f"inverse of B: \n{inverse_B}")

g = a/b
print (f"G: \n{g}")

g = a/b
print (f"G: \n{g}")

# A: 
# [[1 2 3]
#  [4 5 8]
#  [7 1 3]]
# B: 
# [[ 1  0  2]
#  [ 3  5  4]
#  [ 1  7 10]]
# C(element-wise multiplication): 
# [[ 1  0  6]
#  [12 25 32]
#  [ 7  7 30]]
# C: 
# [[ 10  31  40]
#  [ 27  81 108]
#  [ 13  26  48]]
# D: 
# [[1.                inf 1.5       ]
#  [1.33333333 1.         2.        ]
#  [7.         0.14285714 0.3       ]]
# E: 
# [[1 4 7]
#  [2 5 1]
#  [3 8 3]]
# F: 
# [[ 1  3  1]
#  [ 0  5  7]
#  [ 2  4 10]]
# inverse of A: 
# [[  3.5  -1.5   0.5]
#  [ 22.   -9.    2. ]
#  [-15.5   6.5  -1.5]]
# inverse of B: 
# [[ 0.40740741  0.25925926 -0.18518519]
#  [-0.48148148  0.14814815  0.03703704]
#  [ 0.2962963  -0.12962963  0.09259259]]
# G: 
# [[1.                inf 1.5       ]
#  [1.33333333 1.         2.        ]
#  [7.         0.14285714 0.3       ]


a = np.array([[1, 2, 3], [4, 5, 8], [7, 1, 3]])
b = np.array([[1, 0, 2], [3, 5, 4], [1, 7, 10]])


print (f"A: \n{a}")
print (f"B: \n{b}")

c_emul = a*b
print (f"C(element-wise multiplication): \n{c_emul}")


c = np.dot(a,b) 
# or use np.matmul(a,b)
print (f"C: \n{c}")


d = np.divide(a,b)
print (f"D: \n{d}")

# transpose A
e = a.T 
print (f"E: \n{e}")
f = b.T
print (f"F: \n{f}")


inverse_A = np.linalg.inv(a)
print (f"inverse of A: \n{inverse_A}")
inverse_B = np.linalg.inv(b)
print (f"inverse of B: \n{inverse_B}")

g = a/b
print (f"G: \n{g}")

# h = np.linalg.det(a)

# print(f"determinant of A: \n{h}")

