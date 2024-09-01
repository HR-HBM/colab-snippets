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

Open in Colab

-	-	-
Exercise 11 (rows and columns)	Exercise 12 (row and column vectors)	Exercise 13 (diamond)
Exercise 14 (vector lengths)	Exercise 15 (vector angles)	Exercise 16 (multiplication table revisited)

[ ]

Start coding or generate with AI.
NumPy
NumPy is a Python library for handling multi-dimensional arrays. It contains both the data structures needed for the storing and accessing arrays, and operations and functions for computation using these arrays. Although the arrays are usually used for storing numbers, other type of data can be stored as well, such as strings. Unlike lists in core Python, NumPy's fundamental data structure, the array, must have the same data type for all its elements. The homogeneity of arrays allows highly optimized functions that use arrays as their inputs and outputs.

There are several uses for high-dimensional arrays in data analysis. For instance, they can be used to:

store matrices, solve systems of linear equations, find eigenvalues/vectors, find matrix decompositions, and solve other problems familiar from linear algebra

store multi-dimensional measurement data. For example, an element a[i,j] in a 2-dimensional array might store the temperature tij measured at coordinates i, j on a 2-dimension surface.

images and videos can be represented as NumPy arrays:

a gray-scale image can be represented as a two dimensional array
a color image can be represented as a three dimensional image, the third dimension contains the color components red, green, and blue
a color video can be represented as a four dimensional array
a 2-dimensional table might store a sequence of samples, and each sample might be divided into features. For example, we could measure the weather conditions once per day, and the conditions could include the temperature, direction and speed of wind, and the amount of rain. Then we would have one sample per day, and the features would be the temperature, wind, and rain. In the standard representation of this kind of tabular data, the rows corresponds to samples and the columns correspond to features. We see more of this kind of data in the chapters on Pandas and Scikit-learn.

In this chapter we will go through:

Creation of arrays
Array types and attributes
Accessing arrays with indexing and slicing
Reshaping of arrays
Combining and splitting arrays
Fast operations on arrays
Aggregations of arrays
Rules of binary array operations
Matrix operations familiar from linear algebra

[ ]

Start coding or generate with AI.
We start by importing the NumPy library, and we use the standard abbreviation np for it.


[1]
1s
import numpy as np
Creation of arrays
There are several ways of creating NumPy arrays. One way is to give a (nested) list as a parameter to the array constructor:


[2]
0s
np.array([1, 3, 5])   # one dimensional array, 
array([1, 3, 5])
Note that leaving out the brackets from the above expression, i.e. calling np.array(1,2,3) will result in an error.

Two dimensional array can be given by listing the rows of the array:


[3]
0s
np.array([[1,2,3], [4,5,6], [10, 18, 16]])
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [10, 18, 16]])
Similarly, three dimensional array can be described as a list of lists of lists:


[5]
0s
np.array([[[1,2], [3,4]], [[5,6], [7,8]], [[3, 6], [6, 6]]])
array([[[1, 2],
        [3, 4]],

       [[5, 6],
        [7, 8]],

       [[3, 6],
        [6, 6]]])
There are some helper functions to create common types of arrays:


[6]
0s
np.zeros((3,4))
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
To specify that elements are ints instead of floats, use the parameter dtype:


[8]
0s
np.zeros((3,4), dtype=int)
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]])
Similarly ones initializes all elements to one, full initializes all elements to a specified value, and empty leaves the elements uninitialized:


[9]
0s
np.ones((4,3))
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])

[14]
0s
np.full((6,5), fill_value=5)
array([[5, 5, 5, 5, 5],
       [5, 5, 5, 5, 5],
       [5, 5, 5, 5, 5],
       [5, 5, 5, 5, 5],
       [5, 5, 5, 5, 5],
       [5, 5, 5, 5, 5]])

[15]
0s
np.empty((2,4))
array([[4.86486174e-310, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000],
       [0.00000000e+000, 0.00000000e+000, 0.00000000e+000,
        0.00000000e+000]])
The eye function creates the identity matrix, that is, a matrix with elements on the diagonal are set to one, and non-diagonal elements are set to zero:


[17]
0s
np.eye(12, dtype=int)
array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
The arange function works like the range function, but produces an array instead of a list.


[18]
0s
np.arange(35,102,4)
array([35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99])
For non-integer ranges it is better to use linspace:


[19]
0s
np.linspace(0, np.pi, 15)  # Evenly spaced range with 5 elements
array([0.        , 0.22439948, 0.44879895, 0.67319843, 0.8975979 ,
       1.12199738, 1.34639685, 1.57079633, 1.7951958 , 2.01959528,
       2.24399475, 2.46839423, 2.6927937 , 2.91719318, 3.14159265])

[45]
0s
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
