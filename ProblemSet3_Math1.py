import tensorflow as tf
import numpy as np
from datetime import date

print(date.today())
author = "Ophir Samson"
print(np.__version__)
print(tf.__version__)

sess = tf.InteractiveSession()

#Q1
_x = np.array([1, 2, 3])
_y = np.array([-1, -2, -3])

x = tf.convert_to_tensor(_x)
y = tf.convert_to_tensor(_y)

a1 = tf.add(x,y)
print(a1.eval())

#Q2
_x = np.array([3,4,5])
_y = np.array(3)
x  = tf.convert_to_tensor(_x)
y  = tf.convert_to_tensor(_y)
a2 = tf.subtract(x,y)
print(a2.eval())

#Q3
X = tf.constant([3,4,5])
Y = tf.constant([1,0,-1])
print(tf.multiply(X,Y).eval())

#Q4
X = tf.constant([1,2,3])
print((5*X).eval())

#Q5 - predict the result of this:
_x = np.array([10, 20, 30], np.int32)
_y = np.array([2, 3, 5], np.int32)
x = tf.convert_to_tensor(_x)
y = tf.convert_to_tensor(_y)
out1 = tf.div(x, y)
out2 = tf.truediv(x, y)
print(np.array_equal(out1.eval(), out2.eval())) # => False

print(out1.eval(), out1.eval().dtype) # tf.div() returns the same results as input tensors => [5,6,6], int32
print(out2.eval(), out2.eval().dtype)# tf.truediv() always returns floating point results. => [5,6.666,6] float64

#Q6
_x  = np.array([10, 20, 30], np.int32)
_y  = np.array([2, 3, 7], np.int32)
x   = tf.convert_to_tensor(_x)
y   = tf.convert_to_tensor(_y)
a6  = tf.mod(x,y) #Remainder....
print(a6.eval())

#Q7
_x = np.array([1, 2, 3], np.int32)
_y = np.array([4, 5, 6], np.int32)
x = tf.convert_to_tensor(_x)
y = tf.convert_to_tensor(_y)
a7 = tf.cross(x,y)
print(a7.eval())

#Q8
_x = np.array([1, 2, 3], np.int32)
_y = np.array([4, 5, 6], np.int32)
_z = np.array([7, 8, 9], np.int32)
x = tf.convert_to_tensor(_x)
y = tf.convert_to_tensor(_y)
z = tf.convert_to_tensor(_y)
a8 = x+y+z
print(a8.eval())

#Q9 - abs value
_X = np.array([[1, -1], [3, -3]])
X = tf.convert_to_tensor(_X)
print(abs(X).eval())

# Show the positive values of X
aa = tf.where(X>0,X,tf.zeros_like(X))
print(aa.eval())


#Q10
_x = np.array([1, -1])
x = tf.convert_to_tensor(_x)
print(tf.negative(x).eval())

#Q11
_x = np.array([1, 3, 0, -1, -3])
x = tf.convert_to_tensor(_x)
a11 = tf.sign(x)
print(a11.eval())

#Q12
_x = np.array([1, 2, 2/10], dtype= np.float64)
x = tf.convert_to_tensor(_x, dtype = tf.float64)
a12 = tf.reciprocal(x)
a13 = 1/x
print(a13.eval())

#Q13
_x = np.array([1, 2, -1])
x = tf.convert_to_tensor(_x)
print(tf.square(x).eval())

#Q14 - predict
_x = np.array([2.1, 1.5, 2.5, 2.9, -2.1, -2.5, -2.9])
x = tf.convert_to_tensor(_x)
out1 = tf.round(x)
out2 = tf.floor(x)
out3 = tf.ceil(x)
print(out1.eval())
print(out2.eval())
print(out3.eval())

_out1 = np.around(_x)
_out2 = np.floor(_x)
_out3 = np.ceil(_x)
assert np.array_equal(out1.eval(), _out1) # tf.round == np.around
assert np.array_equal(out2.eval(), _out2) # tf.floor == np.floor
assert np.array_equal(out3.eval(), _out3) # tf.ceil == np.ceil

#Q15 - reciprocal of sqrt
_x = np.array([1., 4., 9.])
x = tf.convert_to_tensor(_x)
print( (1/ tf.sqrt(x)).eval())

#Q16
_x = np.array([[1, 2], [3, 4]])
_y = np.array([[1, 2], [1, 2]])
x = tf.convert_to_tensor(_x)
y = tf.convert_to_tensor(_y)
print((x**y).eval())

#Q22
_x = np.array([2, 3, 4])
_y = np.array([1, 5, 1])
x = tf.convert_to_tensor(_x)
y = tf.convert_to_tensor(_y)
print( tf.multiply((x-y),(x-y)).eval())






