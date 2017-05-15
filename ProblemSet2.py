import tensorflow as tf
import numpy as np
from datetime import date

print(date.today())
author = "Ophir Samson"
print(np.__version__)


sess = tf.InteractiveSession()
#Q1
x = tf.random_uniform([])
y = tf.random_uniform([])
a1 = tf.cond(x-y<0, lambda: tf.add(x,y), lambda: tf.subtract(x,y))
print(a1.eval())

#Q2
x2 = tf.random_uniform([],minval = 0, maxval= 5, dtype = tf.int32)
y2 = tf.random_uniform([],minval = 0, maxval= 5, dtype = tf.int32)
a2 = tf.case({x2 < y2 : lambda: x2 + y2,
              x2 > y2 : lambda: x2 - y2},
              default = lambda: tf.constant(0),
              exclusive= True)
print(a2.eval())

#Q3
X  = tf.constant([[-1,-2,-3], [0,1,2]], dtype = tf.int32)
Y  = tf.zeros(shape = X.shape, dtype = X.dtype)
a3 = tf.equal(X,Y)   # Compare two tensors
print(a3.eval())

#Q4
x = tf.constant([True, False, False], tf.bool)
y = tf.constant([True, True, False], tf.bool)
a4a = tf.logical_and(x,y)
a4b = tf.logical_or(x,y)
a4c = tf.logical_xor(x,y)
print(a4a.eval(), a4b.eval(), a4c.eval())

#Q5
x = tf.constant([True,False,False], tf.bool)
a5 = tf.logical_not(x)
print(a5.eval())

#Q6
X  = tf.constant([[-1,-2,-3], [0,1,2]], dtype = tf.int32)
Y  = tf.zeros_like(X)
a6 = tf.not_equal(X,Y)
print(a6.eval())

#Q7
a7 = tf.greater(X,Y)
print(a7.eval())

#Q8
X  = tf.constant([[1,2],[3,4]], dtype = tf.int32)
Y  = tf.constant([[5,6],[7,8]], dtype = tf.int32)
Z  = tf.constant([[True, False], [False, True]], dtype = tf.bool)

# a8 = tf.case({Z: lambda: X}, default = lambda: Y)
a8 = tf.where(Z,X,Y)
print(a8.eval())

