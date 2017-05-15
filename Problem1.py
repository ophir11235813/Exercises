import tensorflow as tf
import numpy as np
from datetime import date

print(date.today())
author = "Ophir Samson"

print(np.__version__)

#Q1
sess = tf.InteractiveSession()
mat_zero = tf.zeros([2,3],dtype = tf.float32)
#Q2
X        = tf.constant([[1,2,3],[4,5,6]], dtype= tf.float32)
print(X.eval())
X2       = tf.zeros(shape= X.shape, dtype = X.dtype)
print(X2.eval())
#Q3
X3       = tf.ones([2,3], dtype = tf.float32)
print(X3.eval())
#Q4
X4       = tf.ones(shape = X.shape, dtype = X.dtype)
print(X4.eval())
#Q5
X5       = 5*tf.ones([3,2], dtype = tf.float32)
print(X5.eval())
#Q6
X6       = tf.constant([[1,3,5],[4,6,8]], dtype= tf.float32)
print(X6.eval())
#Q7
X7       = 4*tf.ones(shape = [2,3], dtype = tf.float32)
print(X7.eval())

#Q8
A1 = tf.constant(np.linspace(5,10,50),shape = [1,50])
print(A1.eval())
#Q9
A2 = tf.range(10,102,2)
print(A2.eval())
#Q10
A3 = tf.random_normal(shape = [3,2],mean = 0, stddev= 2)
print(A3.eval())
#Q11
A4 = tf.truncated_normal(shape=[3,2], mean = 0, stddev=1)
print(A4.eval())
#Q12
A5 = tf.random_uniform(shape = [3,2], minval = 0, maxval=2)
print(A5.eval())
#Q13
Y  = tf.constant([[1,2],[3,4],[5,6]], dtype = tf.float32)
Y1 = tf.random_shuffle(Y)
print(Y1.eval())
#Q14
X  = tf.random_normal(shape = [10,10,3],mean =0, stddev=1)
Y2 = tf.slice(X,begin = [0,0,0], size = [5,5,3])
print(X.eval())
print(Y2.eval()) #Note this will "run" X again, thus producing new random variables








