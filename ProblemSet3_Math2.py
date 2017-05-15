import tensorflow as tf
import numpy as np
from datetime import date

print(date.today())
author = "Ophir Samson"
print(np.__version__)
print(tf.__version__)
sess = tf.InteractiveSession()


#Q1 - Create a diagonal tensor with the diagonal values of x.
_x = np.array([1, 2, 3, 4])
x = tf.convert_to_tensor(_x)
print(tf.diag(x).eval())

#Q2
_X = np.array(
[[1, 0, 0, 0],
 [0, 2, 0, 0],
 [0, 0, 3, 0],
 [0, 0, 0, 4]])
X = tf.convert_to_tensor(_X)
print(tf.diag_part(X).eval())

#HELP Q3 - Permutate the dimensions of x such that the new tensor has shape (3, 4, 2)
Y = tf.range(0,24)
Y = tf.reshape(Y, shape = [2,3,4])
print(Y.eval())

# print(tf.reshape(Y,shape=[3,4,2]).eval())
#Or, using transpose:
print(tf.transpose(Y,perm=[1,2,0]).eval())

#Q4
print(tf.eye(3, dtype = tf.int32).eval())

#Q5 - predict
_X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
X = tf.convert_to_tensor(_X)

diagonal_tensor = tf.matrix_diag(X)
diagonal_part = tf.matrix_diag_part(diagonal_tensor)

print("diagonal_tensor =\n", diagonal_tensor.eval())
print("diagonal_part =\n", diagonal_part.eval())
# print(X.eval())
# print(tf.transpose(X,[1,2,0]).eval())

print('Here')
#Q17 - Complete the einsum function that would yield the same result as the given function.

_X = np.arange(1, 7).reshape((2, 3))
_Y = np.arange(1, 7).reshape((3, 2))

X = tf.convert_to_tensor(_X, dtype= tf.int32)
Y = tf.convert_to_tensor(_Y, dtype= tf.int32)


# Matrix multiplication
out1 = tf.matmul(X, Y)
out1_ = tf.einsum('ij,jk->ik', X, Y)
assert np.allclose(out1.eval(), out1_.eval())


# Dot product
flattened = tf.reshape(X, [-1])
out2 = tf.reduce_sum(flattened * flattened)
out2_ = tf.einsum('i,i->', flattened, flattened)
assert np.allclose(out2.eval(), out2_.eval())


# # Outer product
expanded_a = tf.expand_dims(flattened, 1) # shape: (6, 1)
expanded_b = tf.expand_dims(flattened, 0) # shape: (1, 6)
out3 = tf.matmul(expanded_a, expanded_b)
out3_ = tf.einsum('i,j->ij', flattened, flattened)
assert np.allclose(out3.eval(), out3_.eval())


# Transpose
out4 = tf.transpose(X) # shape: (3, 2)
out4_= tf.einsum('ij->ji',X)
assert np.allclose(out4.eval(), out4_.eval())




