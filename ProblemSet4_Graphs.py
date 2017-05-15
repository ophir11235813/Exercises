import tensorflow as tf
import numpy as np
from datetime import date

print(date.today())
author = "Ophir Samson"
print(np.__version__)
print(tf.__version__)


# Q1. Create a graph
g = tf.Graph()

with g.as_default():
    # Define inputs
    with tf.name_scope("inputs"):
        a = tf.constant(2, tf.int32, name="a")
        b = tf.constant(3, tf.int32, name="b")

    # Ops
    with tf.name_scope("ops"):
        c = tf.multiply(a, b, name="c")
        d = tf.add(a, b, name="d")
        e = tf.subtract(c, d, name="e")

# Q2. Start a session
sess = tf.Session(graph = g)

# Q3. Fetch c, d, e
_c, _d, _e = sess.run([c,d,e])
print("c =", _c)
print("d =", _d)
print("e =", _e)

# Close the session
sess.close()



tf.reset_default_graph()

a    = tf.Variable(tf.random_uniform([]))
b_pl = tf.placeholder(tf.float32, [None])

# Ops
c = a * b_pl
d = a + b_pl
e = tf.reduce_sum(c)
f = tf.reduce_mean(d)
g = e - f

#Initialize the variables
init = tf.global_variables_initializer()

update_op = tf.assign(a,a + g)

# Q4. Create a (summary) writer to `asset`
writer = tf.summary.FileWriter('asset', tf.get_default_graph())

#Q5. Add `a` to summary.scalar
tf.summary.scalar("a", a)

#Q6. Add `c` and `d` to summary.histogram
tf.summary.histogram("c", c)
tf.summary.histogram("d", d)

#Q7. Merge all summaries.
summaries = tf.summary.merge_all()

sess = tf.Session()
sess.run(init)

for step in range(5):
    _b = np.arange(10, dtype=np.float32)
    _, summaries_proto = sess.run([update_op, summaries], {b_pl: _b})

    writer.add_summary(summaries_proto, global_step=step)

sess.close()











