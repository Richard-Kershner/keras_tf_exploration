from keras import backend as K
import tensorflow as tf
import math

# note, if .001 isn't added in, then divide by zero could crash
act_pos_lambda = lambda a: (1 / (1 + math.e ** (-(a + math.e) * math.e)))
act_neg_lambda = lambda a: (1 / (1 + math.e ** (-(-a + math.e) * math.e)))

def act_bell_relu_keras(x, beta=1.0):
    #test = tf.cast(tf.math.greater_equal(x,0.0),act_pos_lambda(x))
    map_pos = tf.math.greater_equal(x,0.0)
    map_neg = tf.math.less(x, 0.0)
    x1 = K.switch(map_pos, x, act_pos_lambda(x))
    x2 = K.switch(map_neg, x1, act_neg_lambda(x1))
    return x2