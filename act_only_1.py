from keras import backend as K
import tensorflow as tf

def act_only_1(x, beta=1.0):
    # Perhaps the truth table should reflect multiple max values
    print("act_only_1  x,  max, truth table ", x, K.max(x), K.equal(x, K.max(x)))
    return K.switch(K.equal(x, K.max(x)), x, 0.0)
