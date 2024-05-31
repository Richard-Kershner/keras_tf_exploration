import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# note below on ops crashing... this is because TF and Windows and GPU issues (runs CPU)
import tensorflow as tf

#keras = tf.keras  # wither of these works
import keras
from keras import layers

# ops crashes as GPU/windows/TF don't play nice
#from keras import ops

print("python version: ", sys.version)
print("numpy version", np.__version__)
print("pandas verson: ", pd.__version__)

path = str( Path(os.getcwd()).absolute() )

print(type(path), path )

sys.path.append(path)
from act_bell_relu_keras import act_bell_relu_keras


# 3 ways to create a keras model --- made it into 4 with sequential model stacking
# https://keras.io/api/models/

def build_compile_pred_train_model(model, in_shape):
    model.build(in_shape)
    model.compile(optimizer='Adam',
       loss='categorical_crossentropy',
       metrics=['accuracy'])
    # according, x and y must both be tensors or numpy arrays.. not mixed?
    print(model.summary())
    x = np.ones(in_shape)  # keras.ops.ones((3, 3))
    print("x", x)

    y_pred = model(x).numpy() # don't mix/match tensors and numpyArrays
    print("y ", y_pred)
    model.fit(x=x, y=y_pred)


in_shape = (1,3)

# ===== 1) Secquential (Serial) model =====
print("\n ========== Sequential_Model ==========")
def create_keras_sequential():
    model = keras.Sequential(
        [
            layers.Dense(3, activation="relu", name="layer1"),
            layers.Dense(3, activation="relu", name="layer2"),
            layers.Dense(3, name="layer3"),
        ]
    )
    return model

model_S = create_keras_sequential()
build_compile_pred_train_model(model_S, in_shape)

# ===== 2) model building =====
print("\n ========== Model_Model ==========")
class Model_Model(keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = keras.layers.Dense(3, activation="relu", name="layer1")
        self.layer2 = keras.layers.Dense(3, activation="relu", name="layer2")
        self.layer3 = keras.layers.Dense(3, name="layer3")

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return self.layer3(x)

model_M = Model_Model()
build_compile_pred_train_model(model_M, in_shape)


print("\n ========== Functional API ==========")
def create_keras_funAPI_model(in_shape):
    in_int = int(in_shape[1])
    input_layer = keras.Input(shape=in_int)
    first_layer = layers.Dense(in_int, activation="relu")(input_layer)
    second_layer = layers.Dense(in_int, activation="relu")(first_layer)
    # models can be used as layers......
    mod_layer = model_M(second_layer)
    output_layer = layers.Dense(in_int)(mod_layer) # two dense layers created

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

model_funAPI = create_keras_funAPI_model(in_shape)
#build_compile_pred_train_model(model_funAPI, in_shape)
build_compile_pred_train_model(model_funAPI, in_shape)

# ===== 4) stacking models =====
print("\n ========== model_model_stack ==========")
# can stack models instead of layers
model_model_stacks = keras.Sequential([
    model_S,
    model_M,
    #model_funAPI, # doesn't play nice... Nan values
    layers.Dense(5),
])

# model_model_stacks.build(in_shape)
# build_compile_pred_train_model(model_model_stacks, in_shape)

# ===== custom activation =====
print("===== custom activation =====")
from keras import backend as K
from keras.layers.core import Activation
from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'act_bell_relu_keras': Activation(act_bell_relu_keras)})

def create_api_with_custom_layer(in_shape):
    in_int = int(in_shape[1])
    input_layer = keras.Input(shape=in_int)
    first_layer = layers.Dense(in_int, activation="relu")(input_layer)
    first_layer = layers.Dense(in_int, activation="relu")(input_layer) # activation="bell_reLU_keras"
    second_layer = layers.Dense(in_int, activation="relu")(first_layer)
    # models can be used as layers......
    mod_layer = model_M(second_layer)
    output_layer = layers.Dense(in_int)(mod_layer) # two dense layers created

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

model_custom_layer_funAPI = create_api_with_custom_layer(in_shape)
build_compile_pred_train_model(model_custom_layer_funAPI, in_shape)