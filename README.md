# keras_tf_exploration
 
#Goal:  Fun and play.
### Different ways to build Keras models.
### How do custom activation layers work?

# models_keras.pyCharm
### Examples of 3 (4) different ways to build models
#### 1) Sequential (stacking layers)
##### .... Traditional builds have one layer into the next.  Easy to read and understand but limitting.
#### 2) Model building (creating a class model that inherits from keras.Model)
##### .... This allows for easy reuse of a model in mulitple instances as an object.
##### .... Noted, this can be functinional API in setup
#### 3) Functional API (independent layer definitions)
##### .... This is become the more standard form as it allows many different applications.
##### .... .... Multiple inputs and outputs
##### .... .... Combining (concatenating) two layers.
##### .... .... Using shared layers in multiple models for crosstraining or encoding/decoding
#### 4) Model/model (models built as layers)
##### .... Models can be treated as layers in other models.  

# Environments (note windows limits and has Cuda errors specific to GPU versus CPU
##### .... current setup throws warnings and reverts to CPU run, which is fine for basic testing.
##### .... Linux or docker linux install should be considered.
### Anaconda kerasTF_exploration python 3.10.14
#### .... In intalling in Anaconda, only --> run python -m pip install "tensorflow<2.11"
##### .... .... This install will include the correct numpy version. 
### pyCharm D:/dev_test/kerasTF/pyCharm10_2_b
#### .... install is as anaconda above