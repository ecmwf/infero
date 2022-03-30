#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import concatenate
import keras.initializers
import tensorflow as tf

# import keras2onnx


if __name__ == "__main__":
    """
    Setup a minimal multi-input model for testing
    """
    
    init_dict = {
        "kernel_initializer": keras.initializers.Constant(0.3)
    }

    # input branch A
    inputA = Input(shape=(32,), dtype=tf.float32)
    x = Dense(8, activation="relu", **init_dict)(inputA)
    x = Dense(4, activation="relu", **init_dict)(x)
    x = Model(inputs=inputA, outputs=x)

    # input branch B
    inputB = Input(shape=(128,), dtype=tf.float32)
    y = Dense(64, activation="relu", **init_dict)(inputB)
    y = Dense(32, activation="relu", **init_dict)(y)
    y = Dense(4, activation="relu", **init_dict)(y)
    y = Model(inputs=inputB, outputs=y)
    
    # concatenate inputs
    combined = concatenate([x.output, y.output])
    
    z = Dense(2, activation="relu", **init_dict)(combined)
    z = Dense(1, activation="linear", **init_dict)(z)
    
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.summary()
    
    # prediction on np.ones inputs
    n_batch = 10
    result = model.predict([
        np.arange(n_batch * 32, dtype=float).reshape(n_batch, 32)/(n_batch * 32),
        np.arange(n_batch * 128, dtype=float).reshape(n_batch, 128)/(n_batch * 128)
    ])
    
    # expected 5112.6704.
    print(result)

    # Write in onnx format
    # onnx_model = keras2onnx.convert_keras(model, "test_onnx")
    # file = open("mimo_model.onnx", "wb")
    # file.write(onnx_model.SerializeToString())
    # file.close()
    
    # write in tf format
    tf.keras.models.save_model(model, "mimo_model_tf", save_format="tf")
    
    # write in TFlite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_path = converter.convert()

    # Save the model.
    with open("mimo_model.tflite", 'wb') as f:
        f.write(tflite_model_path)
    
    
 
