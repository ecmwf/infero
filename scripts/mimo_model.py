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

import keras2onnx


if __name__ == "__main__":
    """
    Setup a minimal multi-input model for testing
    """
    
    init_dict = {
        "kernel_initializer": keras.initializers.Constant(0.3)
    }

    # input branch A
    inputA = Input(shape=(32,))
    x = Dense(8, activation="relu", **init_dict)(inputA)
    x = Dense(4, activation="relu", **init_dict)(x)
    x = Model(inputs=inputA, outputs=x)

    # input branch B
    inputB = Input(shape=(128,))
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
    result = model.predict([np.ones((1, 32), dtype=float), np.ones((1, 128), dtype=float)])
    
    # expected 5112.6704.
    print(result)

    # Write in onnx format
    onnx_model = keras2onnx.convert_keras(model, "test_onnx")
    file = open("test_onnx.onnx", "wb")
    file.write(onnx_model.SerializeToString())
    file.close()
    
 
