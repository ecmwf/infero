#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import os
import numpy as np
import pyinfero
from pyinfero.pyinfero import InferoException

def test_mimo():

    # config
    this_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(this_dir, "../../../../../tests/data/mimo_model")

    model_path = os.path.join(data_dir, "mimo_model.tflite")
    model_type = "tflite" 

    input_tensors = {
        "input_1": np.ones((1,32)),
        "input_2": np.ones((1,128))
    }

    output_shapes = {
        "dense_6": (1,1),
    }

    # inference
    infero = pyinfero.Infero(model_path, model_type)
    output_tensors = infero.infer_mimo(input_tensors, output_shapes)

    # check output
    assert np.abs(output_tensors['dense_6'] - 5112.6704) < 0.01

if __name__=="__main__":
    test_mimo()
