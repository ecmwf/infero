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

def test_cyclone():

    # config
    this_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(this_dir, "../../../../../tests/data/cyclone")
    input_path = os.path.join(data_dir, "cyclone_input_200x200.npy")
    model_path = os.path.join(data_dir, "cyclone_model_200x200.tflite") 
    model_type = "tflite" 
    model_output_shape = (1,200,200,1) 

    # load input
    input_tensor = np.load(input_path)

    # inference
    infero = pyinfero.Infero(model_path, model_type)
    output_tensor = infero.infer(input_tensor, model_output_shape)

    # save output
    # np.save("output_cyclone.npy", output_tensor)
    assert output_tensor.shape == model_output_shape

    assert np.abs(np.max(output_tensor) - 0.967444) < 1e-5
    assert np.abs(np.min(output_tensor) - 0.000507) < 1e-5
    assert np.abs(np.mean(output_tensor) - 0.0180078) < 1e-5
