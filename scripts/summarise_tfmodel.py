#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import argparse
import keras
import tensorflow as tf


if __name__ == "__main__":
    """
    Summarise a TF model
    """
    
    parser = argparse.ArgumentParser("Data Augmentation")
    parser.add_argument('tf_model_path', help="Path of the output tf model")
    
    args = parser.parse_args()
    
    # load the model
    print(f"Loading model from {args.tf_model_path}")

    model = tf.saved_model.load(args.tf_model_path)

    print(list(model.signatures.keys()))

    infer = model.signatures["serving_default"]
    print(f"infer.structured_outputs: {infer.structured_outputs}")

    print("INPUTS")
    for inp_t in infer.inputs:
        print(f"infer.inputs: {str(inp_t)}")

    print("OUTPUTS")
    for inp_t in infer.outputs:
        print(f"infer.outputs: {str(inp_t)}")
 
