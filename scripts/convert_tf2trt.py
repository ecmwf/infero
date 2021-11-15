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
    Lightweight script to convert a tf model into a trt model
    """
    
    parser = argparse.ArgumentParser("Data Augmentation")
    parser.add_argument('tf_model_path', help="Path of the input tf model")
    parser.add_argument('trt_model_path', help="Path of the output trt model")
    
    args = parser.parse_args()
    
    # Load the model
    tf.experimental.tensorrt.Converter(
        input_saved_model_dir=args.tf_model_path, 
        #input_saved_model_tags=None,
        #input_saved_model_signature_key=None, 
        #use_dynamic_shape=None,
        #dynamic_shape_profile_strategy=None, 
        #conversion_params=None
    )
    converter.convert()
    converter.save(args.trt_model_path)

    # save the model
    print(f"Saving model in {args.trt_model_path}")
 
