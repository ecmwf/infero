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
    Lightweight script to convert a keras model into a tf model
    """
    
    parser = argparse.ArgumentParser("Data Augmentation")
    parser.add_argument('keras_model_path', help="Path of the input keras model")
    parser.add_argument('tf_model_path', help="Path of the output tf model")
    
    args = parser.parse_args()
    
    # Load the model
    keras_model = keras.models.load_model(args.keras_model_path)

    # save the model
    print(f"Saving model in {args.tf_model_path}")
    tf.keras.models.save_model(keras_model, args.tf_model_path, save_format="tf")
 
