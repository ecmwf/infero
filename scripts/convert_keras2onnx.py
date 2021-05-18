import os
import numpy as np
import argparse
import keras
import keras2onnx


if __name__ == "__main__":
    """
    Lightweight script to convert a keras model into a TFlite model
    """
    
    parser = argparse.ArgumentParser("Data Augmentation")
    parser.add_argument('keras_model_path', help="Path of the input keras model")
    parser.add_argument('onnx_model_path', help="Path of the output onnx model")
    parser.add_argument("--verify_with", help="Check the model by passing an input numpy path")
    
    args = parser.parse_args()

    # load the keras model
    model = keras.models.load_model(args.keras_model_path)
    model.summary()

    # do the conversion
    onnx_model = keras2onnx.convert_keras(model, model.name)
    
    # write to file
    file = open(args.onnx_model_path, "wb")
    file.write(onnx_model.SerializeToString())
    file.close()
